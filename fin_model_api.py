import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile

# FastAPI 인스턴스 생성
app = FastAPI()

# 단어 리스트 (모델의 클래스 레이블)
word_list = ['운전면허', '바쁘다', '그립다', '권투', '힘', '골키퍼', '구경', '망가지다', '성토', '키우다', '썩다', '유도', 
             '남매', '여동생', '아내', '놀다', '원한', '누나', '견제하다', '낚시', '꿈', '견문', '울보', '장녀', '노래', 
             '마라톤', '상처', '운전면허정지', '딸', '엄마']

# 단어 레이블에 대한 인덱스 맵 생성
label_dict = {i: word for i, word in enumerate(word_list)}

# 학습된 모델 로드
model = tf.keras.models.load_model('model_fold_1.keras')

# MediaPipe Hands 초기화 (양손 인식)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 양손 인식
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

sequence_length = 240  # 사용할 시퀀스 길이 (예: 240 프레임 사용)

# MP4 비디오 파일에서 키포인트 추출 함수
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        keypoints = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손동작 키포인트 추출
                hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                keypoints.append(hand_keypoints)

        if len(keypoints) == 2:
            combined_keypoints = np.concatenate(keypoints)
        elif len(keypoints) == 1:
            combined_keypoints = np.concatenate([keypoints[0], np.zeros(63)])
        else:
            combined_keypoints = np.zeros(126)

        all_keypoints.append(combined_keypoints)

    cap.release()

    if len(all_keypoints) < sequence_length:
        all_keypoints += [np.zeros(126)] * (sequence_length - len(all_keypoints))

    return np.array(all_keypoints[:sequence_length])

# 비디오 파일을 처리하고 수어를 예측하는 FastAPI 엔드포인트
@app.post("/predict/")
async def predict_sign_language(file: UploadFile = File(...)):

    print(f"Received file: {file.filename}, Size: {file.size}")
    
    # 임시 파일에 비디오 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    # 키포인트 추출
    all_keypoints = extract_keypoints_from_video(temp_video_path)
    input_data = np.expand_dims(all_keypoints, axis=0)  # (1, 240, 126)

    # 모델 예측
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions)
    predicted_word = label_dict[predicted_label]
    prediction_confidence = predictions[0][predicted_label]

    # 80% 미만일 경우 다시 입력 요청
    if prediction_confidence < 0.95:
        result = {"message": "다시 입력하세요"}
    else:
        result = {"predicted_word": predicted_word, "confidence": f"{prediction_confidence * 100:.2f}%"}

    # 임시 파일 삭제
    os.remove(temp_video_path)

    return JSONResponse(result)

# FastAPI 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
