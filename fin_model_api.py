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
label_dict = {0: '간호사', 1: '진단서', 2: '빠르다', 3: '설사', 4: '가다', 5: '급하다', 6: '때리다', 
              7: '듣다', 8: '빨리', 9: '흔들리다', 10: '배부르다', 11: '숨차다', 12: '얼마', 13: '식도염', 
              14: '구급차', 15: '즐겁다', 16: '가깝다', 17: '경찰서', 18: '팔', 19: '싫다', 20: '사라지다', 
              21: '아프다', 22: '좋다', 23: '소화불량', 24: '친구', 25: '아들딸', 26: '쓰다', 27: '구조', 
              28: '불안', 29: '소화제', 30: '치아', 31: '과로', 32: '배고프다', 33: '목', 34: '괜찮다', 
              35: '마르다', 36: '피곤하다', 37: '아르바이트', 38: '쓰러지다', 39: '두근거리다', 40: '나', 
              41: '나빠지다', 42: '마을버스', 43: '검사', 44: '실수', 45: '감기', 46: '모르다', 47: '죄송', 
              48: '얼굴', 49: '콧물', 50: '변비', 51: '춥다', 52: '멀다', 53: '목마르다', 54: '잘못하다', 
              55: '부탁', 56: '무섭다', 57: '눈', 58: '서다', 59: '보건소', 60: '싫어하다', 61: '가쁘다', 
              62: '치료', 63: '우울', 64: '골절', 65: '감사', 66: '머리', 67: '가족', 68: '수면제', 
              69: '힘들다', 70: '입원', 71: '안타깝다', 72: '어떻게', 73: '오다', 74: '그렇다', 75: '떨다', 
              76: '생각못하다', 77: '가렵다', 78: '금식', 79: '장애인', 80: '오른쪽', 81: '원하다', 
              82: '통역사', 83: '신용카드', 84: '붕대', 85: '잊다', 86: '백수', 87: '차멀미', 88: '슬프다', 
              89: '병원', 90: '나쁘다', 91: '당뇨병', 92: '너', 93: '의사', 94: '왼쪽'}

# 학습된 모델 로드
model = tf.keras.models.load_model('final_model.keras')

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
    if prediction_confidence < 0.8:
        result = {"message": "다시 입력하세요"}
    else:
        result = {"predicted_word": predicted_word, "confidence": f"{prediction_confidence * 100:.2f}%"}

    # 임시 파일 삭제
    os.remove(temp_video_path)

    return JSONResponse(result)

# FastAPI 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
