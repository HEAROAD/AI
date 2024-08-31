from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import os
import shutil

app = FastAPI()

# 모델과 레이블 인코더 로드
model = load_model('C:/Users/USER/Documents/GitHub/AI/best_model.keras')
with open('C:/Users/USER/Documents/GitHub/AI/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 비디오에서 키포인트 추출 함수
def extract_keypoints_from_video(video_path):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        result_pose = pose.process(frame_rgb)
        if result_pose.pose_landmarks:
            pose_keypoints = []
            for lm in result_pose.pose_landmarks.landmark:
                pose_keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pose_keypoints = [0] * 132  # 33 keypoints * 4 (x, y, z, visibility)

        # Hand detection
        result_hands = hands.process(frame_rgb)
        left_hand_keypoints = [0] * 84
        right_hand_keypoints = [0] * 84

        if result_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
                hand_keypoints = []
                for lm in hand_landmarks.landmark:
                    hand_keypoints.extend([lm.x, lm.y, lm.z, 1])  # 1 is for confidence, as MediaPipe doesn't provide it

                if handedness.classification[0].label == 'Left':
                    left_hand_keypoints = hand_keypoints
                else:
                    right_hand_keypoints = hand_keypoints

        keypoints = np.concatenate([pose_keypoints, left_hand_keypoints, right_hand_keypoints])
        keypoints_list.append(keypoints)

    cap.release()
    return np.array(keypoints_list)

# 키포인트 리스케일 함수
def rescale_keypoints(keypoints, image_width, image_height):
    keypoints[:, 0::4] *= image_width  # x 좌표 스케일 조정
    keypoints[:, 1::4] *= image_height  # y 좌표 스케일 조정
    return keypoints

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    # 파일 저장
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 비디오에서 키포인트 추출 및 처리
    keypoints = extract_keypoints_from_video(video_path)
    keypoints = rescale_keypoints(keypoints, image_width=640, image_height=480)
    keypoints = keypoints / np.max(keypoints)  # 데이터 정규화
    keypoints = keypoints.reshape((keypoints.shape[0], keypoints.shape[1], 1))  # 모델 입력 형식에 맞게 reshape

    # 예측 수행
    predictions = model.predict(keypoints)
    predicted_label = int(np.argmax(np.mean(predictions, axis=0)))  # int로 변환하여 JSON 직렬화 가능하도록 설정
    predicted_word = label_encoder.inverse_transform([predicted_label])[0]

    # 임시 파일 삭제
    os.remove(video_path)

    # 결과 반환
    return JSONResponse(content={"predicted_word": str(predicted_word)})  # 문자열로 변환하여 반환

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
