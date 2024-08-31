from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import uvicorn
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import os
import shutil
import json

app = FastAPI()

# 현재 파일이 위치한 디렉토리 경로 얻기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델과 레이블 인코더 로드
model_path = os.path.join(BASE_DIR, 'best_model.keras')
model = load_model(model_path)
label_encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# 레이블 리스트 로드
labels_list_path = os.path.join(BASE_DIR, 'labels_list.json')
with open(labels_list_path, 'r', encoding='utf-8') as f:
    labels_list = json.load(f)

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
    predicted_label = np.argmax(np.mean(predictions, axis=0))  # 각 프레임의 예측값을 평균내어 최종 예측 결정
    predicted_word = labels_list[predicted_label]  # labels_list에서 해당 예측값에 매칭되는 텍스트 찾기

    # 임시 파일 삭제
    os.remove(video_path)

    # 결과를 텍스트로 반환
    return PlainTextResponse(predicted_word)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
