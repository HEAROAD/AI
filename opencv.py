import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# 모델 로드
model = tf.keras.models.load_model('sign_language_model.h5')

# 웹캠에서 비디오 캡처
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe로 손 키포인트 탐지
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand_points_list = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # 키포인트 추출 (21개 관절에 대한 x, y, z 좌표)
            hand_points = []
            for lm in hand_landmarks.landmark:
                hand_points.append([lm.x, lm.y, lm.z])
            
            hand_points_list.append(hand_points)
        
        # 양손 키포인트 결합 (2개의 손에 대한 데이터를 연결하여 (42, 3) 형태로)
        if len(hand_points_list) == 1:  # 한 손만 감지된 경우
            hand_points_list.append(np.zeros((21, 3)))  # 두 번째 손을 0으로 채움
            
        hand_points_combined = np.concatenate(hand_points_list, axis=0)  # (42, 3)
        hand_points_combined = np.expand_dims(hand_points_combined, axis=0)  # (1, 42, 3)
        
        # 예측 수행
        predictions = model.predict(hand_points_combined)
        predicted_class = np.argmax(predictions)
        
        # 결과 출력
        print(f"Predicted class: {predicted_class}")
    
    # 결과를 화면에 표시
    cv2.imshow('Sign Language Detection', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
