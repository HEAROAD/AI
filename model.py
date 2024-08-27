import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 데이터 로드 예시
def load_data(json_directory):
    data = []
    labels = []
    for file_name in os.listdir(json_directory):
        if file_name.endswith('.json'):
            with open(os.path.join(json_directory, file_name), 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                keypoints = json_data['keypoints']
                
                # 각 키포인트 데이터를 numpy 배열로 변환
                for keypoint in keypoints:
                    left_hand = np.array(keypoint['hand_left_keypoints_2d']).reshape(-1, 3)
                    right_hand = np.array(keypoint['hand_right_keypoints_2d']).reshape(-1, 3)
                    
                    # 데이터와 레이블 추가
                    data.append(np.concatenate([left_hand, right_hand], axis=0))
                    labels.append(json_data['word'])
    
    return np.array(data), np.array(labels)

# 간단한 LSTM 모델 예시
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# 데이터 로드
json_directory = '/Users/simmee/Documents/GitHub/AI/output_01'
X, y = load_data(json_directory)

# 데이터 형태 출력
print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 레이블 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 원핫 인코딩
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# 모델 구성 및 학습
model = create_model((X.shape[1], X.shape[2]), num_classes=y_onehot.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_onehot, epochs=5, validation_split=0.2)

# 모델 저장
label_map = {str(i): label for i, label in enumerate(label_encoder.classes_)}
with open('/Users/simmee/Documents/GitHub/AI/label_map.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)
