from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import os
import audioread

app = FastAPI()

import random

@app.get("/random-voice-analysis/")
async def random_voice_analysis():
    # 랜덤 값 생성
    frequency_category = random.choice(["저음", "중음", "고음"])
    pitch_category = random.choice(["높음", "중간", "낮음"])
    speed_category = random.choice(["빠름", "보통", "느림"])
    volume_category = random.choice(["높음", "보통", "낮음"])
    spectral_category = random.choice(["밝은 목소리", "동굴 목소리"])
    
    # 캐릭터 결정
    if pitch_category == "높음" and speed_category == "빠름" and spectral_category == "밝은 목소리":
        character = "귀여운 다람쥐"
    elif pitch_category == "중간" and volume_category == "보통" and spectral_category == "밝은 목소리":
        character = "명랑한 새"
    elif pitch_category == "낮음" and volume_category == "낮음" and spectral_category == "동굴 목소리":
        character = "고요한 곰"
    elif frequency_category == "저음" and spectral_category == "동굴 목소리":
        character = "중후한 사자"
    elif pitch_category == "낮음" and speed_category == "보통" and spectral_category == "동굴 목소리":
        character = "개구리"
    else:
        character = "평범한 고양이"

    result = {
        "frequencyCategory": frequency_category,
        "pitchCategory": pitch_category,
        "speedCategory": speed_category,
        "volumeCategory": volume_category,
        "spectralCategory": spectral_category,
        "character": character
    }

    return JSONResponse(content=result)


@app.post("/analyze-voice/")
async def analyze_voice(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}, Content Type: {file.content_type}")

    if not file.filename.endswith(('.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV or M4A file.")

    # 파일을 서버의 임시 파일로 저장
    temp_dir = os.path.dirname(__file__)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        with audioread.audio_open(file_path) as input_file:
            y, sr = librosa.load(input_file, sr=None)

        # 오디오 파일 분석
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        mean_f0 = np.nanmean(f0)

        frequency_category = "저음" if 85 <= mean_f0 < 255 else "중음" if 255 <= mean_f0 < 500 else "고음"
        pitch_category = "높음" if mean_f0 >= 300 else "중간" if 150 <= mean_f0 < 300 else "낮음"
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # 수정된 부분
        speed_category = "빠름" if tempo > 150 else "보통" if 110 <= tempo <= 150 else "느림"
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)
        volume_category = "높음" if mean_rms > 0.1 else "보통" if 0.05 <= mean_rms <= 0.1 else "낮음"
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroids)
        spectral_category = "밝은 목소리" if mean_centroid >= 2000 else "동굴 목소리"

        # 캐릭터 결정
        if pitch_category == "높은 피치" and speed_category == "빠른 속도" and spectral_category == "밝은 목소리":
            character = "귀여운 다람쥐"
        elif pitch_category == "중간 피치" and volume_category == "보통 음량" and spectral_category == "밝은 목소리":
            character = "명랑한 새"
        elif pitch_category == "낮은 피치" and volume_category == "낮은 음량" and spectral_category == "동굴 목소리":
            character = "고요한 곰"
        elif frequency_category == "저음" and spectral_category == "동굴 목소리":
            character = "중후한 사자"
        elif pitch_category == "낮은 피치" and speed_category == "보통 속도" and spectral_category == "동굴 목소리":
            character = "개구리"
        else:
            character = "평범한 고양이"

        result = {
            "frequencyCategory": frequency_category,
            "pitchCategory": pitch_category,
            "speedCategory": speed_category,
            "volumeCategory": volume_category,
            "spectralCategory": spectral_category,
            "character": character
        }

        return JSONResponse(content=result)
    finally:
        # 처리 후 임시 파일 삭제
        os.remove(file_path)

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile

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

#########
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')


# 요청 데이터 모델 정의
class GlossRequest(BaseModel):
    gloss: list[str]  # 수어 단어들의 리스트


###############POST 요청 처리 - OpenAI API와 통신
@app.post("/generate_sentence")
async def generate_sentence(request: GlossRequest):
    try:
        # 수어 단어들을 하나의 문자열로 결합
        gloss_sentence = ' '.join(request.gloss)

        # OpenAI API 요청
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 모델
            messages=[
                {
                    "role": "system",
                    "content": "Convert the given list of words into a grammatically correct and natural-sounding sentence in Korean without adding any extra meaning or information. Both input and output will be in Korean.",
                },
                {
                    "role": "user",
                    "content": f"[{gloss_sentence}]",
                },
            ],
        )

        # 응답에서 생성된 문장 추출
        generated_sentence = completion.choices[0].message['content'].strip()

        # 응답을 "answer" 키로 반환
        return {"answer": generated_sentence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
