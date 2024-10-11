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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
