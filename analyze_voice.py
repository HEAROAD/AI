from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import json
import uvicorn

app = FastAPI()

@app.post("/analyze-voice/")
async def analyze_voice(file: UploadFile = File(...)):
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV file.")

    # Save the uploaded file to a temporary location
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    # Load the audio file
    y, sr = librosa.load(file_location, sr=None)

    # 주파수 기반 분석
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    mean_f0 = np.nanmean(f0)

    frequency_category = "저음" if 85 <= mean_f0 < 255 else "중음" if 255 <= mean_f0 < 500 else "고음"

    # 피치, 속도, 음량 및 스펙트럼 중심 분석
    pitch_category = "높음" if mean_f0 >= 300 else "중간" if 150 <= mean_f0 < 300 else "낮음"
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
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

    # JSON 결과 반환
    result = {
        "frequencyCategory": frequency_category,
        "pitchCategory": pitch_category,
        "speedCategory": speed_category,
        "volumeCategory": volume_category,
        "spectralCategory": spectral_category,
        "character": character
    }

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
