from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# FastAPI 앱 생성
app = FastAPI()

# 요청 데이터 모델 정의
class GlossRequest(BaseModel):
    gloss: list[str]  # 수어 단어들의 리스트


# POST 요청 처리 - OpenAI API와 통신
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
