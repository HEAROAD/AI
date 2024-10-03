import os
os.environ['JAVA_HOME'] = "/Users/songle/Library/Java/JavaVirtualMachines/corretto-21.0.3/Contents/Home"

from fastapi import FastAPI, Request
from konlpy.tag import Okt

app = FastAPI()
okt = Okt()


@app.post("/extract_keywords")
async def extract_keywords(request: Request):
    data = await request.json()
    message = data.get("message", "")

    if isinstance(message, list):
        message = " ".join(message)  # 리스트를 공백으로 구분해 문자열로 변환

    if not message:
        return {"keywords": "No message provided"}

    try:
        # 품사 분석 수행
        pos_tags = okt.pos(message, norm=True)
        print(f"POS Tags: {pos_tags}")  # 디버그 출력

        # 명사(Noun), 동사(Verb), 형용사(Adjective) 추출
        keywords = [word for word, tag in pos_tags if tag in ('Noun', 'Verb', 'Adjective')]

        return {"keywords": ", ".join(keywords)}
    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return {"keywords": "Keyword Extraction Failed"}
