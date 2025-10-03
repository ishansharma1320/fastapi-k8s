from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from transformers import AutoTokenizer, MarianMTModel
from inference_translate import translate
import os, uvicorn, traceback

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "Helsinki-NLP/opus-mt-en-hi")
model = MarianMTModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


app = FastAPI()

@app.post("/translate")
def get_translation(request: dict):
    try:
        text = request.get("text", None)
        if text is None:
            raise HTTPException(status_code=400, detail="Text `text` parameter is missing.")
        translated_text = translate(model, tokenizer, text)
        return {"response": translated_text}
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.get("/")
def health_check():
    return "PONG"

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)