from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import TOP_K
from src.pipeline import answer_question

app = FastAPI(title="RAG Documents Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=10)


@app.get("/")
def home():
    return {"message": "RAG Documents Assistant API", "docs": "/docs"}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_question(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
