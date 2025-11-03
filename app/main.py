from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

from app.core.personalized_rag import Personalized_RAG

app = FastAPI(title="Joel's Assistant API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    user_id: Optional[str] = None
    question: str

class AskResponse(BaseModel):
    answer: str
    documents: Optional[List[str]] = None


rag = Personalized_RAG(
    file_path="data/user_information/",
    user_id="default_user",
    persist_dir="./chroma_db",
)

@app.get("/")
def root():
    return {"message": "Welcome to Joel's Personalized RAG API 🚀"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest):
    try:
        # Generate user_id if not provided
        user_id = request.user_id or str(uuid4())
        rag.user_id = user_id

        # Process the question
        answer = rag.ask(request.question)

        # Retrieve the documents used in the retrieval step, if available
        docs_retrieved = getattr(rag.retriever, "last_retrieved_docs", [])

        return AskResponse(
            answer=answer,
            documents=[doc.page_content for doc in docs_retrieved] if docs_retrieved else [],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")


@app.post("/reindex")
def reindex_data():
    try:
        docs_splits = rag.indexer.load_and_split()
        rag.vectorstore = rag.indexer.build_vectorstore(docs_splits)
        return {"status": "success", "message": "Reindexing completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reindexing: {e}")



@app.options("/ask")
def options_ask():
    """Handle preflight OPTIONS request explicitly (for CORS)."""
    return {"status": "ok"}

