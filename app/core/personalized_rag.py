import os
import app.config
from app.core.indexer import Indexer
from app.core.retriever import Retriever
from app.core.llm_agent import LLM_Agent
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class Personalized_RAG:
    def __init__(self, file_path: str, user_id: str = "terminal_user", persist_dir: str = "./chroma_db", urls: list = None):
        self.file_path = file_path
        self.urls = urls
        self.persist_dir = persist_dir
        self.user_id = user_id
        
        # Initialize indexer
        self.indexer = Indexer(file_path=self.file_path, persist_dir=self.persist_dir, urls=urls)
        
        # Check if vectorstore already exists
        if self.is_indexed():
            print("Loading existing index...")
            # Load existing Chroma collection
            self.vectorstore = Chroma(
                collection_name="my_text_docs",
                persist_directory=self.persist_dir,
                embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
        else:
            print("Indexing documents...")
            docs_splits = self.indexer.load_and_split()
            self.vectorstore = self.indexer.build_vectorstore(docs_splits)

        # Initialize retriever and agent
        self.retriever = Retriever(self.vectorstore)
        self.agent = LLM_Agent()
        print("Personalized_RAG is ready!")

    def is_indexed(self):
        # Simple check: directory exists and has some files
        return os.path.exists(self.persist_dir) and any(os.scandir(self.persist_dir))

    def ask(self, question: str):
        docs_retrieved = self.retriever.retrieve(question)
        answer = self.agent.ask(self.user_id, question, docs_retrieved)
        return answer


if __name__ == "__main__":
    rag = Personalized_RAG(file_path="user_information/")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = rag.ask(question)
        print("Assistant:", answer)
        print("-" * 50)
