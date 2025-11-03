import os
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class Indexer:
    def __init__(self, file_path: str, urls: list = None, persist_dir: str = "./chroma_db", chunk_size: int = 250, chunk_overlap: int = 0):
        self.file_path = file_path
        self.urls = urls or []
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None

    def load_and_split(self):
        docs = []

        # Load local text files
        if os.path.isdir(self.file_path):
            for filename in os.listdir(self.file_path):
                full_path = os.path.join(self.file_path, filename)
                if os.path.isfile(full_path) and filename.lower().endswith(".txt"):
                    loader = TextLoader(full_path)
                    docs.extend(loader.load())
        else:
            loader = TextLoader(self.file_path)
            docs = loader.load()

        # Load documents from URLs
        for url in self.urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load URL {url}: {e}")

        # Split all documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def build_vectorstore(self, docs_splits):
        self.vectorstore = Chroma.from_documents(
            documents=docs_splits,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}),
            collection_name="my_text_docs",
            persist_directory=self.persist_dir
        )
        return self.vectorstore
