import os
import tempfile
from typing import Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


# CONFIGURATION
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 5
LLM_MODEL = "llama3-70b-8192"


#CACHED RESOURCES
@st.cache_resource
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load embedding model once per session."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", "?", "!", ",", " "],
    )



#FILE HANDLING
def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded PDF to temporary storage."""
    suffix = Path(uploaded_file.name).suffix
    tmp_dir = tempfile.mkdtemp()
    file_path = Path(tmp_dir) / f"uploaded{suffix}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)



# VECTOR DATABASE
def ingest_pdf(pdf_path: str, db_path: str) -> FAISS:
    """Load PDF, split into chunks, embed, and store in FAISS."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    vectordb = FAISS.from_documents(chunks, get_embedding_model())
    vectordb.save_local(db_path)

    return vectordb


def load_vectordb(db_path: str) -> FAISS:
    """Load existing FAISS index."""
    return FAISS.load_local(
        db_path,
        embeddings=get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def get_or_create_vectordb(pdf_path: str, index_dir: str) -> FAISS:
    """Reuse existing index or create new one."""
    index_file = Path(index_dir) / "index.faiss"

    if index_file.exists():
        return load_vectordb(index_dir)

    os.makedirs(index_dir, exist_ok=True)
    return ingest_pdf(pdf_path, index_dir)


# PROMPT ENGINEERING
def get_medical_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=(
            "You are a careful, patient-friendly medical AI assistant.\n"
            "Analyze the provided medical report or prescription.\n\n"
            "Responsibilities:\n"
            "1. Summarize in simple patient-friendly language\n"
            "2. Highlight abnormal findings and significance\n"
            "3. Provide possible interpretations (not diagnosis)\n"
            "4. Mention urgent red flags\n"
            "5. Suggest next steps and doctor discussion points\n\n"
            "Rules:\n"
            "- Do NOT replace licensed medical advice\n"
            "- Do NOT prescribe medications/dosages\n"
            "- Clearly state uncertainty when present\n"
            "- Use structured sections\n\n"
            "<REPORT_CONTEXT>\n{context}\n</REPORT_CONTEXT>\n\n"
            "Patient Question: {question}\n\n"
            "Format:\n"
            "Summary\n"
            "Findings\n"
            "Concerns\n"
            "Possible Interpretations\n"
            "Red Flags\n"
            "Next Steps\n"
            "Questions for Doctor"
        ),
        input_variables=["context", "question"],
    )

