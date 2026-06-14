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


# CONFIGURATIOn
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
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


# LLM CHAIN
def build_medical_chain(vectordb: FAISS, groq_api_key: str):
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=groq_api_key,
        temperature=0,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": get_medical_prompt()},
        return_source_documents=True,
    )

# STREAMLIT UI
def main():
    st.set_page_config(
        page_title="Medical Report Analyzer Agent",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🩺 Medical Report Analyzer Agent")

    st.info(
        "This AI provides educational insights only and does NOT replace professional medical advice."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        groq_key = st.text_input("Groq API Key", type="password")
        persist_dir = st.text_input("FAISS Index Folder", value=DEFAULT_INDEX_DIR)
        st.caption(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

 # Upload
    uploaded = st.file_uploader(
        "Upload a PDF medical report or prescription",
        type=["pdf"],
    )
 # Query Section
    col1, col2 = st.columns([2, 1])

    default_query = (
        "Please analyze this medical report. Summarize, highlight abnormal values, "
        "possible interpretations, red flags, and questions to ask the doctor."
    )

    with col1:
        user_query = st.text_area(
            "What would you like to know?",
            value=default_query,
            height=120,
        )

    with col2:
        st.markdown("### Quick Prompts")
        if st.button("Summarize Report"):
            user_query = "Summarize this report for a patient in simple language."
        elif st.button("Highlight Abnormal Values"):
            user_query = "List abnormal values and explain what they may indicate."
        elif st.button("Red Flags & Next Steps"):
            user_query = "Identify urgent concerns and recommended next steps."

     if st.button("🔎 Analyze"):
        if not groq_key:
            st.error("Please enter your Groq API Key.")
            return

        if not uploaded:
            st.error("Please upload a PDF report.")
            return

        try:
            with st.spinner("Processing document..."):
                pdf_path = save_uploaded_file(uploaded)
                index_dir = os.path.join(
                    persist_dir,
                    Path(uploaded.name).stem,
                )

                vectordb = get_or_create_vectordb(pdf_path, index_dir)

            with st.spinner("Analyzing report..."):
                chain = build_medical_chain(vectordb, groq_key)
                response = chain({"query": user_query})

            st.subheader("Analysis Result")
            st.write(response.get("result", "No response generated."))

            with st.expander("Retrieved Source Context"):
                for i, doc in enumerate(response.get("source_documents", []), start=1):
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"**Chunk {i} — Page {page}**")
                    st.code(doc.page_content[:2000])

            st.success(
                "Analysis complete. Consult a licensed healthcare professional for medical decisions."
            )

        except Exception as e:
            st.error(f"Application Error: {str(e)}")


if __name__ == "__main__":
    main()

