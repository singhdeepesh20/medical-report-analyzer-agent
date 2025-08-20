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


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)



def ingest_pdf(pdf_path: str, db_path: str = "faiss_index") -> FAISS:
    """
    Load a PDF, chunk its content, create embeddings, and store in FAISS DB.
    Returns the FAISS vectorstore.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ".", "?", "!", ",", " "]
    )
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local(db_path)
    return vectordb


def load_vectordb(db_path: str = "faiss_index") -> FAISS:
    """Reload FAISS index from local storage."""
    return FAISS.load_local(
        db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def build_medical_chain(vectordb: FAISS, groq_api_key: str):
    """
    Medical Report Analyzer Agent.
    """
    llm = ChatGroq(
        model="llama3-70b-8192",  # or "llama-3.1-70b-versatile" if available in your Groq account
        api_key=groq_api_key,
        temperature=0,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    medical_prompt = PromptTemplate(
        template=(
            "You are a careful, patient-friendly **medical AI assistant**.\n"
            "You will analyze a doctor's report or prescription from the provided context.\n\n"
            "Your responsibilities:\n"
            "1) Summarize the document in simple language.\n"
            "2) List *abnormal values* and why they may matter.\n"
            "3) Offer *possible interpretations* (no definitive diagnosis).\n"
            "4) Mention *red flags* that may need urgent attention.\n"
            "5) Suggest *questions to ask a doctor* and *next steps* (labs/lifestyle).\n\n"
            "Rules:\n"
            "- Do NOT provide medication dosages or replace clinical judgment.\n"
            "- If information is missing or unclear, state the uncertainty.\n"
            "- Keep it concise, structured, and readable for a non-expert.\n\n"
            "<REPORT CONTEXT>\n{context}\n</REPORT CONTEXT>\n\n"
            "Patient's Question: {question}\n\n"
            "Return your answer with clear sections: Summary, Findings, Concerns, Possible Interpretations, Red Flags, Next Steps, Questions for Doctor."
        ),
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": medical_prompt},
        return_source_documents=True,
    )
    return chain



def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, f"uploaded{suffix}")
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath


st.set_page_config(page_title="Medical Report Analyzer (Agent)", page_icon="🩺", layout="wide")

st.title("🩺 Medical Report Analyzer Agent")

st.markdown(
    ":information_source: **Disclaimer:** This AI provides educational insights only and does **not** replace professional medical advice, diagnosis, or treatment. Always consult a licensed clinician."
)

with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API Key", type="password", help="Create at console.groq.com")
    persist_dir = st.text_input("FAISS Index Folder", value="faiss_index", help="Where to store the vector index.")
    st.caption(f"Embeddings: {EMBEDDING_MODEL_NAME}")

uploaded = st.file_uploader("Upload a PDF medical report or prescription", type=["pdf"]) 

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_area(
        "What would you like to know?",
        value=(
            "Please analyze this medical report. Summarize, highlight abnormal values, possible interpretations, "
            "red flags, and questions to ask the doctor."
        ),
        height=120,
    )

with col2:
    st.markdown("**Quick Prompts**")
    if st.button("Summarize Report"):
        user_query = "Summarize this report for a patient in simple language."
    if st.button("Highlight Abnormal Values"):
        user_query = "List abnormal values and what they could indicate."
    if st.button("Red Flags & Next Steps"):
        user_query = "Identify red flags and suggest next steps to discuss with a doctor."

run = st.button("🔎 Analyze")


if run:
    if not groq_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    if not uploaded:
        st.error("Please upload a PDF report.")
        st.stop()

    with st.spinner("Processing and indexing your document..."):
        pdf_path = save_uploaded_file(uploaded)
        # Create a unique index folder per uploaded file name to avoid collisions
        base_index_dir = os.path.join(persist_dir, os.path.splitext(os.path.basename(uploaded.name))[0])
        os.makedirs(base_index_dir, exist_ok=True)

        # If index exists, load; else, ingest
        index_exists = os.path.exists(os.path.join(base_index_dir, "index.faiss"))
        try:
            if index_exists:
                vectordb = load_vectordb(base_index_dir)
            else:
                vectordb = ingest_pdf(pdf_path, db_path=base_index_dir)
        except Exception as e:
            st.error(f"Failed to build/load vector index: {e}")
            st.stop()

    with st.spinner("Building medical analysis chain and querying..."):
        try:
            chain = build_medical_chain(vectordb, groq_api_key=groq_key)
            response = chain({"query": user_query})
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()


    st.subheader("Result")
    st.write(response.get("result", "No response."))

    with st.expander("Sources (retrieved context)"):
        src_docs = response.get("source_documents", [])
        if not src_docs:
            st.caption("No source documents returned.")
        for i, d in enumerate(src_docs, 1):
            st.markdown(f"**Chunk {i}** — page {getattr(d.metadata, 'page', d.metadata.get('page', 'N/A'))}")
            st.code(d.page_content[:2000])

    st.success("Analysis complete. Remember to consult a licensed medical professional for decisions.")


