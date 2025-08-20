# Medical-report-analyzer-agent

Medical Report Analyzer Agent

An AI-powered medical assistant that helps patients and caregivers understand PDF-based medical reports (e.g., blood tests, prescriptions, diagnostic results).

This project uses LangChain, FAISS, HuggingFace embeddings, and Groq LLMs to analyze reports, highlight abnormal values, and suggest patient-friendly insights — while not replacing professional medical advice.

✨ Features

* Upload Reports: Upload any medical report or prescription in PDF format.

* Smart Analysis: AI automatically summarizes reports in plain language.

* Abnormal Values: Detects unusual test values and explains their significance.

* Red Flags: Highlights critical findings that may require urgent attention.

* Next Steps: Suggests possible lifestyle changes, labs, or questions for your doctor.

* Groq-powered AI: Ultra-fast responses using Llama 3 (70B) via Groq API.

* FAISS Indexing: Report chunks are embedded & stored locally for efficient retrieval.

🛠️ Tech Stack

* Frontend: Streamlit

* LLM: Groq
 (llama3-70b-8192)

* Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

* Vector DB: FAISS

* Framework: LangChain


Usage Guide

* Upload your medical report (PDF).

* Type your question or select from quick prompts:

* Summarize report

* Highlight abnormal values

* Identify red flags

* Click 🔎 Analyze.

*View results + retrieved context (chunks from your document)


📈 Applications in the Medical Field

This project can be applied in multiple areas of healthcare:

🧑‍🤝‍🧑 Patient Education: Helps patients understand complex test results in simple language.

🩸 Blood & Lab Report Insights: Quickly identifies abnormal lab values and possible causes.

🏥 Hospital Support: Can be integrated into hospital systems to provide first-level explanations for reports.

📑 Prescription Review: Summarizes doctor prescriptions to ensure patients understand the treatment plan.

🌍 Rural Healthcare: Useful in regions with low doctor-patient ratio — patients can at least understand their reports before consulting a clinician.

🧑‍⚕️ Doctor’s Assistant: Doctors can use it as a quick report summarization tool during consultations.

📚 Medical Training: Medical students can practice report interpretation with AI support.

⚠️ Important: It is only an assistive tool — not a replacement for licensed healthcare professionals.

⚠️ Disclaimer

This project is for educational and informational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a licensed clinician for health decisions.

📌 Example Use Case

* Upload a blood report PDF → Get:

* Simple summary in patient-friendly language

* Abnormal parameters (e.g., “Hemoglobin low”)

* Why it matters (e.g., “May indicate anemia”)

* Red flags (urgent findings)

* Suggested questions to ask your doctor

























