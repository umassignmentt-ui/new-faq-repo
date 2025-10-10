# === Internal FAQ Assistant — Full Colab Script ===
# This file contains the complete code from your working Google Colab project.
# Includes PDF parsing, RAG pipeline (Hugging Face FLAN-T5), ML classifier, and Gradio interface.

!pip install -q --upgrade requests==2.32.5
!pip install -q langchain langchain-community faiss-cpu sentence-transformers pdfplumber gradio transformers accelerate scikit-learn

import os, pdfplumber, warnings, requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gradio as gr

warnings.filterwarnings("ignore")
print("✅ Environment ready — Requests version:", requests.__version__)

# ---------------- PDF Upload & Extraction ----------------
pdf_candidates = ["/mnt/data/CAIE Final Project.pdf", "/content/CAIE Final Project.pdf"]
pdf_path = None
for p in pdf_candidates:
    if os.path.exists(p):
        pdf_path = p
        break

if pdf_path is None:
    from google.colab import files
    print("Please upload your CAIE Project PDF")
    uploaded = files.upload()
    pdf_path = list(uploaded.keys())[0]

print("Using:", pdf_path)
pages = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        pages.append({"page": i+1, "text": text})
print(f"✅ Extracted text from {len(pages)} pages")

# ---------------- Text Chunking + Embedding ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
documents = []
for p in pages:
    for c in splitter.split_text(p["text"]):
        documents.append(Document(page_content=c, metadata={"page": p["page"]}))
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
print("✅ FAISS index built with Sentence-Transformers")

# ---------------- LLM (Hugging Face FLAN-T5) ----------------
rag_model = pipeline("text2text-generation", model="google/flan-t5-base", device_map="auto")
def generate_answer(question, context):
    prompt = f"Answer the question based only on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = rag_model(prompt, max_new_tokens=256, do_sample=False)
    return output[0]["generated_text"]
def rag_answer(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = generate_answer(question, context)
    pages = sorted({d.metadata["page"] for d in docs})
    return answer, pages

# ---------------- Classical ML: Question Classifier ----------------
training_data = [
    ("What is the leave policy?", "HR"),
    ("How to apply for annual leave?", "HR"),
    ("What are the project deliverables?", "Deliverables"),
    ("When is the report due?", "Deliverables"),
    ("Who is the project supervisor?", "General"),
    ("What is the objective of the project?", "General"),
]
X_train = [q for q, _ in training_data]
y_train = [label for _, label in training_data]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_vec, y_train)
def classify_question(question):
    X_q = vectorizer.transform([question])
    return clf.predict(X_q)[0]

# ---------------- Gradio Interface ----------------
def chatbot_fn(question):
    category = classify_question(question)
    answer, pages = rag_answer(question)
    page_str = ", ".join(str(p) for p in pages)
    return f"🔸 **Category:** {category}\n\n💬 **Answer:** {answer}\n\n📄 **Source pages:** {page_str}"

demo = gr.Interface(
    fn=chatbot_fn,
    inputs=gr.Textbox(placeholder="Ask about HR policies, deliverables, etc."),
    outputs="markdown",
    title="📚 SME Internal FAQ Assistant (Hugging Face + ML)",
    description="RAG chatbot built from CAIE PDF + question classification"
)

demo.launch(share=False)
