# 🧠 Internal FAQ Assistant for SMEs

This project is a lightweight **Internal FAQ Assistant** designed for **Small and Medium Enterprises (SMEs)**.  
It allows teams to **upload their internal documents** (such as employee handbooks, policy PDFs, or manuals) and **query them in natural language** using an AI chatbot interface.

The system is built entirely using **open-source tools**, making it free to run without OpenAI API costs.

---

## 🚀 Features

- 📄 **PDF Ingestion** — Upload company handbooks or internal docs to build a private knowledge base.  
- 🔍 **RAG (Retrieval-Augmented Generation)** — Uses embeddings and vector search to find the most relevant sections before answering.  
- 🤖 **Hugging Face LLM** — Uses FLAN-T5 and Sentence Transformers to generate context-aware answers without relying on OpenAI APIs.  
- 🧠 **Complementary ML Component** — Logistic Regression model used to classify questions into categories for more structured responses.  
- 🌐 **Gradio Interface** — User-friendly chat interface deployed directly from Google Colab.  
- 🧪 **Sample Data** — Comes with a small example CSV file to test without uploading real data.

---

## 🧰 Tech Stack

- **Language Model**: Hugging Face FLAN-T5  
- **Embeddings**: Sentence Transformers  
- **Vector Database**: FAISS  
- **ML Model**: scikit-learn Logistic Regression  
- **Interface**: Gradio  
- **PDF Processing**: pdfplumber  
- **LangChain**: for RAG pipeline orchestration

---

## 🧪 Sample Data

The [`sample_data/sample_faq.csv`](sample_data/sample_faq.csv) file contains a few example FAQ entries you can use to test the chatbot without uploading your own documents.

You can also upload any company policy PDFs through the interface in the notebook.

---

## 📓 Run on Google Colab

You can run this project entirely in Google Colab — no local installation required.

1. Open the notebook in Google Colab.  
2. Upload your PDF file (e.g., employee handbook).  
3. Run all cells.  
4. When the Gradio link appears, click it to open the chatbot in a new tab.  
5. Start asking questions!

---

## 💻 Run Locally (Optional)

You can also run this project on your local machine.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/umassignmentt-ui/new-faq-repo.git
cd new-faq-repo
