

# 🤖 RAG Chatbot – Multi-User Document-Aware Assistant

A powerful **Retrieval-Augmented Generation (RAG)** chatbot that enables multiple users to upload documents (PDFs) or crawl websites, index the content in **Elasticsearch**, and chat with the data using **Ollama LLMs** — all with session persistence via **Redis**.

---

## 🚀 Features

- 🔐 **Multi-User Authentication** (username-based)
- 💬 **Session Management** (create, switch, delete conversations)
- 🧠 **RAG Pipeline**
  - PDF ingestion & chunking
  - Webpage crawling & indexing
  - Embeddings with Ollama (`nomic-embed-text`)
  - Vector search powered by Elasticsearch
- 🗣️ **Conversational Memory** via Redis
- 🧩 **Customizable Parameters**
  - Chunk size, overlap, top_k, and temperature
- 🧹 **Document Management**
  - Upload, list, delete, or filter indexed documents
- 🌐 **Web Crawling** up to depth 2
- ⚙️ **Streamlit Interface** with clean UI

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit |
| **LLM** | Ollama (`phi3:mini`) |
| **Embeddings** | `nomic-embed-text` |
| **Vector Store** | Elasticsearch |
| **Memory Store** | Redis |
| **Document Loader** | LangChain PyPDFLoader |
| **Text Splitter** | RecursiveCharacterTextSplitter |

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/<your-username>/rag-chatbot.git

### Install Dependencies

pip install -r requirements.txt

### Start Required Services with Docker Compose

docker compose up -d

This will:
-🧠 Start Redis – used for storing chat memory and session data
-📦 Start Elasticsearch – used as the vector store for document embeddings
-🌐 Start Elasticvue – a web-based GUI for managing and exploring your Elasticsearch indices
-🧩 Connect to Ollama – runs locally to provide the LLM (phi3:mini) and embeddings (nomic-embed-text) for your RAG pipeline

To stop everything:

docker compose down

🧩 Ollama Setup

Make sure Ollama is installed and running locally, then pull your required models:
ollama pull phi3:mini
ollama pull nomic-embed-text

