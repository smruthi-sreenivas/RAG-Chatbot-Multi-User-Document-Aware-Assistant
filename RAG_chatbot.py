
import streamlit as st
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import redis
import tempfile
from datetime import datetime
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from elasticsearch import Elasticsearch

# Configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
INDEX_NAME = "rag_documents"

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .user-info {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session Management Functions
def get_user_sessions(username):
    """Get all session IDs for a user from Redis"""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        user_sessions_key = f"user_sessions:{username}"
        sessions = redis_client.smembers(user_sessions_key)
        return [s.decode('utf-8') for s in sessions] if sessions else []
    except:
        return []

def save_user_session(username, session_id, session_name):
    """Save a session ID for a user in Redis"""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        user_sessions_key = f"user_sessions:{username}"
        redis_client.sadd(user_sessions_key, session_id)
        
        session_metadata_key = f"session_metadata:{session_id}"
        redis_client.hset(session_metadata_key, mapping={
            "username": username,
            "session_name": session_name,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        })
        return True
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False

def update_session_activity(session_id):
    """Update last activity timestamp for a session"""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        session_metadata_key = f"session_metadata:{session_id}"
        redis_client.hset(session_metadata_key, "last_active", datetime.now().isoformat())
    except:
        pass

def get_session_metadata(session_id):
    """Get metadata for a session"""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        session_metadata_key = f"session_metadata:{session_id}"
        metadata = redis_client.hgetall(session_metadata_key)
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()} if metadata else None
    except:
        return None

def delete_session(username, session_id):
    """Delete a session"""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        
        user_sessions_key = f"user_sessions:{username}"
        redis_client.srem(user_sessions_key, session_id)
        
        session_metadata_key = f"session_metadata:{session_id}"
        redis_client.delete(session_metadata_key)
        
        message_key = f"message_store:{session_id}"
        redis_client.delete(message_key)
        
        return True
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "checked_existing_index" not in st.session_state:
    st.session_state.checked_existing_index = False
if "chat_history_loaded" not in st.session_state:
    st.session_state.chat_history_loaded = False
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "top_k" not in st.session_state:
    st.session_state.top_k = 4
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

class RAGChatbot:
    def __init__(self, session_id, temperature=0.7):
        self.llm = OllamaLLM(model="phi3:mini", temperature=temperature)
        
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        self.vector_store = None
        self.es_url = ELASTICSEARCH_URL
        self.index_name = INDEX_NAME
        
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL)
            self.redis_client.ping()
            self.message_history = RedisChatMessageHistory(
                session_id=session_id,
                url=REDIS_URL
            )
        except Exception as e:
            st.error(f"Redis connection failed: {e}")
            self.message_history = None
        
        if self.message_history:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                chat_memory=self.message_history,
                return_messages=True,
                output_key="answer"
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        self.qa_chain = None
        
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant analyzing documents. Use the following context to answer the question accurately and concisely.

Context from documents:
{context}

Previous conversation:
{chat_history}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain the answer, say so clearly
- Be specific and cite information from the documents when possible
- Keep responses clear and well-structured

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
    
    def _initialize_vector_store(self):
        if self.vector_store is None:
            try:
                # FIXED: Added timeout configuration
                es_client = Elasticsearch(
                    [self.es_url],
                    request_timeout=300,  # 5 minutes
                    max_retries=3,
                    retry_on_timeout=True
                )

                if not es_client.indices.exists(index=self.index_name):
                    # Create index with proper mapping for ES 7.x
                    index_body = {
                        "mappings": {
                            "properties": {
                                "text": {"type": "text"},
                                "vector": {
                                    "type": "dense_vector",
                                    "dims": 768  # nomic-embed-text dimension
                                },
                                "metadata": {"type": "object"}
                            }
                        }
                    }
                    es_client.indices.create(index=self.index_name, body=index_body)
                    st.info(f"✅ Created index: {self.index_name}")

                # Works with both ES 7.x and 8.x
                self.vector_store = ElasticsearchStore(
                    es_url=self.es_url,
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    es_connection=es_client
                )
                print("✅ Connected to Elasticsearch vector store")
            except Exception as e:
                st.error(f"Elasticsearch connection failed: {e}")
                raise
    
    def load_and_index_pdfs(self, pdf_files, chunk_size=1000, chunk_overlap=200, progress_callback=None):
        all_documents = []
        indexed_filenames = []
        
        # FIXED: Added try-except wrapper
        try:
            st.write(f"📄 Starting to load {len(pdf_files)} PDF file(s)...")
            
            for idx, pdf_file in enumerate(pdf_files):
                if progress_callback:
                    progress_callback(idx, len(pdf_files), pdf_file.name)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    filename = pdf_file.name
                    st.write(f"   ✅ Loaded {len(documents)} page(s) from {filename}")
                    
                    for doc in documents:
                        doc.metadata["source"] = filename
                        doc.metadata["file_path"] = tmp_path
                        doc.metadata["indexed_date"] = datetime.now().isoformat()
                        doc.metadata["file_size"] = len(pdf_file.getvalue())
                    
                    all_documents.extend(documents)
                    indexed_filenames.append(filename)
                except Exception as e:
                    st.error(f"❌ Error loading {pdf_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            if not all_documents:
                st.error("No documents were loaded")
                return [], 0
            
            st.write("✂️ Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(all_documents)
            st.write(f"   ✅ Created {len(splits)} chunk(s)")
            
            self._initialize_vector_store()
            
            # FIXED: Reduced batch size from 50 to 10 for CPU performance
            st.write("🔄 Indexing chunks (this may take a few minutes on CPU)...")
            batch_size = 10
            total_batches = (len(splits) + batch_size - 1) // batch_size
            
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                
                st.write(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                try:
                    self.vector_store.add_documents(batch)
                    st.write(f"   ✅ Batch {batch_num} completed")
                except Exception as e:
                    st.error(f"❌ Error in batch {batch_num}: {e}")
                    continue
            
            st.success(f"🎉 Successfully indexed {len(splits)} chunks!")
            return indexed_filenames, len(splits)
            
        except Exception as e:
            st.error(f"❌ Fatal error during indexing: {e}")
            import traceback
            st.code(traceback.format_exc())
            return [], 0
    
    def create_qa_chain(self, metadata_filter=None, top_k=4):
        self._initialize_vector_store()
        search_kwargs = {"k": top_k}
        
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        
        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": self.prompt_template}
        )
        
        return self.qa_chain
    
    def query(self, question, metadata_filter=None, top_k=4):
        self.create_qa_chain(metadata_filter, top_k)
        response = self.qa_chain.invoke({"question": question})
        
        return {
            "answer": response["answer"],
            "source_documents": response["source_documents"]
        }
    
    def clear_history(self):
        if self.message_history:
            self.message_history.clear()
    
    def get_indexed_files(self):
        """Get list of indexed files from Elasticsearch"""
        try:
            # Don't initialize vector store, just query ES directly
            es_client = Elasticsearch(
                [self.es_url],
                request_timeout=10  # Short timeout for quick check
            )
            
            # Quick ping test
            if not es_client.ping():
                return []
            
            # Check if index exists
            if not es_client.indices.exists(index=self.index_name):
                return []
            
            # Get unique sources
            query = {
                "size": 0,
                "aggs": {
                    "unique_sources": {
                        "terms": {
                            "field": "metadata.source.keyword",
                            "size": 100
                        }
                    }
                }
            }
            
            response = es_client.search(index=self.index_name, body=query)
            
            # Check if aggregations exist
            if "aggregations" in response and "unique_sources" in response["aggregations"]:
                sources = [bucket["key"] for bucket in response["aggregations"]["unique_sources"]["buckets"]]
                return sources
            
            return []
            
        except Exception as e:
            # Fail silently and return empty list
            print(f"Error getting indexed files: {e}")
            return []
    
    def delete_documents_by_source(self, source_filename):
        try:
            self._initialize_vector_store()
            es_client = Elasticsearch(
                [self.es_url],
                request_timeout=60
            )
            
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": source_filename
                    }
                }
            }
            
            es_client.delete_by_query(index=self.index_name, body=query)
            return True
        except Exception as e:
            st.error(f"Error deleting documents: {e}")
            return False
    
    def get_index_stats(self):
        try:
            self._initialize_vector_store()
            es_client = Elasticsearch(
                [self.es_url],
                request_timeout=30
            )
            
            if not es_client.indices.exists(index=self.index_name):
                return None
            
            stats = es_client.count(index=self.index_name)
            return stats.get("count", 0)
        except:
            return None
    
    def get_chat_history(self):
        """Retrieve chat history from Redis for display in UI"""
        try:
            if not self.message_history:
                return []
            
            messages = self.message_history.messages
            formatted_messages = []
            
            for msg in messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = "user" if msg.type == "human" else "assistant"
                    formatted_messages.append({
                        "role": role,
                        "content": msg.content,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            
            return formatted_messages
        except Exception as e:
            st.warning(f"Could not load chat history: {e}")
            return []
    
    def crawl_and_index_webpage(self, url, max_depth=1, progress_callback=None):
        """Crawl a webpage and index its content"""
        visited_urls = set()
        documents = []
        base_url = url
        
        def extract_text_from_html(html_content, url):
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            title = soup.title.string if soup.title else urlparse(url).path
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r' +', ' ', text)
                return title, text
            
            return title, ""
        
        def get_links(soup, base_url):
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    links.append(full_url)
            
            return links
        
        def create_source_name(url, title):
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = re.sub(r'\s+', '_', clean_title.strip())
            clean_title = clean_title[:50]
            
            if clean_title and clean_title.lower() not in ['home', 'index', 'main']:
                return f"web_{clean_title}"
            
            if path and path != '/':
                clean_path = path.strip('/').replace('/', '_')
                clean_path = re.sub(r'[^\w-]', '_', clean_path)
                clean_path = clean_path[:50]
                return f"web_{clean_path}"
            
            return f"web_{domain}"
        
        def crawl_recursive(url, depth=0):
            if depth > max_depth or url in visited_urls:
                return
            
            visited_urls.add(url)
            
            if progress_callback:
                progress_callback(len(visited_urls), url)
            
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                title, text = extract_text_from_html(response.content, url)
                
                if text:
                    page_source = create_source_name(url, title)
                    
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": page_source,
                            "url": url,
                            "title": title,
                            "indexed_date": datetime.now().isoformat(),
                            "content_type": "webpage",
                            "depth": depth,
                            "domain": urlparse(url).netloc
                        }
                    )
                    documents.append(doc)
                
                if depth < max_depth:
                    links = get_links(soup, url)
                    for link in links[:10]:
                        crawl_recursive(link, depth + 1)
            
            except Exception as e:
                st.warning(f"Failed to crawl {url}: {str(e)}")
        
        crawl_recursive(url, 0)
        
        if not documents:
            return None, 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        self._initialize_vector_store()
        
        # FIXED: Reduced batch size
        batch_size = 10
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            self.vector_store.add_documents(batch)
        
        source_name = create_source_name(base_url, documents[0].metadata['title'] if documents else "webpage")
        return source_name, len(splits)

# Authentication UI
if not st.session_state.authenticated:
    st.title("🔐 Login to RAG Chatbot")
    
    tab1, tab2 = st.tabs(["Login", "About"])
    
    with tab1:
        st.subheader("Enter your username")
        st.caption("For demo purposes, any username works. In production, add proper authentication.")
        
        username_input = st.text_input("Username", key="login_username")
        
        if st.button("Login", type="primary"):
            if username_input.strip():
                st.session_state.authenticated = True
                st.session_state.username = username_input.strip()
                st.rerun()
            else:
                st.error("Please enter a username")
    
    with tab2:
        st.markdown("""
        ### Multi-User RAG Chatbot
        
        **Features:**
        - 🔐 User-based session management
        - 💾 Persistent conversation history
        - 📚 Document indexing with Elasticsearch
        - 🔄 Multiple conversation sessions per user
        - 🗑️ Delete and manage sessions
        
        **How it works:**
        1. Login with a username
        2. Create or select a conversation session
        3. Upload and index documents
        4. Ask questions about your documents
        5. Switch between sessions seamlessly
        
        **⚠️ Note:** Running on CPU. Indexing takes ~30s per chunk.
        """)
    
    st.stop()

# Main Application
st.title(f"🤖 RAG Chatbot - Welcome, {st.session_state.username}!")

# Sidebar
with st.sidebar:
    st.markdown(f"<div class='user-info'>👤 <strong>{st.session_state.username}</strong></div>", unsafe_allow_html=True)
    
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.session_state.chatbot = None
        st.rerun()
    
    st.divider()
    st.subheader("💬 Conversation Sessions")
    
    user_sessions = get_user_sessions(st.session_state.username)
    
    with st.expander("➕ New Session", expanded=not user_sessions):
        new_session_name = st.text_input("Session Name", placeholder="e.g., Q4 Reports Analysis")
        if st.button("Create Session"):
            if new_session_name.strip():
                import uuid
                new_session_id = f"{st.session_state.username}_{uuid.uuid4()}"
                if save_user_session(st.session_state.username, new_session_id, new_session_name):
                    st.session_state.current_session_id = new_session_id
                    st.session_state.current_session_name = new_session_name
                    st.session_state.messages = []
                    st.session_state.chatbot = None
                    st.session_state.chat_history_loaded = False
                    st.session_state.checked_existing_index = False
                    st.success(f"Created session: {new_session_name}")
                    st.rerun()
            else:
                st.error("Please enter a session name")
    
    if user_sessions:
        st.caption(f"You have {len(user_sessions)} session(s)")
        
        for session_id in user_sessions:
            metadata = get_session_metadata(session_id)
            if metadata:
                session_name = metadata.get('session_name', 'Unnamed Session')
                created_at = metadata.get('created_at', 'Unknown')
                
                is_active = st.session_state.current_session_id == session_id
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        button_label = f"{'✅' if is_active else '💬'} {session_name}"
                        if st.button(button_label, key=f"session_{session_id}", use_container_width=True):
                            st.session_state.current_session_id = session_id
                            st.session_state.current_session_name = session_name
                            st.session_state.messages = []
                            st.session_state.chatbot = None
                            st.session_state.chat_history_loaded = False
                            st.session_state.checked_existing_index = False
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"delete_{session_id}"):
                            if delete_session(st.session_state.username, session_id):
                                if st.session_state.current_session_id == session_id:
                                    st.session_state.current_session_id = None
                                    st.session_state.messages = []
                                    st.session_state.chatbot = None
                                    st.session_state.chat_history_loaded = False
                                    st.session_state.checked_existing_index = False
                                st.rerun()
                    
                    if is_active:
                        st.caption(f"🕒 Active • Created: {created_at[:10]}")
    
    if st.session_state.current_session_id:
        st.divider()
        st.title("📚 Document Management")
        
        # DISABLED: Automatic check causes blank screens
        # Just mark as checked to skip this step
        if not st.session_state.checked_existing_index:
            st.session_state.checked_existing_index = True
        
        # Show existing files from session state
        if st.session_state.indexed_files:
            st.info(f"📚 {len(st.session_state.indexed_files)} document(s) in this session")
        
        with st.expander("⚙️ Advanced Settings"):
            st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size, 100)
            st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 500, st.session_state.chunk_overlap, 50)
            st.session_state.top_k = st.slider("Top K Results", 1, 10, st.session_state.top_k)
            st.session_state.temperature = st.slider("LLM Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
            
            if st.button("Apply Settings"):
                st.session_state.chatbot = None
                st.success("Settings updated!")
        
        tab1, tab2 = st.tabs(["📄 PDF Documents", "🌐 Web Pages"])
        
        with tab1:
            uploaded_files = st.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True, key="pdf_uploader")
            
            if uploaded_files and st.button("📥 Index PDFs", type="primary"):
                if len(uploaded_files) > 4:
                    st.error("Please upload maximum 4 PDF files")
                else:
                    # FIXED: Better error handling
                    with st.container():
                        try:
                            if st.session_state.chatbot is None:
                                st.session_state.chatbot = RAGChatbot(
                                    st.session_state.current_session_id,
                                    st.session_state.temperature
                                )
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(current, total, filename):
                                progress = (current + 1) / total
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {filename}")
                            
                            indexed_files, num_chunks = st.session_state.chatbot.load_and_index_pdfs(
                                uploaded_files,
                                chunk_size=st.session_state.chunk_size,
                                chunk_overlap=st.session_state.chunk_overlap,
                                progress_callback=update_progress
                            )
                            
                            st.session_state.indexed_files = list(set(st.session_state.indexed_files + indexed_files))
                            progress_bar.empty()
                            status_text.empty()
                            
                        except Exception as e:
                            st.error(f"❌ Error during indexing: {e}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
        
        with tab2:
            st.caption("Enter a URL to crawl and index its content")
            
            url_input = st.text_input(
                "Website URL",
                placeholder="https://example.com",
                help="Enter the full URL including https://"
            )
            
            crawl_depth = st.slider(
                "Crawl Depth",
                min_value=0,
                max_value=2,
                value=0,
                help="0 = Single page, 1 = Include linked pages, 2 = Two levels deep"
            )
            
            if url_input and st.button("🌐 Crawl & Index", type="primary"):
                if not url_input.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    with st.container():
                        try:
                            if st.session_state.chatbot is None:
                                st.session_state.chatbot = RAGChatbot(
                                    st.session_state.current_session_id,
                                    st.session_state.temperature
                                )
                            
                            progress_text = st.empty()
                            
                            def update_crawl_progress(count, url):
                                progress_text.text(f"Crawled {count} page(s)... Current: {url[:50]}...")
                            
                            source_name, num_chunks = st.session_state.chatbot.crawl_and_index_webpage(
                                url_input,
                                max_depth=crawl_depth,
                                progress_callback=update_crawl_progress
                            )
                            
                            if source_name:
                                st.session_state.indexed_files = list(set(st.session_state.indexed_files + [source_name]))
                                progress_text.empty()
                                st.success(f"✅ Successfully indexed website: {source_name} ({num_chunks} chunks)")
                            else:
                                progress_text.empty()
                                st.error("No content was extracted from the website")
                        
                        except Exception as e:
                            st.error(f"❌ Error crawling website: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
            
            st.divider()
            st.caption("💡 Tips:")
            st.caption("• Depth 0: Only the specified page")
            st.caption("• Depth 1: Main page + linked pages from same domain")
            st.caption("• Depth 2: Goes two levels deep (may take longer)")
        
        if st.session_state.indexed_files:
            st.subheader("📄 Indexed Documents")
            
            if st.session_state.chatbot:
                total_chunks = st.session_state.chatbot.get_index_stats()
                if total_chunks:
                    st.metric("Total Chunks", total_chunks)
            
            for file in st.session_state.indexed_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file}")
                with col2:
                    if st.button("🗑️", key=f"delete_doc_{file}"):
                        if st.session_state.chatbot:
                            if st.session_state.chatbot.delete_documents_by_source(file):
                                st.session_state.indexed_files.remove(file)
                                st.rerun()
        
        st.divider()
        st.subheader("🔍 Filter Documents")
        filter_enabled = st.checkbox("Enable metadata filter")
        selected_files = []
        
        if filter_enabled and st.session_state.indexed_files:
            selected_files = st.multiselect("Select documents", st.session_state.indexed_files)
    else:
        st.info("👈 Please create or select a session to continue")

# Main chat interface
if st.session_state.current_session_id:
    st.caption(f"Session: **{st.session_state.current_session_name}** • ID: `{st.session_state.current_session_id[:16]}...`")
    
    update_session_activity(st.session_state.current_session_id)
    
    if not st.session_state.chat_history_loaded:
        with st.spinner("Loading chat history..."):
            try:
                if st.session_state.chatbot is None:
                    st.session_state.chatbot = RAGChatbot(
                        st.session_state.current_session_id,
                        st.session_state.temperature
                    )
                
                history = st.session_state.chatbot.get_chat_history()
                if history:
                    st.session_state.messages = history
                    st.info(f"📜 Loaded {len(history)} previous messages")
            except Exception as e:
                st.warning(f"Could not load chat history: {e}")
            
            st.session_state.chat_history_loaded = True
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"• **{source}**")
            if "timestamp" in message:
                st.caption(f"🕒 {message['timestamp']}")
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.indexed_files:
            st.warning("⚠️ Please upload and index documents first!")
        else:
            try:
                if st.session_state.chatbot is None:
                    st.session_state.chatbot = RAGChatbot(
                        st.session_state.current_session_id,
                        st.session_state.temperature
                    )
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": timestamp
                })
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                    st.caption(f"🕒 {timestamp}")
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            metadata_filter = None
                            if 'filter_enabled' in locals() and filter_enabled and selected_files:
                                if len(selected_files) == 1:
                                    metadata_filter = [{"term": {"metadata.source.keyword": selected_files[0]}}]
                                else:
                                    metadata_filter = [{"terms": {"metadata.source.keyword": selected_files}}]
                            
                            response = st.session_state.chatbot.query(
                                prompt,
                                metadata_filter,
                                st.session_state.top_k
                            )
                            
                            st.markdown(response["answer"])
                            
                            sources = list(set([doc.metadata["source"] for doc in response["source_documents"]]))
                            if sources:
                                with st.expander("📚 Sources"):
                                    for source in sources:
                                        st.markdown(f"• **{source}**")
                            
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.caption(f"🕒 {timestamp}")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": sources,
                                "timestamp": timestamp
                            })
                            
                            update_session_activity(st.session_state.current_session_id)
                            
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            with st.expander("🔍 Error Details"):
                                import traceback
                                st.code(traceback.format_exc())
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg,
                                "sources": [],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
            except Exception as e:
                st.error(f"❌ Critical error: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
else:
    st.info("👈 Create or select a session from the sidebar to start chatting!")

st.divider()
st.caption("🤖 Multi-User RAG Chatbot with Session Management • Powered by Phi3 & Elasticsearch")