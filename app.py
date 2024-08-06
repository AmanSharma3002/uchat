__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import asyncio
import os
import tempfile
from scraping import scrape_website
from retrieval import query_retrieval
from embedding import CustomEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Load configuration
def load_configuration():
    load_dotenv()
    groq_api_key = os.getenv("API_KEY")
    hf_token = os.getenv("HF")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    return groq_api_key, api_url, headers

groq_api_key, api_url, headers = load_configuration()

# Set up Streamlit page
st.title("Web Page Scraper and Chatbot")

# Create a temporary directory and initialize session state
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'db' not in st.session_state:
    st.session_state.db = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

def clear_chat_and_db():
    st.session_state.chat_history = []
    if st.session_state.db:
        st.session_state.db.delete_collection()
        st.session_state.db = None
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

# Scrape and Process Form
with st.sidebar.form(key="scrape_form", clear_on_submit=True):
    url_input = st.text_area("Enter URLs (separated by commas)", key="url_input")
    scrape_and_process_button = st.form_submit_button("Scrape and Process")

if scrape_and_process_button:
    if url_input:
        urls = [url.strip() for url in url_input.split(",") if url.strip()]
        if urls:
            st.write("Scraping URLs...")
            docs = asyncio.run(scrape_website(urls))
            if docs:
                st.write(f"Fetched {len(docs)} documents")
                embeddings = CustomEmbeddings(api_url, headers)
                db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=st.session_state.temp_dir)
                st.session_state.db = db
                st.write(f"Number of documents in the collection: {db._collection.count()}")
            else:
                st.write("No documents fetched.")
        else:
            st.write("Please enter valid URLs.")
    else:
        st.write("Please enter at least one URL.")

# Clear Chat and Database Buttons
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    st.rerun()

if st.sidebar.button("Clear Chat and Database"):
    clear_chat_and_db()
    st.rerun()

# Custom CSS for chat alignment
st.markdown("""
    <style>
    .user-prompt {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        text-align: right;
        color: black;
    }
    .llm-response {
        background-color: #E4E6EB;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        text-align: left;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

st.write("### Chat History")
for entry in st.session_state.chat_history:
    st.markdown(f'<div class="user-prompt">**Q:** {entry["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="llm-response">**A:** {entry["answer"]}</div>', unsafe_allow_html=True)

# Ask Question Form
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_area("Your question", key="query_input")
    submit_button = st.form_submit_button(label="Ask")

if submit_button:
    if query:
        if st.session_state.db:
            answers = query_retrieval(st.session_state.db, query, groq_api_key, st.session_state.memory)
            if answers:
                st.session_state.chat_history.append({"question": query, "answer": answers['answer']})
                st.rerun()
            else:
                st.write("No answers found.")
        else:
            st.write("Please scrape and process URLs first.")
    else:
        st.write("Please enter a question.")

