import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# --- Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "DATA"
GROQ_MODEL_NAME = "gemma2-9b-it"

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Embeddings Model ---
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# --- Streamlit App ---
st.set_page_config(page_title="🩺 Ask MediBot", layout="wide")
st.title("🩺 Ask MediBot")

# Session state for multi-session chat history
if "store" not in st.session_state:
    st.session_state.store = {}

session_id = st.text_input("Session ID", value="default_session")

uploaded_files = st.file_uploader(
    "Upload optional prescription/report/book PDFs (you can upload multiple):",
    type=["pdf"],
    accept_multiple_files=True
)

@st.cache_resource
def load_vectorstore(uploaded_files=None):
    if not os.path.exists(DB_FAISS_PATH):
        with st.spinner("Processing base PDFs..."):
            loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            db = FAISS.from_documents(chunks, embedding_model)
            db.save_local(DB_FAISS_PATH)
    else:
        with st.spinner("Loading vector store..."):
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    if uploaded_files:
        with st.spinner("Processing uploaded PDFs..."):
            new_docs = []
            for uploaded_file in uploaded_files:
                # Save UploadedFile to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
                loader = PyPDFLoader(tmp_file_path)
                new_docs.extend(loader.load())
                # Optionally delete temp file here if you want cleanup (not mandatory)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            new_chunks = splitter.split_documents(new_docs)
            user_db = FAISS.from_documents(new_chunks, embedding_model)
            db.merge_from(user_db)  # merge new uploaded docs into existing db

    return db

# Load or update vectorstore with optional uploaded PDFs
vectorstore = load_vectorstore(uploaded_files)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt for contextualizing the question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME),
                                                          retriever=retriever,
                                                          prompt=contextualize_q_prompt)

# Prompt for final QA
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know. "
    "Keep the answer concise (max three sentences).\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create retrieval + answer chain
question_answer_chain = create_stuff_documents_chain(ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME), qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to manage chat history
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Create conversational chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Get session chat history
session_history = get_session_history(session_id)

# User Input Chat UI
user_input = st.chat_input("Ask your medical question...")

if user_input:
    # Add user message *before* rendering loop
    #session_history.add_user_message(user_input)

    with st.spinner("Thinking..."):
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

    # Add assistant response
    #session_history.add_ai_message(response["answer"])

# Render entire history once, including newly added messages
for message in session_history.messages:
    role = "user" if message.type == "human" else "assistant"
    st.chat_message(role).markdown(message.content)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Created by Priyank 💡</div>", unsafe_allow_html=True)
