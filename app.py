import streamlit as st
from backend.rag_engine import RAGEngine
from backend.utils import load_and_split_pdf

st.set_page_config(page_title="WPIntelliChat Analytics", layout="wide")

# Persistent state for Engine and History
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar UI
with st.sidebar:
    st.title("🕒 Chat History")
    st.markdown("---")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.caption(f"Q: {msg['content'][:50]}...")
    
    if st.button("Clear All"):
        st.session_state.messages = []
        st.rerun()

# Main Application
st.title("📄 WPIntelliChat RAG Engine")

uploaded_file = st.file_uploader("Upload Business Data (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing document..."):
        docs = load_and_split_pdf(uploaded_file)
        st.session_state.rag.create_vectorstore(docs)
    st.success("Analysis Engine Ready!")

# Display current conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Ask about sales, finance, or HR trends..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        response = st.session_state.rag.ask(query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)