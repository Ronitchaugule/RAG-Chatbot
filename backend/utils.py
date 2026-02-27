import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(uploaded_file):
    # Save the uploaded file temporarily
    temp_path = "temp_upload.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load and split
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        return docs
    finally:
        # Clean up the file even if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)