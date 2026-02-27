import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

class RAGEngine:
    def __init__(self):
        # Uses local Ollama - no HuggingFace dependency
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatGroq(
            temperature=0,  
            model_name="llama-3.1-8b-instant",
            groq_api_key="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"
        )
        self.vectorstore = None

    def create_vectorstore(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def ask(self, query):
        if not self.vectorstore:
            return "Please upload a PDF first."

        # 1. Manual Retrieval: Get top 3 relevant snippets
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Manual Augmentation: Create the final prompt
        prompt = f"""
        You are a WPIntelliChat Business Analyst. Use the context below to answer.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""

        # 3. Generation: Get response directly from the LLM
        response = self.llm.invoke(prompt)
        return response.content