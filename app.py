import warnings
import logging
import os

# ---------------------------------------------------------
# 🛑 SUPPRESS MESSY LOGS & WARNINGS 🛑
# ---------------------------------------------------------
warnings.filterwarnings("ignore")  # Ignores Pydantic and Deprecation warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)  # Silences the PDF FontBBox spam
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents huggingface warnings

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # 🚨 Updated to the new, warning-free package
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_all_documents():
    print("📄 Scanning the 'data' folder for documents...")
    # Turned off the progress bar for a cleaner look
    loader = DirectoryLoader('./data', glob="**/*.*", show_progress=False)
    documents = loader.load()
    return documents

def chunk_documents(documents):
    print("✂️  Splitting documents into bite-sized chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_database(chunks):
    print("🧠 Loading the offline Math Engine (Embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="./local_minilm_model")
    
    print("💾 Building the FAISS Vector Database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("./faiss_db")
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INITIALIZING SECURE ENTERPRISE RAG PIPELINE")
    print("="*60 + "\n")

    # 1. Build the Database
    docs = load_all_documents()
    chunks = chunk_documents(docs)
    vectorstore = create_vector_database(chunks)
    
    print("\n✅ Database ready! Waking up Llama 3.2...\n")
    
    # 2. Setup the Brain
    llm = OllamaLLM(model="llama3.2")
    
    template = """Use ONLY the following pieces of context to answer the question at the end. 
    If you don't know the answer or if the context doesn't contain it, just say "I don't know based on the provided documents."
    Do not make up an answer. Keep the answer concise and professional.

    Context:
    {context}

    Question: {input}
    Answer:"""
    
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "input"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 3. Create the Chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    
    # 4. Execute Chat
    print("="*60)
    query = "What are the two main components of the Transformer architecture?"
    print(f"🗣️  USER: {query}")
    print("="*60)
    
    print("\n🤖 AI is reading the documents and thinking...\n")
    response = rag_chain.invoke(query)
    
    print("✨ FINAL ANSWER ✨")
    print(response)
    print("\n" + "="*60)