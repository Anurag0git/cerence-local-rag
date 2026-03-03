from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # NEW DB!

def load_all_documents():
    print("Scanning the 'data' folder for documents...")
    loader = DirectoryLoader('./data', glob="**/*.*", show_progress=True)
    documents = loader.load()
    print(f"✅ Successfully loaded {len(documents)} document(s)!")
    return documents

def chunk_documents(documents):
    print(f"\n✂️ Splitting documents into bite-sized chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Sliced the text into {len(chunks)} chunks.")
    return chunks

def create_vector_database(chunks):
    print("\n🧠 Loading the Embedding Model from LOCAL folder...")
    embeddings = HuggingFaceEmbeddings(model_name="./local_minilm_model")
    
    print("💾 Storing chunks into FAISS (Facebook AI Vector Database)...")
    # FAISS builds the index completely in memory first for extreme speed
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Then we save it to disk
    persist_directory = "./faiss_db"
    vectorstore.save_local(persist_directory)
    
    print(f"✅ Database created successfully! Data saved to '{persist_directory}'.")
    return vectorstore

if __name__ == "__main__":
    # 1. Ingest
    docs = load_all_documents()
    # 2. Split
    chunks = chunk_documents(docs)
    # 3. Store
    vectorstore = create_vector_database(chunks)
    
    # --- THE FINAL TEST: SEMANTIC SEARCH ---
    print("\n==================================================")
    print("🔍 RUNNING SEMANTIC SEARCH TEST")
    print("==================================================")
    
    # We ask a question related to your Confluence PDF
    query = "1. What are the two main components of the Transformer architecture?"
    print(f"Question: '{query}'\n")
    
    # k=2 means "Bring me the top 2 chunks that best answer this question"
    
    results = vectorstore.similarity_search(query, k=2)
    
    print("--- 🥇 Top Result Found in PDF ---")
    print(results[0].page_content)
    print("\n--- 🥈 Second Best Result ---")
    print(results[1].page_content)