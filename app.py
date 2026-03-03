from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter






def load_all_documents():
    print("Scanning the 'data' folder for documents...")
    
    # The DirectoryLoader will look at every file (*.*) in the ./data folder
    # and use Unstructured behind the scenes to parse them.
    loader = DirectoryLoader('./data', glob="**/*.*", show_progress=True)
    
    # Think of this as returning a List<Document> in Java.
    documents = loader.load()
    
    print(f"\n✅ Successfully loaded {len(documents)} document(s)!")
    
    # Let's peek at the content of the first file it found to prove it worked
    if documents:
        print("\n--- Snippet of the first document ---")
        # We print just the first 300 characters so it doesn't flood your terminal
        print(documents[0].page_content[:300]) 
        print("\n-------------------------------------")
        print(f"Source file: {documents[0].metadata['source']}")

    return documents

def chunk_documents(documents):
    print(f"\n✂️ Splitting documents into bite-sized chunks...")
    
    # Think of this like batching rows in a PostgreSQL migration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Sliced the text into {len(chunks)} chunks.")
    
    # Let's peek at the first two chunks to see the overlap in action
    if len(chunks) > 1:
        print("\n--- Chunk 1 ---")
        print(chunks[0].page_content)
        print("\n--- Chunk 2 (Notice how it overlaps with the end of Chunk 1) ---")
        print(chunks[1].page_content[:300] + " ... [TRUNCATED]")
        
    return chunks

# --- UPDATE YOUR MAIN BLOCK ---
if __name__ == "__main__":
    # Step 1: Ingest
    docs = load_all_documents()
    
    # Step 2: Split
    chunks = chunk_documents(docs)