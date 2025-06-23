import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# Step 1: Load PDF files individually
DATA_PATH = "Pdf_books/"

def load_pdf_files(data):
    pdf_files = glob(os.path.join(data, "*.pdf"))
    documents = []

    print(f"🔍 Found {len(pdf_files)} PDF(s) in {data}")
    for path in tqdm(pdf_files):
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed to load {os.path.basename(path)}: {e}")
    
    return documents

documents = load_pdf_files(DATA_PATH)
print("📄 Total documents loaded:", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

text_chunks = create_chunks(documents)
print("✂️ Total text chunks created:", len(text_chunks))


# Step 3: Embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()
print("📌 Embedding model loaded.")


# Step 4: Store in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"✅ FAISS DB saved to: {DB_FAISS_PATH}")
