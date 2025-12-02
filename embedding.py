import chromadb
import csv
import sys
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "./data"
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chromadb")
PREBUILT_CSV_PATH = os.path.join(DATA_DIR, "prebuilt-pc.csv")
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# --- LOAD EMBEDDING MODEL ---
def load_model():
    """Tải mô hình embedding SentenceTransformer."""
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    try:
        # Tự động chọn thiết bị (CUDA nếu có, nếu không thì CPU)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"FATAL: Could not load the model. Error: {e}")
        sys.exit(1)

def create_embeddings_from_csv(file_path, text_columns, metadata_columns, output_dir='data'):
    """
    Reads data from a CSV file, creates embeddings, and saves them.

    Args:
        file_path (str): Path to the input CSV file.
        text_columns (list): List of columns to combine for creating the text to be embedded.
        metadata_columns (list): List of all columns to be saved as metadata.
        output_dir (str): Directory to save the output files.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    df = pd.read_csv(file_path)
    df = df.dropna(subset=text_columns)
    df['combined_text'] = df[text_columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)

    documents = df['combined_text'].tolist()
    metadata = df[metadata_columns].to_dict(orient='records')

    return documents, metadata

def main():
    """Hàm chính để chạy quy trình embedding và ingestion."""
    model = load_model()

    # --- SETUP CHROMA DB ---
    if not os.path.exists(PREBUILT_CSV_PATH):
        print(f"Lỗi: Không tìm thấy file CSV trong thư mục '{DATA_DIR}'.")
        print("Vui lòng đảm bảo 'prebuilt-pc.csv' tồn tại.")
        sys.exit(1)

    print(f"Initializing ChromaDB client at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    prebuilt_collection = chroma_client.get_or_create_collection(name="prebuilt_pcs")

    # --- PROCESS PRE-BUILT PCs (prebuilt-pc.csv) ---
    print("\nProcessing pre-built PCs from prebuilt-pc.csv...")
    prebuilt_documents, prebuilt_ids, prebuilt_metadatas = [], [], []
    with open(PREBUILT_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(list(reader), desc="Reading pre-built PCs")):
            doc_parts = [
                f"Nhu cầu: {row.get('Nhu cầu')}",
                f"CPU: {row.get('Hãng CPU')} {row.get('CPU')}",
                f"MAIN: {row.get('MAIN')}",
                f"RAM: {row.get('RAM')}",
                f"VGA: {row.get('VGA')}",
                f"Storage: {row.get('Storage')}",
                f"PRICE: {row.get('PRICE')}"
            ]
            doc = ". ".join(filter(None, doc_parts))
            prebuilt_documents.append(doc)
            prebuilt_ids.append(f"prebuilt_{i+1}")
            meta = dict(row)
            link_value = meta.get('LINK') or meta.get('LINK SP') or ''
            meta['LINK'] = link_value
            prebuilt_metadatas.append(meta)

    print(f"Generating embeddings for {len(prebuilt_documents)} pre-built PCs...")
    prebuilt_embeddings = model.encode(prebuilt_documents, show_progress_bar=True)
    prebuilt_collection.upsert(embeddings=prebuilt_embeddings.tolist(), documents=prebuilt_documents, ids=prebuilt_ids, metadatas=prebuilt_metadatas)
    print(f"Successfully loaded {prebuilt_collection.count()} pre-built PCs into ChromaDB.")
    
    # Lưu trữ embeddings vào FAISS
    all_documents = prebuilt_documents
    all_metadata = prebuilt_metadatas

    if not all_documents:
        print("No documents to process. Exiting.")
        return

    print(f"Creating embeddings for {len(all_documents)} documents...")
    embeddings = model.encode(all_documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("\n✅ Embedding and ingestion complete!")

if __name__ == "__main__":
    main()