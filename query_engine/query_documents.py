import sys
import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.core.settings import Settings

# Diagnostic print
print("PYTHON PATH:\n", sys.path)
print("PYTHON EXECUTABLE:\n", sys.executable)

# === CONFIGURATION ===
INPUT_DIR = "C:/Users/joles/Documents/pdf-knowledge-processor/input_pdfs"
OUTPUT_DIR = "C:/Users/joles/Documents/pdf-knowledge-processor/output_texts"
MODEL_PATH = "C:/Users/joles/.lmstudio/models/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf"

# === VALIDATION ===
if not os.path.exists(INPUT_DIR):
    raise ValueError(f"Input directory '{INPUT_DIR}' does not exist.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SETTINGS ===
Settings.llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_new_tokens=512,
    context_window=4096,
    generate_kwargs={"top_p": 0.95, "top_k": 40},
    model_kwargs={"n_gpu_layers": 35},
    verbose=True,
)

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === LOAD DOCS ===
documents = SimpleDirectoryReader(INPUT_DIR).load_data()

# === BUILD INDEX ===
index = VectorStoreIndex.from_documents(documents)

# === SAVE TO OUTPUT DIRECTORY ===
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print(f"✅ Index built and saved to {OUTPUT_DIR}")
