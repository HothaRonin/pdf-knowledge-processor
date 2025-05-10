from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.core.settings import Settings
import os

# Set global modules
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = LlamaCPP(
    model_path="C:/Users/joles/.lmstudio/models/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
    temperature=0.3,
    max_new_tokens=512,
    context_window=4096,
    generate_kwargs={"top_k": 40, "top_p": 0.95},
    model_kwargs={"n_gpu_layers": 33},
    verbose=True
)

# Build the index from your .txt files
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("output_texts").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Index built and saved.")
else:
    print("📁 Index already exists. Delete 'storage/' to rebuild.")
