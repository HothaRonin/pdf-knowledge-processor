import sys
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.settings import Settings

print("PYTHON PATH:\n", sys.path)
print("PYTHON EXECUTABLE:\n", sys.executable)

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LlamaCPP model
llm = LlamaCPP(
    model_path=os.path.expanduser("~/.lmstudio/models/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf"),
    temperature=0.1,
    max_new_tokens=256,
    context_window=4096,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 32},
    verbose=True,
)

# Configure global settings
settings = Settings()
settings.llm = llm
settings.embed_model = embed_model

# Build the vector index
index = VectorStoreIndex.from_documents(documents, settings=settings)

# Create a query engine
query_engine = index.as_query_engine()

# Run a test query
response = query_engine.query("What is the document about?")
print(response)


