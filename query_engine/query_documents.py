import os
import sys
print("PYTHON PATH:\n", sys.path)
print("PYTHON EXECUTABLE:\n", sys.executable)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding

# Path to your local GGUF model
MODEL_PATH = "/Users/joles/.lmstudio/models/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf"

# Set up the LLM (local)
llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_new_tokens=512,
    context_window=4096,
    generate_kwargs={"top_p": 0.95, "top_k": 40},
    model_kwargs={"n_gpu_layers": 100, "n_ctx": 4096}
)

# Use a local HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Service context for LlamaIndex
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=512,
    chunk_overlap=64,
)

# Load documents from local folder
documents = SimpleDirectoryReader("output_texts").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Interactive loop
query_engine = index.as_query_engine()
while True:
    query = input("\nAsk a question about your documents: ")
    if query.lower() in {"exit", "quit"}:
        break
    response = query_engine.query(query)
    print(f"\nResponse:\n{response}")

