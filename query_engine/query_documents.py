import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, LLMPredictor
from llama_index.llms import OpenAI
from getpass import getpass

# Load API key or replace with local LLM-compatible LLM class
api_key = getpass("Enter your OpenAI API key (or replace with local LLM): ")
llm_predictor = LLMPredictor(llm=OpenAI(api_key=api_key, temperature=0.1, model="gpt-3.5-turbo"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

documents = SimpleDirectoryReader("documents").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()

while True:
    user_query = input("\nAsk a question about your documents: ")
    response = query_engine.query(user_query)
    print(f"\nResponse:\n{response}")
