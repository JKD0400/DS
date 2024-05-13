import streamlit as st

# Import necessary libraries from llama_index (replace placeholders with actual imports)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

# Import error logging and system interaction (if needed)
import logging
import sys

# Import libraries for embeddings (replace placeholders with actual imports)
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Other necessary imports (consider error handling, data loading, etc.)
import torch
import json
import os

# Function to configure logging (optional)
def configure_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Function to load data (replace with your data loading logic)
def load_data():
    documents = SimpleDirectoryReader("data_pdf").load_data()
    return documents

# Function to create the language model and embedder (replace placeholders)
def create_llm_and_embedder():
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin",
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="llama-2-7b-chat.Q2_K.gguf",
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 30},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    # embed_model = LangchainEmbedding(
    #     HuggingFaceEmbeddings(model_name="thenlper/gte-large")  # Replace with your model
    # )

    # loads BAAI/bge-small-en
    # embed_model = HuggingFaceEmbedding()

    # loads BAAI/bge-small-en-v1.5
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return llm, embed_model

# Function to create the service context
def create_service_context(llm, embed_model):
    service_context = ServiceContext.from_defaults(
        chunk_size=256, llm=llm, embed_model=embed_model
    )
    return service_context

# Function to create the vector store index
def create_index(documents, service_context):
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

# Function to handle chatbot response
def chatbot_response(text, index):
    query_engine = index.as_query_engine()
    response = query_engine.query(text)
    return response.response

def main():
    # Configure logging (optional)
    # configure_logging()

    # Load data (replace with your data loading logic)
    documents = load_data()

    # Create language model and embedder (replace placeholders)
    llm, embed_model = create_llm_and_embedder()

    # Create service context
    service_context = create_service_context(llm, embed_model)

    # Create vector store index
    index = create_index(documents, service_context)

    st.title("Medical Chatbot")

    query = st.text_input("Enter your query:")

    if query:
        response = chatbot_response(query, index)
        st.write(response)

if __name__ == "__main__":
    main()
