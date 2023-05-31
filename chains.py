import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import pandas as pd
import chromadb
from chromadb.config import Settings
from utils.handy import openai_ef
import pretty_errors

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
SAVE_PATH = os.getenv("SAVE_PATH")

"""
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=SAVE_PATH # Optional, defaults to .chromadb/ in the current directory
))
"""

#vectordb = client.get_collection(name="jobs_test", embedding_function=openai_ef)

docsearch = Chroma(collection_name="jobs_llm",persist_directory=SAVE_PATH, embedding_function=openai_ef)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.1
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever())

query = "I am a python developer, what job id do you recommend?"
qa.run(query)