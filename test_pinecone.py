from pinecone import Pinecone, ServerlessSpec
print("Import successful! No deprecation errors.")
from dotenv import load_dotenv
import os
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Connected to Pinecone!")