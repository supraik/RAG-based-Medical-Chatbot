# test.py
from chatbot import HaleAI
bot = HaleAI()
bot.connect()

query = "vomiting blood"
docs = bot.vector_store.retrieve_documents(query)

print(f"Found {len(docs)} sources:")
for i, doc in enumerate(docs):
    print(f"\n--- Page {doc.metadata['page']} ---")
    print(doc.page_content[:200])