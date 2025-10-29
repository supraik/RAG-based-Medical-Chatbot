# test_generic_rag.py
from chatbot import HaleAI
from llm_handler import LLMHandler

bot = HaleAI()
bot.connect()
llm = LLMHandler()

q = "I am vomiting blood. What should I do?"

print("RAW LLM:")
print(llm.generate(q)[:200])
print("\n" + "="*60 + "\n")

print("GENERIC RAG:")
resp = bot.query(q)
print(resp["answer"])