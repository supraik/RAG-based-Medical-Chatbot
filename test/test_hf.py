# test_hf.py
"""
Test Hugging Face LLM directly (NO RAG)
Compares raw model response vs HaleAI
"""
from llm_handler import LLMHandler
from config import LLM_MODEL

print("Testing Hugging Face LLM (NO RAG)...")
print("="*60)

# Initialize LLM handler
llm = LLMHandler()

# Test questions
questions = [
    "I have pain in my eyes"
    ]

for q in questions:
    print(f"\nQ: {q}")
    print("-" * 50)
    
    # Raw LLM response (no context)
    prompt = f"Question: {q}\nAnswer briefly:"
    try:
        raw_answer = llm.generate(prompt)
        print(f"RAW LLM: {raw_answer}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("-" * 50)