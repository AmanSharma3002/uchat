import os
from dotenv import load_dotenv

def load_configuration():
    load_dotenv()
    groq_api_key = os.getenv("API_KEY")
    hf_token = os.getenv("HF")
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    return groq_api_key, api_url, headers
