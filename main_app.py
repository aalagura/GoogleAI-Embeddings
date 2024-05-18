import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Load the API key from the .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

genai.configure(api_key=GOOGLE_API_KEY)

text = "Hello World"
result = genai.embed_content(model=EMBEDDING_MODEL,content=text)

print(len(result['embedding']))

print(result)
