# list_models.py
import google.generativeai as genai

API_KEY = ""
# Replace with your actual API key 
# or set the environment variable GOOGLE_API_KEY
# Set the API key for Google Generative AI

genai.configure(api_key=API_KEY)
print("Available models and supported methods:\n")
for m in genai.list_models():                   # ← list_models() is a generator
    methods = ", ".join(m.supported_generation_methods)
    print(f"{m.name:<30}  supports → {methods}")
