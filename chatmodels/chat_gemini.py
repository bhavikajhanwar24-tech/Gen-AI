from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

response = model.invoke("write a poem on AI")

print(response.content[0]["text"])