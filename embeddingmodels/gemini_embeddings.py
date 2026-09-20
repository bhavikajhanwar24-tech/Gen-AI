from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

texts = [
    "Hello this is Bhavika",
    "Hello your name is XYZ",
    "And you all are very beautiful"
]

vectors = embeddings.embed_documents(texts) # embed_documents for multiple text and for single text embed_query()

print(len(vectors))      # 3
print(len(vectors[0]))   # embedding dimension
print(vectors)