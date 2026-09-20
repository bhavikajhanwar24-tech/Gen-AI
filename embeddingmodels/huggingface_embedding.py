from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector = embedding.embed_query("Hello, Bhavika here")

print(vector)
print(len(vector))   