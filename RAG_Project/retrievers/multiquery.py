from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

# Generate and print the queries

generated_queries = multi_query_retriever.llm_chain.invoke({"question": query})

print("\nGenerated Queries:\n")

for q in generated_queries:
    print(q)

# Retrieve documents

docs = multi_query_retriever.invoke(query)

print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)