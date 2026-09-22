from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#RecursiveCharacterTextSplitter splits text intelligently using paragraphs, lines, and spaces 
#whereas TokenTextSplitter splits strictly based on token count to match LLM context limits.

data = PyPDFLoader("RAG_Project/document_loaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)