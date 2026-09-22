from langchain_community.document_loaders import ArxivLoader

# create the retriever
loader = ArxivLoader(
    query="large language models",
    load_max_docs=2,      # number of papers to retrieve
    load_all_available_meta=True
)

# query arxiv
docs = loader.load()

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:")  
    print(doc.page_content[:500]) # print first 500 characters