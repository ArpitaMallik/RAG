from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(len(docs))

# print(docs[30].page_content)
# print(docs[90].metadata)

docs = loader.load()

for document in docs:
    print(document.metadata)
