from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

json_path = "capture.json"

loader = JSONLoader(
    file_path=json_path,
    jq_schema=".[] | ._source.layers | del(.data)",
    text_content=False
)
documents = loader.load()
print(f"📦 Loaded {len(documents)} packet docs")

# --- Recursive split ---
print("\n🔹 Splitting with RecursiveCharacterTextSplitter")
recursive = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
r_chunks = recursive.split_documents(documents)
print(f"🔹 Recursive: {len(r_chunks)} chunks")

for i, c in enumerate(r_chunks[:2]):
    print(f"\n🧩 Recursive Chunk {i+1}:\n{c.page_content[:300]}...\n")

# --- Semantic split ---
print("\n🔸 Splitting with SemanticChunker (OpenAI embeddings)")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
semantic_splitter = SemanticChunker(embedding)
s_chunks = semantic_splitter.split_documents(documents)
print(f"🔸 Semantic: {s_chunks and len(s_chunks) or 0} chunks")

for i, c in enumerate(s_chunks[:2]):
    print(f"\n🧠 Semantic Chunk {i+1}:\n{c.page_content[:300]}...\n")
