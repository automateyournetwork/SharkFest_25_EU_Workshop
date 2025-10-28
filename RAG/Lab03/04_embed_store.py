from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid

json_path = "capture.json"

loader = JSONLoader(
    file_path=json_path,
    jq_schema=".[] | ._source.layers | del(.data)",
    text_content=False
)
docs = loader.load()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = SemanticChunker(embedding)
chunks = splitter.split_documents(docs)

print(f"🧩 Total semantic chunks: {len(chunks)}")

# --- In-memory store ---
print("\n💾 Creating in-memory Chroma store ...")
mem_db = Chroma.from_documents(chunks, embedding)
print("📥 In-memory store ready ✅")

data = mem_db._collection.get()
print(f"📊 Stored {len(data['documents'])} docs → sample ID {data['ids'][:1]}")

# --- Persistent store ---
session_id = str(uuid.uuid4())
persist_dir = f"chroma_pcap_{session_id}"

print(f"\n💽 Persisting to {persist_dir}")
persist_db = Chroma.from_documents(chunks, embedding, persist_directory=persist_dir)
persist_db.persist()
print("✅ Persistent DB created")
