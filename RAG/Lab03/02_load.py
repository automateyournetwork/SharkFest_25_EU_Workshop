from langchain_community.document_loaders import JSONLoader

json_path = "capture.json"

print(f"📂 Loading JSON packets from {json_path}")
loader = JSONLoader(
    file_path=json_path,
    jq_schema=".[] | ._source.layers | del(.data)",
    text_content=False
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} packet documents")

print("\n--- First Document Content ---")
print(documents[0].page_content[:600])
print("\n📎 Metadata:")
print(documents[0].metadata)
