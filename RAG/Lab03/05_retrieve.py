from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

json_path = "capture.json"

# --- Load + Embed ---
loader = JSONLoader(file_path=json_path, jq_schema=".[] | ._source.layers", text_content=False)
docs = loader.load()
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
chunks = SemanticChunker(embedding).split_documents(docs)

# --- Store in Chroma ---
vector_store = Chroma.from_documents(chunks, embedding)
print("📦 Vector store ready")

# --- Method 1: Direct search ---
questions = [
    "Which protocols are in this capture?",
    "Any DNS queries?",
    "Are there HTTPS sessions?"
]
for q in questions:
    print(f"\n❓ Q: {q}")
    results = vector_store.similarity_search(q, k=2)
    for i, doc in enumerate(results):
        print(f"\n📄 Match {i+1}:\n{doc.page_content[:300]}...\n")

# --- Method 2: Conversational RAG ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k":5}),
    return_source_documents=True
)

response = qa.invoke({"question": "Summarize the capture traffic types", "chat_history": []})
print(f"\n🧠 Answer:\n{response['answer']}\n")
