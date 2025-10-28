import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import tempfile, os, subprocess, json

st.set_page_config(page_title="Packet Copilot Chat", page_icon="🔍")
st.title("🔍 Chat with Your Packet Capture")

uploaded = st.file_uploader("Upload PCAP file", type=["pcap","pcapng"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(uploaded.read())
        pcap_path = tmp.name
    json_path = pcap_path + ".json"

    st.info("Running tshark conversion…")
    subprocess.run(f'tshark -nlr "{pcap_path}" -T json > "{json_path}"', shell=True, check=True)

    loader = JSONLoader(file_path=json_path, jq_schema=".[] | ._source.layers", text_content=False)
    docs = loader.load()
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    chunks = SemanticChunker(embedding).split_documents(docs)
    vectordb = Chroma.from_documents(chunks, embedding)
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-4o"),
        vectordb.as_retriever(search_kwargs={"k":5}),
        return_source_documents=True
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    question = st.text_input("💬 Ask something about your PCAP:")
    if question:
        with st.spinner("Analyzing packets…"):
            res = qa.invoke({"question": question, "chat_history": st.session_state.chat})
            st.session_state.chat.append((question, res["answer"]))

    for q,a in reversed(st.session_state.chat):
        st.markdown(f"**🧑‍💻 You:** {q}")
        st.markdown(f"**🤖 Packet Copilot:** {a}")
        st.markdown("---")
