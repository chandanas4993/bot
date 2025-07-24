import os
import streamlit as st
import pickle

from dotenv import load_dotenv
load_dotenv()  # Loads environment variables from .env file

# LangChain modules
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# HuggingFace pipeline (local model)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Load model and tokenizer locally
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Wrap Hugging Face pipeline with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Streamlit UI
st.title("🧠 Research Tool")
st.sidebar.title("Article URLs")

# Input URLs from user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"  # Save using huggingface-specific name

main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("🔄 Loading content from URLs...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    main_placeholder.text("✂️ Text split done.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.success("✅ Embeddings created and saved.")

query = main_placeholder.text_input("💬 Ask a question about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
