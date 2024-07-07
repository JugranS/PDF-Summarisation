
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain.callbacks.manager import get_openai_callback

def process_text(text, embeddings):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base

    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def summarise(llm, embeddings):
    st.title("📄Semantic Scholar")
    st.write("Created by Arnav Khanna, Salaj Shekhar Jugran and Surya Prakash Singh")
    st.divider()

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text, embeddings)

        query = "Summarize the content of the uploaded PDF file. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledgeBase.similarity_search(query)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('Summary Results:')
            st.write(response)