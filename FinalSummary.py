import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

def load_openai_api_key():
    dotenv_path = ".env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OpenAI_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
    return openai_api_key

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='allenai/longformer-base-4096')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def main():
    st.title("📄Semantic Scholar")
    st.write("Created by Arnav Khanna, Salaj Shekhar Jugran and Surya Prakash Singh")
    st.divider()

    try:
        os.environ["OPENAI_API_KEY"] = load_openai_api_key()
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF file. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledgeBase.similarity_search(query)
            OpenAIModel = "gpt-3.5-turbo"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.5)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('Summary Results:')
            st.write(response)

if __name__ == '__main__':
    main()