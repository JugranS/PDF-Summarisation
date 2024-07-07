from summarise import summarise, sidebar
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


LLM_CHOICES = [
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4"
]

def load_openai_api_key():
    dotenv_path = ".env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OpenAI_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
    return openai_api_key

def main():
    try:
        os.environ["OPENAI_API_KEY"] = load_openai_api_key()
    except ValueError as e:
        st.error(str(e))
        return
    embeddings = HuggingFaceEmbeddings(model_name='allenai/longformer-base-4096')
    OpenAIModel = sidebar(LLM_CHOICES)
    llm = ChatOpenAI(model=OpenAIModel, temperature=0.5)
    summarise(llm, embeddings)

if __name__ == '__main__':
    main()