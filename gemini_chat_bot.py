import streamlit as st
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_json_text(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = json.dumps(data)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "answer is not available in the context." Do not provide a wrong answer.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response.get("output_text", "No answer available."))

def main():
    st.set_page_config(page_title="University Query Assistant")
    st.header("Ask Your Question to the University Information System")

    user_question = st.text_input("Ask a Question about Students, Faculty, Alumni, Placements, etc.")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload Documents")
        st.write("The system will automatically use the latest data from the uploaded documents.")
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                all_text = ""
                base_path = "./data/"  # Directory where JSON files are stored
                for filename in os.listdir(base_path):
                    file_path = os.path.join(base_path, filename)
                    if filename.endswith(".json"):
                        all_text += get_json_text(file_path)
                text_chunks = get_text_chunks(all_text)
                get_vector_store(text_chunks)
                st.success("Processing complete. You can now ask questions.")

if __name__ == "__main__":
    main()
