import streamlit as st
from streamlit_gsheets import GSheetsConnection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd
import threading

st.set_page_config(page_title="Cooper", layout="centered")

# Google API Key
api_key = st.secrets["default"]["GOOGLE_API_KEY"]

prompt_template = st.secrets["custom"]["prompt_template"]
sheet_names = st.secrets["custom"]["sheet_names"]

# Google Sheets connection using streamlit_gsheets
conn = st.connection("gsheets", type=GSheetsConnection)

def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

def get_google_sheets_data(sheet_names):
    all_data = []
    for sheet in sheet_names:
        try:
            data = conn.read(worksheet=sheet)
            if not data.empty:
                all_data.append(data.to_string(index=False))
            else:
                st.warning(f"No data found in the Google Sheet: {sheet}")
        except Exception as e:
            st.error(f"Error reading sheet '{sheet}': {str(e)}")
    if all_data:
        return "\n".join(all_data)
    else:
        st.error("No data was retrieved from any of the provided sheets.")
        return None

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Prompt template now retrieved from secrets
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["documents", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, document_variable_name="documents")
    return chain

def store_unknown_question_to_sheets(question):
    unknown_questions_sheet = "unknown_questions"
    try:
        existing_data = conn.read(worksheet=unknown_questions_sheet, usecols=list(range(1)), ttl=5)
        existing_data = existing_data.dropna(how="all")
        if existing_data["unknown_questions"].str.contains(question).any():
            st.stop()
        else:
            data_to_add = pd.DataFrame([{"unknown_questions": question}])
            update_df = pd.concat([existing_data, data_to_add], ignore_index=True)
            conn.update(worksheet=unknown_questions_sheet, data=update_df)
    except Exception as e:
        st.error(f"Error storing question in Google Sheets: {str(e)}")

def user_input(user_question, api_key):
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
    st.write("**Cooper:**")    
    with st.container(height=290):

        with st.spinner("Thinking..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            reply_text = response["output_text"]
           
            st.markdown(f"<div style='text-align: justify;'>{reply_text}</div>", unsafe_allow_html=True)
            
            if "Oops! It looks like I’m not trained on that topic just yet, or it might be a little out of my scope. Could you try asking something else? 😊" in reply_text:
                store_unknown_question_to_sheets(user_question)

# This will fetch the data in a separate thread to avoid showing the spinner.
def fetch_data_in_background(sheet_names):
    sheet_data = get_google_sheets_data(sheet_names)
    if sheet_data:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_text(sheet_data)
        get_vector_store(text_chunks, api_key)
        st.success("Data loaded successfully!")

# Main App
def main():
    # hide_st_style = """
    # <style>
    # #MainMenu {visibility: hidden;}
    # footer {visibility: hidden;}
    # header {visibility: hidden;}
    # </style>
    # """
    # st.markdown(hide_st_style, unsafe_allow_html=True)
    
    st.title("Hi, I'm Cooper!")

    # Clear the cache every time the main function is called
    clear_cache()
    
    # Start fetching data in a separate thread
    data_fetch_thread = threading.Thread(target=fetch_data_in_background, args=(sheet_names,))
    data_fetch_thread.start()

    # Ask user a question
    user_question = st.text_input("Ask me a question...", key="user_question")
    if user_question and api_key:
        user_input(user_question, api_key)

if __name__ == "__main__":
    main()
