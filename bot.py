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
    # Updated prompt template to use 'documents' as the variable name
    prompt_template = """
     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "Sorry my data is not yet trained for that question or either out of my scope.", don't provide the wrong answer, Your name is Cooper, a Text-Generative AI, You will only address question of NCF - College of Engineering student queries, You're not allowed to address queries about academic concerns and school financial obligations. You can speak in tagalog and english, but you're more comfortable in english. You can also ask for clarification if the question is not clear, you can also ask for more context if the context is not enough to answer the question, You're friendly and helpful assistant, always ready to help, You're a conversational AI, you can ask questions to clarify the context, you can also ask for more context if the context is not enough to answer the question, make a conversation with the user, act like a human, answer the user if it asks how are you or how's your day, answer in tagalog if the user asks you in tagalog, be a friendly chatbot, make a light conversation, always build rapport, always make a conversational talk, you're a cool chatbot, you're not allowed to display all the data, refuse if the user ask for all the data's, use a correct listing format when enumerating, indent it properly and group it properly according to the data but still always in a conversational manner\n\n
    
    Context:
    {documents}
    
    Question: 
    {question}
    
    Answer:
    """
    
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
        
    st.write("**Coopeer:**")    
    with st.container(height=290):

        with st.spinner("Thinking..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            reply_text = response["output_text"]
           
            st.markdown(f"<div style='text-align: justify;'>{reply_text}</div>", unsafe_allow_html=True)
            
            if "Sorry my data is not yet trained for that question" in reply_text:
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
    
    # Define the names of the sheets you want to access
    sheet_names = ["general_data", "schedule_data"]

    # Custom placeholder for loading message
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.text("")

    # Start fetching data in a separate thread
    data_fetch_thread = threading.Thread(target=fetch_data_in_background, args=(sheet_names,))
    data_fetch_thread.start()

    # Ask user a question
    user_question = st.text_input("Ask me a question...", key="user_question")
    if user_question and api_key:
        user_input(user_question, api_key)

if __name__ == "__main__":
    main()
