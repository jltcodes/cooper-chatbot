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
from streamlit_option_menu import option_menu

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
        Answer the question in as detailed as posible using the provided context. If the answer is not available, respond with:
        "Oops! It looks like I’m not trained on that topic just yet, or it might be a little out of my scope. Could you try asking something else? 😊". 
        Do not provide incorrect answers.

        Your name is Cooper, a friendly and conversational Text-Generative AI designed for answering NCF - College of Engineering student queries. 
        Avoid answering questions about academic concerns or financial obligations. You can communicate in both English and Tagalog, but you're more comfortable in English.

        - If the user greets with "hi," "hello," or similar greetings, respond warmly, like: "Hi there! How can I assist you today?"
        - Always express gratitude when the user says "thank you" or similar phrases.
        - If the user asks "How are you?" or something similar, respond in a friendly manner and reply back accordingly.
        - Acknowledge positive user replies, like "I'm good!" with a follow-up such as: "That's great to hear! What can I help you with today?"
        - Acknowledge each user reply to keep the conversation flowing.
        - Ask for clarification if the question is unclear or if more context is needed.
        - Make the conversation light and friendly, building rapport as a cool chatbot.
        - Answer in Tagalog if asked in Tagalog.
        - Use proper formatting and indentation when listing information.
        - Politely refuse if asked to display all data.
        - Greet the user at the beginning and end of the conversation.
        - You're always ready to assist the user with their queries.
        - Act like a human and maintain a friendly tone throughout the conversation.
        - You're cool, friendly, and always ready to help!

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
            
    st.write("**Cooper:**")    

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
    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.subheader("Menu")
        page = option_menu(
            menu_title = None, 
            options = ["Chat", "Guides", "About", "Contact"], 
            icons = ['chat-text', 'list-ol', 'info-circle', 'telephone'],
            menu_icon = "cast",
               styles={
                    "container": {"padding": "3px", "background-color": "#0e1117", "border-radius": "10px"},
                    "icon": {"color": "white", "font-size": "17px"}, 
                    "nav-link": {"font-size": "17px", "text-align": "left", "margin":"5px", "--hover-color": "#262730", "font-family": "monospace", "border-radius": "10px"},
                    "nav-link-selected": {"background-color": "#096F4E", "border-radius": "10px"},
    })
        

    # Home Page
    if page == "Chat":
        st.header("Hi, I'm Cooper!")
                
        # Clear the cache every time the home page is called
        clear_cache()

        # Define the names of the sheets you want to access
        sheet_names = ["general_data", "schedule_data"]

        # Start fetching data in a separate thread
        data_fetch_thread = threading.Thread(target=fetch_data_in_background, args=(sheet_names,))
        data_fetch_thread.start()

        # Ask user a question
        user_question = st.text_input("Ask me a question...", key="user_question")
        if user_question and api_key:
            user_input(user_question, api_key)

    # About Page
    elif page == "About":
        st.header("About Cooper")
        st.write(" Cooper is a text-generative AI developed in 2024 by Jay, Emil, Hans, and Erica from NCF - College of Engineering. This friendly chatbot is designed to assist students at NCF - College of Engineering. It can answer questions about various topics related to the NCF - College of Engineering and provide guidance. However, please note that Cooper cannot address queries about academic or school financial obligations.")

    # Contact Page
    elif page == "Contact":
        st.header("Contact Us")
        st.write("For further assistance, please reach out to:")
        st.markdown("""
        <ul>
            <li>Address: NCF - College of Engineering, MT. Villanueva Ave., Naga City, Philippines</li>
            <li>Email: support@ncf.edu</li>
            <li>Phone: (123) 456-7890</li>
        </ul>
        """, unsafe_allow_html=True)

    # Instructions Page
    elif page == "Guides":
        st.header("Tips!")
        st.write("Here are some tips to interact with Cooper:")
        st.markdown("""
        <ul>
            <li>Ask anything related to NCF - College of Engineering.</li>
            <li>Use clear and concise questions for better responses.</li>
            <li>Use clear and concise questions for better responses.</li>
            <li>Provide context when asking specific questions to improve accuracy.</li>
            <li>If Cooper doesn’t understand your question, try rephrasing it.</li>
            <li>Be patient; sometimes it may take a moment to generate a response.</li>
            <li>Remember that Cooper cannot answer academic concerns or financial obligations.</li>
        </ul>
        """, unsafe_allow_html=True)
     
if __name__ == "__main__":
    main()

