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
import json
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Cooper", layout="centered")

api_key = st.secrets["default"]["GOOGLE_API_KEY"]
sheet_names = st.secrets["default"]["SHEET_NAMES"].split(", ")

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
    # Prompt template to use 'documents' as the variable name
    prompt_template = """
        You are Cooper, a friendly and conversational Text-Generative AI assistant designed to help students of NCF - College of Engineering with their queries. You prefer communicating in English but can also speak Tagalog when needed. Avoid answering questions related to academics or financial obligations and apolozie if you're unable to provide an answer.
        
        You must never ask the user for additional information.
        You cannot provide the user's personal schedule or data unless it's a general query. Always maintain a friendly and helpful tone in your responses.
        You cannot ask the user for their specifically names, course, year level, student number, or any other personal information.
        
        When interacting with users, you must strictly adhere to the following guidelines:

        1. Greeting and Small Talk:
            * If the user greets with "hi," "hello," or similar, respond warmly. Example: "Hi there! How can I assist you today?"
            * If the user asks "How are you?" respond in a friendly manner like: "I'm doing great, thanks for asking! How can I help you today?"
            * When the user expresses gratitude (e.g., "thank you"), acknowledge it warmly, and avoid asking follow-up questions. Example: "You're welcome! I'm glad I could help."
        2. Handling Positive Responses:
            * If the user replies positively (e.g., "I'm good!" or "None"), acknowledge the response with enthusiasm. Example: "That's great to hear! Feel free to ask if you have more questions."
            * For replies like "No, thanks" or "No, thank you," respond with: "I'm glad I could help! Let me know if you need assistance later."
        3. General Conversation:
            * Keep conversations light, friendly, and engaging to build rapport. Always maintain a human-like tone.
            * Acknowledge each user reply to keep the conversation flowing naturally. Make sure to always be attentive and ready to assist.
        4. Language Preferences:
            * If the user communicates in Tagalog, respond in Tagalog. If the user switches back to English, do the same.
        5. Providing Accurate Information:
            * If an answer isn't available, respond honestly. Example: "Oops! It looks like I’m not trained on that topic just yet, or it might be a little out of my scope. Could you try asking something else?"
            * Never provide incorrect information.
        6. Restrictions:
            * Politely decline requests for full data display. Example: "Sorry, I can't show you all the data, but I can help with specific questions."
            * Never ask the user to provide more specific details.
        7. Handling Data-Related Queries:
            * When asked about schedules or similar data-related requests, respond by narrowing down the inquiry. Example: "I can help with that! Could you specify which day or class you're referring to?"
            * Never display all data from the database at once; instead, guide the user to refine their query.
        8. Proper Enumeration in Responses:
            * When providing a numbered list or sequence of steps, use proper numbering. Example:
                * "Here are the steps you need to follow:
                    1. First, open the application.
                    2. Then, log in with your credentials.
                    3. Finally, navigate to the dashboard."
            * When listing items without a specific order, use commas or semicolons. Example:
                * "You can choose from these courses: Computer Engineering, Geodetic Engineering, Civil Engineering."
            * For more detailed explanations with multiple points, use paragraphs or sub-points (like a, b, c). Example:
                * "The following factors are important: a. Time management helps you balance your studies and personal life. b. Prioritizing tasks ensures you focus on what's most important."
            * Avoid bulleting every line unless necessary for emphasis or simple lists.
        9. Closing the Conversation:
            * Never ask, "Is there anything else I can help you with today?" Instead, let the user decide when the conversation ends naturally.
    \n\n    
    Context:
    {documents}

    Question:
    {question}
    
    Answer:
    """

    # Dynamic template in the PromptTemplate
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["documents", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, document_variable_name="documents")
    return chain

def store_unknown_question_to_sheets(question):
    unknown_questions_sheet = "unknown_questions"
    try:
        existing_data = conn.read(worksheet=unknown_questions_sheet, usecols=list(range(1)), ttl=5)
        existing_data = existing_data.dropna(how="all")
        if existing_data["unknown_data"].str.contains(question).any():
            st.stop()
        else:
            data_to_add = pd.DataFrame([{"unknown_data": question}])
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
        
    if "Oops! It looks like I’m not trained on that topic just yet, or it might be a little out of my scope. Could you try asking something else?" in reply_text:
        store_unknown_question_to_sheets(user_question)
            

# This will fetch the data in a separate thread to avoid showing the spinner.
def fetch_data_in_background(sheet_names):
    sheet_data = get_google_sheets_data(sheet_names)
    if sheet_data:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_text(sheet_data)
        get_vector_store(text_chunks, api_key)
        st.success("Data loaded successfully!")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

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
    
    # set sidebar width
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 100%;
    </style>
    """,
    unsafe_allow_html=True,
    )

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
        
        with st.container():
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                lottie_bot = load_lottiefile("lottie_files/bot.json")
                st_lottie(
                    lottie_bot, 
                    speed=1, 
                    width=200, 
                    height=200,
                    reverse=False,
                    loop=True, 
                    key="lottie_bot")
            
            with col2:
                st.write("")
                st.title("Hi, I'm Cooper!")
                st.caption("I'm here to help you with any questions you have about NCF - College of Engineering. Feel free to ask me anything!")
                    
            # Clear the cache every time the home page is called
            clear_cache()

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
        st.caption(' **Collaborative Operations for Optimizing Practical Engineering Resources** (COOPER) is a text-generative AI developed in 2024 by Jay, Emil, Hans, and Erica from NCF - College of Engineering. This friendly chatbot is designed to assist students at NCF - College of Engineering. It can answer questions about various topics related to the NCF - College of Engineering and provide guidance. However, please note that Cooper cannot address queries about academic or school financial obligations.')

    # Contact Page
    elif page == "Contact":
        st.header("Contact Us")
        st.caption("For further assistance, please reach out to:")
        st.markdown("""
        <ul>
            <li>Address: NCF - College of Engineering, MT. Villanueva Ave., Naga City</li>
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
            <li>Provide context when asking specific questions to improve accuracy.</li>
            <li>If Cooper doesn’t understand your question, try rephrasing it.</li>
            <li>Be patient; sometimes it may take a moment to generate a response.</li>
            <li>Remember that Cooper cannot answer academic concerns or school financial obligations queries.</li>
        </ul>
        """, unsafe_allow_html=True)
     
if __name__ == "__main__":
    main()

