import streamlit as st
from pathlib import Path
import yaml
import time

def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def navigate_to(page):
    st.session_state.page = page

def main():
    initialize_session_state()
    
    if st.session_state.page == 'home':
        show_home()
    elif st.session_state.page == 'load_data':
        show_load_data()
    elif st.session_state.page == 'load_vectorstore':
        show_load_vectorstore()
    elif st.session_state.page == 'qa':
        show_qa()

def show_home():
    st.title("RAG System")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Data", use_container_width=True):
            navigate_to('load_data')
    with col2:
        if st.button("Load Vectorstore", use_container_width=True):
            navigate_to('load_vectorstore')

def show_load_data():
    st.title("Load Data")
    
    # Sidebar parameters
    with st.sidebar:
        st.header("Processing Parameters")
        chunk_size = st.slider("Chunk Size", 100, 1000, 256)
        overlap = st.slider("Chunk Overlap", 0, 100, 20)
        st.divider()
        st.markdown("Other parameters if needed...")

    # File uploads
    st.subheader("Upload Files")
    pdfs = st.file_uploader("Upload PDF Files", type=['pdf'], accept_multiple_files=True)
    
    st.divider()
    csvs = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
    
    st.divider()
    config = st.file_uploader("Upload Config File", type=['yml', 'yaml'])

    if st.button("Process and Store", type="primary", use_container_width=True):
        if pdfs or csvs:  # Allow processing even if one type is missing
            with st.spinner("Processing files..."):
                try:
                    # Your processing logic here
                    # Simulate processing with progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    st.success("Processing complete!")
                    st.session_state.vectorstore = "path_to_vectorstore"  # Save vectorstore path/object
                    
                    if st.button("Go to Q&A", type="primary", use_container_width=True):
                        navigate_to('qa')
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        else:
            st.warning("Please upload at least one PDF or CSV file")

def show_load_vectorstore():
    st.title("Load Existing Vectorstore")
    
    vectorstore_path = st.text_input("Enter path to existing vectorstore:")
    
    if st.button("Load Vectorstore", type="primary", use_container_width=True):
        if vectorstore_path:
            with st.spinner("Loading vectorstore..."):
                try:
                    # Your loading logic here
                    time.sleep(2)  # Simulate loading
                    st.session_state.vectorstore = vectorstore_path
                    st.success("Vectorstore loaded successfully!")
                    
                    if st.button("Go to Q&A", type="primary", use_container_width=True):
                        navigate_to('qa')
                except Exception as e:
                    st.error(f"Error loading vectorstore: {str(e)}")
        else:
            st.warning("Please enter a vectorstore path")

def show_qa():
    st.title("Q&A Interface")

    # Display chat history
    for q, a in st.session_state.chat_history:
        st.text_area("Question:", value=q, height=100, disabled=True)
        st.text_area("Answer:", value=a, height=200, disabled=True)
        st.divider()

    # Query input
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit", type="primary", use_container_width=True):
        if query:
            if st.session_state.vectorstore:
                with st.spinner("Generating response..."):
                    try:
                        # Your RAG query logic here
                        # Simulate response generation
                        time.sleep(2)
                        response = f"Sample response for: {query}"
                        
                        # Add to chat history
                        st.session_state.chat_history.append((query, response))
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.error("No vectorstore loaded. Please load data first.")
        else:
            st.warning("Please enter a question")

    # Clear chat button
    if st.button("Clear Chat", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()




import streamlit as st

# Set up the Streamlit app
st.set_page_config(page_title="RAG Pipeline", page_icon="📚", layout="wide")

# Main function
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a Page",
        ["Home", "Load Data", "Load Vectorstore", "Q&A"],
        index=0,
    )

    if page == "Home":
        home_page()
    elif page == "Load Data":
        load_data_page()
    elif page == "Load Vectorstore":
        load_vectorstore_page()
    elif page == "Q&A":
        qa_page()

# Home Page
def home_page():
    st.title("Welcome to the RAG Pipeline")
    st.write("Choose an option to get started:")
    choice = st.selectbox("Options", ["", "Load Data", "Load Vectorstore"])

    if choice == "Load Data":
        st.experimental_set_query_params(page="Load Data")
    elif choice == "Load Vectorstore":
        st.experimental_set_query_params(page="Load Vectorstore")

# Load Data Page
def load_data_page():
    st.title("Load Data")

    # File upload
    st.subheader("Step 1: Upload PDFs")
    pdf_files = st.file_uploader(
        "Upload one or more PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    st.subheader("Step 2: Upload CSVs")
    csv_files = st.file_uploader(
        "Upload one or more CSV files", 
        type=["csv"], 
        accept_multiple_files=True
    )

    st.subheader("Step 3: Upload Config")
    config_file = st.file_uploader("Upload your config.yml file", type=["yml", "yaml"])

    # Sidebar for chunk size
    st.sidebar.title("Settings")
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=1, value=500, step=50)

    # Submit button
    if st.button("Submit"):
        if pdf_files and csv_files and config_file:
            st.write("Processing data in the backend... Please wait.")
            # Backend processing logic can be called here
            st.success("Vectorstore created successfully!")
            if st.button("Go to Q&A"):
                st.experimental_set_query_params(page="Q&A")
        else:
            st.warning("Please upload all required files before proceeding.")

# Load Vectorstore Page
def load_vectorstore_page():
    st.title("Load Vectorstore")

    st.subheader("Enter the path to the existing Vectorstore:")
    vectorstore_path = st.text_input("Vectorstore Path")

    if st.button("Go to Q&A"):
        if vectorstore_path:
            st.experimental_set_query_params(page="Q&A")
        else:
            st.warning("Please enter the vectorstore path before proceeding.")

# Q&A Page
def qa_page():
    st.title("Q&A")

    st.subheader("Ask a question based on the loaded data:")
    query = st.text_input("Enter your question here")

    if st.button("Submit"):
        if query:
            st.write("Fetching response...")  # Backend response handling here
            st.success("Response: [Your response here]")  # Replace with actual response
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

