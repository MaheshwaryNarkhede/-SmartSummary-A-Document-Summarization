import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os  

# Model and tokenizer loading
@st.cache_resource  # Cache the model loading to avoid reloading it each time
def load_model():
    checkpoint = "LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
    return base_model, tokenizer

base_model, tokenizer = load_model()

# File loader and preprocessing
def file_preprocessing(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = " ".join([text.page_content for text in texts])  # Combine all page contents
        return final_texts
    except Exception as e:
        st.error(f"Error in file preprocessing: {e}")
        return ""

# LLM pipeline
def llm_pipeline(input_text):
    try:
        pipe_sum = pipeline(
            'summarization',
            model=base_model,
            tokenizer=tokenizer,
            max_length=500, 
            min_length=50
        )
        result = pipe_sum(input_text)
        summary = result[0]['summary_text']
        return summary
    except Exception as e:
        st.error(f"Error in LLM pipeline: {e}")
        return ""

# Function to display the PDF of a given file 
@st.cache_data
def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("📘 SmartSummary ")
    st.title("Document Summarization App using Language Model")

    # Upload file
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            # Create a temp directory to store the file
            temp_dir = 'temp_files'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file to a local directory
            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded File")
                display_pdf(file_path)  # Display PDF in the first column

            with col2:
                st.info("Summarizing... This might take a while")
                input_text = file_preprocessing(file_path)  # Preprocess PDF to extract text
                if input_text:
                    summary = llm_pipeline(input_text)  # Summarize the extracted text
                    if summary:
                        st.success("Summarization Complete!")
                        st.write(summary)
                    else:
                        st.error("Summarization failed. Please try again.")
                else:
                    st.error("Failed to extract text from PDF.")

if __name__ == "__main__":
    main()
