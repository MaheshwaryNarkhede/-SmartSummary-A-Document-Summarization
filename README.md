# 📘SmartSummary-A-Document-Summarization


SmartSummary is a document summarization application built using Streamlit, LangChain, and HuggingFace transformers. The app allows users to upload a PDF file, preprocesses the document, and provides a summarized version using the LaMini-Flan-T5 language model.

## Features

- **PDF Upload**: Upload PDF documents for summarization.
- **Document Preprocessing**: Extracts and preprocesses text from PDF documents.
- **Summarization**: Summarizes long documents using the LaMini-Flan-T5 model from HuggingFace.
- **PDF Viewer**: Displays the uploaded PDF in the app for easy review.
- **Streamlit Interface**: Simple and interactive UI for summarizing documents.

## Requirements

Make sure you have the following dependencies installed in your environment:

- Python 3.8 or above
- Streamlit
- LangChain
- HuggingFace transformers
- PyTorch
- T5Tokenizer and T5ForConditionalGeneration (from transformers)
- Other necessary libraries

You can install the required libraries using pip:

```bash
pip install streamlit langchain transformers torch

