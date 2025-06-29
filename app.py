import gradio as gr
import tempfile
import os
from typing import List, Tuple, Optional
import logging

# Try imports with fallbacks
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFReaderApp:
    def __init__(self):
        \"\"\"Initialize the PDF Reader application\"\"\"
        self.vector_store = None
        self.qa_chain = None
        self.current_pdf_name = None
        self.pdf_text = \"\"
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning(\"LangChain not available, using basic PDF reading\")
    
    def _initialize_models(self):
        \"\"\"Initialize the language models and embeddings\"\"\"
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=\"sentence-transformers/all-MiniLM-L6-v2\"
            )
            
            # Initialize language model pipeline
            self.llm_pipeline = pipeline(
                \"text2text-generation\",
                model=\"google/flan-t5-base\",
                max_length=512,
                temperature=0.7,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
            logger.info(\"Models initialized successfully\")
            
        except Exception as e:
            logger.error(f\"Error initializing models: {e}\")
            self.embeddings = None
            self.llm = None
    
    def extract_text_basic(self, pdf_file) -> str:
        \"\"\"Extract text from PDF using PyPDF2 (fallback method)\"\"\"
        try:
            if PYPDF2_AVAILABLE:
                reader = PyPDF2.PdfReader(pdf_file)
                text = \"\"
                for page in reader.pages:
                    text += page.extract_text() + \"\\n\"
                return text
            else:
                return \"PDF text extraction not available. Please install PyPDF2.\"
        except Exception as e:
            return f\"Error extracting text: {str(e)}\"
    
    def process_pdf(self, pdf_file) -> str:
        \"\"\"Process uploaded PDF file\"\"\"
        if pdf_file is None:
            return \"❌ Please upload a PDF file.\"
        
        try:
            self.current_pdf_name = pdf_file.name
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=\".pdf\") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file.flush()
                
                if LANGCHAIN_AVAILABLE and self.embeddings:
                    return self._process_with_langchain(tmp_file.name)
                else:
                    return self._process_basic(tmp_file.name)
                    
        except Exception as e:
            logger.error(f\"Error processing PDF: {e}\")
            return f\"❌ Error processing PDF: {str(e)}\"
        finally:
            # Clean up temporary file
            if 'tmp_file' in locals():
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass

if __name__ == \"__main__\":
    # Minimal demo version for deployment
    with gr.Blocks(title=\"📄 PDF Reader AI\") as demo:
        gr.Markdown(\"# 📄 Interactive PDF Reader with AI Q&A\")
        gr.Markdown(\"Upload PDFs and ask questions about their content!\")
        
        with gr.Row():
            pdf_upload = gr.File(label=\"Upload PDF\", file_types=[\".pdf\"])
            question_input = gr.Textbox(label=\"Ask a question\")
        
        output = gr.Textbox(label=\"Answer\", lines=5)
        
        gr.Button(\"Process\").click(
            lambda f, q: \"Demo: PDF processing and Q&A coming soon!\",
            inputs=[pdf_upload, question_input],
            outputs=output
        )
    
    demo.launch()
