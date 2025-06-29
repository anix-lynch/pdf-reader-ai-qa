# 📄 Interactive PDF Reader with AI Q&A

An intelligent PDF reader that allows you to upload documents and ask questions about their content using AI-powered search and retrieval.

## 🚀 Features

- **PDF Upload & Processing**: Extract text from any PDF document
- **AI-Powered Q&A**: Ask natural language questions about PDF content
- **Semantic Search**: Find relevant information using vector embeddings
- **Document Summarization**: Generate automatic summaries
- **Interactive Interface**: User-friendly Gradio web interface
- **Fallback Support**: Works with or without advanced dependencies

## 🛠️ Technology Stack

- **Document Processing**: LangChain + PyPDF2
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Language Model**: Google FLAN-T5-base
- **Frontend**: Gradio
- **Text Processing**: Recursive character text splitter

## 🎯 How It Works

1. **PDF Upload**: Upload any PDF document through the web interface
2. **Text Extraction**: Extract and process text content from the PDF
3. **Embedding Creation**: Convert text chunks into vector embeddings
4. **Vector Storage**: Store embeddings in FAISS for fast similarity search
5. **Question Processing**: Convert user questions into embeddings
6. **Retrieval**: Find most relevant document sections
7. **Answer Generation**: Generate contextual answers using language models

## 🔧 Usage

### Basic Workflow:
1. **Upload PDF**: Click \"Upload PDF\" and select your document
2. **Process**: Click \"Process PDF\" to extract and index content
3. **Ask Questions**: Enter natural language questions about the content
4. **Get Answers**: Receive AI-generated answers with source references

### Example Questions:
- \"What is the main topic of this document?\"
- \"Summarize the key findings\"
- \"What methodology was used?\"
- \"Who are the main authors or contributors?\"
- \"What data supports the conclusions?\"

## 📊 Capabilities

### Advanced Mode (with LangChain):
- **Semantic Q&A**: Context-aware question answering
- **Source Attribution**: Answers linked to specific document sections
- **Vector Similarity**: Advanced retrieval using embeddings
- **Chunk Processing**: Intelligent text segmentation

### Basic Mode (fallback):
- **Text Extraction**: Basic PDF text reading
- **Keyword Search**: Simple text matching
- **Document Stats**: Word count, character analysis
- **Text Preview**: Content exploration

## 🎨 Interface Features

- **Clean Design**: Intuitive Gradio interface
- **Real-time Processing**: Live status updates
- **Example Prompts**: Suggested questions to get started
- **Error Handling**: Graceful fallbacks and error messages
- **File Support**: Standard PDF format compatibility

## ⚙️ Technical Details

- **Chunk Size**: 1000 characters with 200 character overlap
- **Embedding Model**: 384-dimensional sentence embeddings
- **Retrieval**: Top-k similarity search (k=3)
- **Language Model**: Text-to-text generation
- **Memory**: Optimized for moderate document sizes

## 🔍 Use Cases

- **Research**: Academic paper analysis and Q&A
- **Business**: Report summarization and insights
- **Legal**: Document review and information extraction
- **Education**: Study material comprehension
- **Personal**: Book/article analysis and note-taking

## 📈 Performance

- **Processing Speed**: ~30 seconds for typical documents
- **Accuracy**: High relevance with semantic understanding
- **Scalability**: Handles documents up to ~100 pages efficiently
- **Memory Usage**: Optimized for consumer hardware

## 🛡️ Error Handling

- **Dependency Fallbacks**: Works without full LangChain stack
- **File Validation**: PDF format verification
- **Memory Management**: Automatic cleanup of temporary files
- **Graceful Degradation**: Basic functionality when advanced features unavailable

---

**Built by**: Bchan  
**Purpose**: Demonstrating AI-powered document analysis and Q&A  
**License**: MIT
