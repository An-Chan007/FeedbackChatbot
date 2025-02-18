import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
import docx  # Add this import for Word files
from io import BytesIO
import time

# Paths
CHROMA_PATH = "vector_index"
TEACHER_ANSWER_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Suggested Answer.pdf"
STUDENT_UPLOADS_PATH = "student_uploads"
QUESTION_CONTEXT_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Formative Question.pdf"

# LM Studio Model Initialization
llm = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed",
    model="openhermes-2.5-mistral-7b",
    temperature=0.05,  # Very low temperature
    max_tokens=1500,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    top_p=0.1
)

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Move all function definitions to the top
def process_student_upload(uploaded_file):
    """Process uploaded student file (PDF or DOCX)."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        # Process PDF
        import PyPDF2
        pdf_bytes = BytesIO(uploaded_file.read())
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
    elif file_extension in ['docx', 'doc']:
        # Process Word document
        doc = docx.Document(BytesIO(uploaded_file.read()))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Please upload PDF or Word document.")
    
    return text

def load_pdf_as_text(file_path):
    """Load and extract text from a PDF."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return "\n".join(chunks)

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # If not the last chunk, find a good breaking point
        if end < len(text):
            # Try to break at a period or newline
            while end > start and text[end] not in '.?\n':
                end -= 1
            if end == start:  # If no good break found, use maximum length
                end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Create overlap with previous chunk
    return chunks

def filter_hallucinations(feedback, teacher_text):
    """Filter out potential hallucinations from feedback."""
    lines = feedback.split("\n")
    filtered_lines = []
    
    for line in lines:
        # Only check lines starting with bullet points that mention legal references
        if line.strip().startswith("•") and any(word in line.lower() for word in ["v.", "act", "case"]):
            # Check if any part of this legal reference appears in teacher's text
            if not any(ref in teacher_text.lower() for ref in line.lower().split()):
                continue
        filtered_lines.append(line)
    
    return "\n".join(filtered_lines)

def compare_answers(teacher_answer, student_answer):
    """Generate feedback comparing answers against complete teacher's answer."""
    try:
        comparison_prompt = """<|im_start|>system
Compare student answer with teacher's model answer. Provide detailed feedback.
<|im_end|>
<|im_start|>user
TEACHER'S MODEL ANSWER:
{teacher_text}

STUDENT'S ANSWER:
{student_text}

Provide comprehensive feedback in these sections:

STRENGTHS:
• Does the answer contain the same content as the Suggested Answer?
• Does it identify all relevant legal issues that need to be addressed?
• Does it correctly cite and explain relevant law (cases and statutes)?
• Does it successfully apply law to the specific scenario?
• Is the analysis thorough and critical?
• Is the structure logical and clear?

WEAKNESSES:
• Which key legal concepts need better explanation?
• Which cases or statutes are missing or incorrectly explained?
• Where does the legal analysis need more development?
• Which structural elements need improvement?

SUGGESTIONS FOR IMPROVEMENT:
• Identify specific legal concepts needing clearer explanation
• List missing case law or statutory references
• Point out where analysis needs more depth
• Suggest structural improvements

Important: Provide detailed feedback for each section with specific examples.
<|im_end|>
<|im_start|>assistant""".format(
            teacher_text=teacher_answer,
            student_text=student_answer
        )
        
        response = llm.invoke(
            input=comparison_prompt,
            temperature=0.01,
            max_tokens=4000,  # Increased significantly
            stop=["</s>", "<|im_end|>", "<|im_start|>"]
        )
        
        if response and isinstance(response, str):
            cleaned_response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
            return cleaned_response
            
    except Exception as e:
        st.write(f"Error details: {str(e)}")
        return f"Error: {str(e)}"

def chat_with_rag(query, vectordb, student_text=None, teacher_answer=None):
    """Enhanced chat function with better query handling."""
    
    # First, check if it's a simple greeting or short query
    simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if query.lower().strip() in simple_greetings:
        return "Hello! How can I help you with the analysis today?"
    
    # For actual questions, proceed with RAG
    rd = vectordb.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in rd])
    
    additional_context = ""
    if student_text and teacher_answer:
        additional_context = f"\nStudent's Answer: {student_text}\nTeacher's Answer: {teacher_answer}"
    
    combined_context = f"{context}{additional_context}"
    
    prompt_template = """
    You are a knowledgeable legal assistant specializing in property law. 
    
    If the question is about the legal documents or student answers:
    - Provide specific legal analysis
    - Reference relevant cases and principles
    - Explain legal concepts clearly
    - Compare with the teacher's answer when relevant
    
    If the question is general or non-legal:
    - Provide a simple, direct response
    - Stay professional but conversational
    - Offer to help with legal analysis if needed

    Context:
    {context}

    Question: {query}

    Please provide a clear, focused response:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "context": combined_context,
            "query": query
        })
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Load files before Streamlit interface
# Load or Create Vector Database
if os.path.exists(CHROMA_PATH):
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
else:
    st.write("Building vector database...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    documents = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            pdf_file = os.path.join("data", file)
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())

    chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    vectordb.persist()

# Load Teacher's Answer and Licence to Occupy Context
teacher_answer = load_pdf_as_text(TEACHER_ANSWER_PATH)
# licence_context = load_pdf_as_text(LICENCE_TO_OCCUPY_PATH)
question_context = load_pdf_as_text(QUESTION_CONTEXT_PATH)

# Then start Streamlit interface
st.title("Legal Document Analysis Assistant")

# Initialize session state for chat history and uploaded file status
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# File uploader in the main chat area
uploaded_student_file = st.file_uploader(
    "Upload Student Answer (PDF or Word)", 
    type=["pdf", "docx", "doc"],
    key="file_uploader"
)

# Process uploaded file
if uploaded_student_file:
    with st.spinner('Processing submission...'):
        try:
            if st.session_state.current_file != uploaded_student_file.name:
                st.session_state.current_file = uploaded_student_file.name
                student_text = process_student_upload(uploaded_student_file)
                
                if student_text:
                    with st.spinner('Analyzing...'):
                        feedback = compare_answers(
                            teacher_answer=teacher_answer,
                            student_answer=student_text
                        )
                        
                        if feedback and not feedback.startswith("Error"):
                            st.session_state.messages = [{
                                "role": "assistant",
                                "content": feedback
                            }]
                            st.session_state.file_uploaded = True
                        else:
                            st.error("Failed to generate feedback. Please try again.")
                else:
                    st.error("Failed to process the uploaded file.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and add response
    response = chat_with_rag(
        prompt,
        vectordb,
        student_text=student_text if 'student_text' in locals() else None,
        teacher_answer=teacher_answer
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
