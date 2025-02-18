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
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict
# Use this import instead
from llama_index.llms.langchain import LangChainLLM

# Paths
CHROMA_PATH = "vector_index"
TEACHER_ANSWER_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Suggested Answer.pdf"
STUDENT_UPLOADS_PATH = "student_uploads"
QUESTION_CONTEXT_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Formative Question.pdf"

# Initialize LLM with correct parameters
langchain_llm = OpenAI(
    model_name="openhermes-2.5-mistral-7b",  # Changed from model to model_name
    temperature=0.01,
    base_url="http://127.0.0.1:1234/v1",     # Changed from api_base to base_url
    api_key="not-needed",
    max_tokens=2000
)

llm = LangChainLLM(llm=langchain_llm)

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

def analyze_legal_content(text: str) -> str:
    """Analyzes legal content for key land law components."""
    analysis_prompt = """Analyze this land law text and identify:
    1. UK land law principles mentioned
    2. UK case law citations
    3. UK statutory references
    4. Key arguments about lease vs license
    
    Text to analyze: {text}
    """
    return llm.complete(analysis_prompt.format(text=text))

def verify_references(analysis: str) -> str:
    """Verifies legal references and claims."""
    verify_prompt = """Verify these elements:
    1. Check each case citation
    2. Confirm statutory references
    3. Validate legal principles
    
    Analysis to verify: {analysis}
    """
    return llm.complete(verify_prompt.format(analysis=analysis))

# Create function tools
analysis_tool = FunctionTool.from_defaults(
    fn=analyze_legal_content,
    name="analyze_legal_content",
    description="Analyzes legal content for key components"
)

verification_tool = FunctionTool.from_defaults(
    fn=verify_references,
    name="verify_references",
    description="Verifies legal references and claims"
)

# Create agents
analysis_agent = ReActAgent.from_tools(
    tools=[analysis_tool],
    llm=llm,
    verbose=True
)

verification_agent = ReActAgent.from_tools(
    tools=[verification_tool],
    llm=llm,
    verbose=True
)

def compare_answers(teacher_answer: str, student_answer: str) -> str:
    """Generate feedback comparing answers."""
    try:
        # First use agents for analysis
        analysis = analysis_agent.chat(f"Analyze both answers: Teacher: {teacher_answer}, Student: {student_answer}")
        verified = verification_agent.chat(f"Verify this analysis: {analysis}")
        
        comparison_prompt = """<|im_start|>system
You are a law professor providing feedback. You must ONLY reference content that actually exists in the teacher's answer.
If the teacher's answer explicitly states something is NOT required, you must NOT mention it as a weakness or suggestion.
<|im_end|>
<|im_start|>user
Compare:

TEACHER'S MODEL ANSWER:
{teacher_text}

STUDENT'S ANSWER:
{student_text}

Based on this verified analysis:
{analysis}

CRITICAL INSTRUCTIONS:
1. Only mention elements that are actually required in the teacher's answer
2. If teacher states something is NOT needed, do NOT include it in feedback
3. Do not invent requirements
4. Stay strictly within the teacher's model answer

Provide feedback in this EXACT format:

STRENGTHS:
• [Each strength must reference actual content from the student's answer]
• Focus on legal principles, case law, and analysis that are actually present
• Be specific about what the student has done well

WEAKNESSES:
• [Each weakness must be based on actual requirements in the teacher's answer]
• Only mention elements that the teacher's answer explicitly requires
• Be specific about genuine gaps in analysis

SUGGESTIONS FOR IMPROVEMENT:
1. [Number each suggestion]
2. Make each suggestion specific and actionable
3. Link suggestions to identified weaknesses
4. Focus on concrete improvements based on the teacher's requirements
<|im_end|>
<|im_start|>assistant"""
        
        response = llm.complete(comparison_prompt.format(
            teacher_text=teacher_answer,
            student_text=student_answer,
            analysis=verified
        ))
        
        return response.text if response else "Error generating feedback"
            
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

# Streamlit UI
def main():
    st.title("Feedback Analysis System")
    
    uploaded_file = st.file_uploader(
        "Upload Student Answer (PDF or Word)", 
        type=["pdf", "docx", "doc"],
        key="student_file_uploader"
    )
    
    if uploaded_file:
        with st.spinner('Processing submission...'):
            try:
                student_text = process_student_upload(uploaded_file)
                teacher_answer = load_pdf_as_text(TEACHER_ANSWER_PATH)
                
                if student_text:
                    feedback = compare_answers(
                        teacher_answer=teacher_answer,
                        student_answer=student_text
                    )
                    
                    st.write("### Feedback")
                    st.markdown(feedback)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Add chat interface
    st.write("---")
    st.write("### Ask Questions")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = chat_with_rag(
                prompt, 
                vectordb,
                student_text=student_text if 'student_text' in locals() else None,
                teacher_answer=teacher_answer if 'teacher_answer' in locals() else None
            )
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

def compare_legal_answers(student_content: Dict, teacher_content: Dict) -> Dict:
    """Compares analyzed legal content between student and teacher answers."""
    comparison_prompt = """Compare these legal analyses:
    
    TEACHER'S CONTENT:
    {teacher}
    
    STUDENT'S CONTENT:
    {student}
    
    Identify:
    1. Matching legal principles
    2. Matching case citations
    3. Missing elements
    4. Areas needing improvement
    """
    
    response = llm.complete(comparison_prompt.format(
        teacher=str(teacher_content),
        student=str(student_content)
    ))
    return response.text if response else "Error in comparison"

def generate_structured_feedback(comparison_results: Dict) -> str:
    """Generates structured feedback from comparison results."""
    feedback_prompt = """Generate structured feedback with these sections:
    
    STRENGTHS:
    • List key legal concepts correctly explained
    • Note accurate case law usage
    • Highlight strong analytical points
    
    WEAKNESSES:
    • Identify missing legal principles
    • List omitted or incorrect cases
    • Note gaps in analysis
    
    SUGGESTIONS:
    • Recommend specific improvements
    • List cases to include
    • Suggest analytical enhancements
    
    Based on this comparison:
    {results}
    """
    
    response = llm.complete(feedback_prompt.format(results=str(comparison_results)))
    return response.text if response else "Error in feedback generation"

def review_feedback_quality(feedback: str) -> Dict:
    """Reviews feedback for quality and accuracy."""
    review_prompt = """Review this feedback for:
    1. Accuracy of legal analysis
    2. Specificity of suggestions
    3. Completeness of review
    4. Clarity of presentation
    
    Feedback to review:
    {feedback}
    
    Provide approval or specific improvement requests.
    """
    
    response = llm.complete(review_prompt.format(feedback=feedback))
    return response.text if response else "Error in review"

def process_with_agents(teacher_answer: str, student_answer: str) -> str:
    """Process answers using agent workflow."""
    try:
        # Keep the same successful formatting but use agents for analysis
        comparison_prompt = """<|im_start|>system
You are a law professor providing feedback. You must ONLY reference content that actually exists in the teacher's answer.
If the teacher's answer explicitly states something is NOT required, you must NOT mention it as a weakness or suggestion.
<|im_end|>
<|im_start|>user
Compare:

TEACHER'S MODEL ANSWER:
{teacher_text}

STUDENT'S ANSWER:
{student_text}

Provide feedback in this EXACT format:

STRENGTHS:
• [Each strength on a new line with a bullet point]
• Focus on legal principles, case law, and analysis
• Be specific about what the student has done well

WEAKNESSES:
• [Each weakness on a new line with a bullet point]
• Only mention genuinely missing elements
• Be specific about gaps in analysis

SUGGESTIONS FOR IMPROVEMENT:
1. [Number each suggestion]
2. Make each suggestion specific and actionable
3. Link suggestions to identified weaknesses
4. Focus on concrete improvements
<|im_end|>
<|im_start|>assistant"""

        # Use agents for analysis before formatting
        analysis_result = analysis_agent.run(f"Analyze both answers: Teacher: {teacher_answer}, Student: {student_answer}")
        verified_result = verification_agent.run(f"Verify this analysis: {analysis_result}")
        
        # Generate final feedback using verified analysis but keeping your format
        response = llm.complete(comparison_prompt.format(
            teacher_text=teacher_answer,
            student_text=student_answer,
            analysis=verified_result
        ))
        
        return response.text if response else "Error generating feedback"
            
    except Exception as e:
        return f"Error in agent processing: {str(e)}"
