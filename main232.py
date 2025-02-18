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
from typing import List

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

def load_training_data():
    """Load all student answers and teacher feedback."""
    training_data = []
    
    # Load teacher's suggested answer first
    suggested_answer = load_pdf_as_text(TEACHER_ANSWER_PATH)
    
    # Load each student answer and feedback
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        try:
            student_answer_path = f"data/Student Answer {letter}.pdf"
            feedback_path = f"data/Student {letter} feedback.pdf"
            
            student_answer = load_pdf_as_text(student_answer_path)
            teacher_feedback = load_pdf_as_text(feedback_path)
            
            if student_answer and teacher_feedback:
                training_data.append({
                    'suggested_answer': suggested_answer,
                    'student_answer': student_answer,
                    'teacher_feedback': teacher_feedback,
                    'letter': letter
                })
                st.write(f"✓ Loaded training data for Student {letter}")
            
        except Exception as e:
            st.error(f"Error loading data for Student {letter}: {str(e)}")
    
    if len(training_data) > 0:
        st.write(f"✓ Successfully loaded {len(training_data)} training examples")
    
    return training_data

def create_training_prompt(training_data):
    """Create a structured prompt from teacher's feedback patterns."""
    examples = ""
    for data in training_data:
        examples += f"""
FEEDBACK PATTERN:
1. Legal Concept Explanation:
- How the teacher explains exclusive possession
- How cases like Lace v Chantler are explained
- How legal principles are broken down for clarity

2. Authority Citations:
- How cases support each legal point
- Example: {data['teacher_feedback']}

3. Analysis Development:
- How points are carried to conclusion
- How cases like Esso v Fumegrange are applied
- Complete discussion patterns

4. Structural Elements:
- Topic organization
- Logical flow of arguments
- Use of given structure
---
"""
    return examples

def validate_legal_references(feedback: str, teacher_answer: str) -> bool:
    """Validate that legal references exist in teacher's answer."""
    # Key cases from the suggested answer (exactly as they appear)
    key_cases = [
        "street v mountford",
        "ashburn anstalt v arnold",
        "esso petroleum v fumegrange",
        "aslan v murphy",
        "antoniades v villiers",
        "facchini v bryson",
        "central estates (belgravia) v woolgar",
        "billson v residential apartments"
    ]
    
    feedback_lower = feedback.lower()
    teacher_lower = teacher_answer.lower()
    
    # Check if any case mentioned in feedback is not in teacher's answer
    for case in key_cases:
        if case in feedback_lower and case not in teacher_lower:
            return False
    return True

def validate_feedback_structure(feedback: str) -> bool:
    """Validate feedback has required sections and format."""
    required_sections = [
        "STRENGTHS:",
        "WEAKNESSES:",
        "SUGGESTIONS FOR IMPROVEMENT:"
    ]
    
    # Make case-sensitive exact matches and ensure they appear in order
    feedback_lines = feedback.split('\n')
    section_found = [False] * len(required_sections)
    
    for line in feedback_lines:
        for i, section in enumerate(required_sections):
            if line.strip() == section:
                section_found[i] = True
                break
    
    return all(section_found)

def compare_answers(teacher_answer: str, student_answer: str, training_examples: str) -> str:
    """Generate feedback comparing answers."""
    try:
        comparison_prompt = f"""<|im_start|>system
You are a law professor providing feedback on a Land Law answer. Your feedback must reference these key cases correctly:
- Street v Mountford (exclusive possession test)
- Ashburn Anstalt v Arnold (rent not required)
- Esso Petroleum v Fumegrange (commercial context)
- Aslan v Murphy (sham clauses)
- Antoniades v Villiers (sharing arrangements)
- Facchini v Bryson (family arrangements)
- Central Estates (Belgravia) v Woolgar (waiver)
- Billson v Residential Apartments (peaceable re-entry)

Only mention these cases if they are relevant to the student's answer and appear in the model answer.
<|im_end|>
<|im_start|>user
Compare this student answer with the model answer. Analyze:
1. Explanation of lease vs license importance
2. Application of Street v Mountford test
3. Analysis of commercial context using Esso v Fumegrange
4. Discussion of sham clauses (Aslan v Murphy)
5. Treatment of forfeiture process

Teacher's Answer: {teacher_answer}

Student's Answer: {student_answer}

Provide specific feedback in these three sections:
STRENGTHS:
WEAKNESSES:
SUGGESTIONS FOR IMPROVEMENT:
<|im_end|>
<|im_start|>assistant
STRENGTHS:
• """
        
        response = llm.invoke(
            input=comparison_prompt,
            temperature=0.01,
            max_tokens=2000,
            stop=["</s>", "<|im_end|>", "<|im_start|>"]
        )
        
        # Validate legal references
        if not validate_legal_references(response, teacher_answer):
            st.error("Generated feedback contains invalid case references. Regenerating...")
            return None
            
        # Ensure response has all sections
        if not "WEAKNESSES:" in response:
            response += "\n\nWEAKNESSES:\n• "
        if not "SUGGESTIONS FOR IMPROVEMENT:" in response:
            response += "\n\nSUGGESTIONS FOR IMPROVEMENT:\n1. "
        
        return response
            
    except Exception as e:
        st.error(f"Error generating feedback: {str(e)}")
        return "Error generating feedback. Please try again."

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

def main():
    st.title("Feedback Analysis System")
    
    # Load suggested answer with correct path
    suggested_answer = load_pdf_as_text("data/Suggested Answer.pdf")
    
    # Load training data first
    with st.spinner('Loading training examples...'):
        training_data = load_training_data()
        training_examples = create_training_prompt(training_data)
    
    uploaded_file = st.file_uploader(
        "Upload Student Answer (PDF or Word)", 
        type=["pdf", "docx", "doc"],
        key="file_uploader"
    )
    
    if uploaded_file:
        with st.spinner('Processing submission...'):
            try:
                student_text = process_student_upload(uploaded_file)
                
                if student_text:
                    feedback = compare_answers(
                        teacher_answer=suggested_answer,
                        student_answer=student_text,
                        training_examples=training_examples
                    )
                    
                    st.write("### Generated Feedback")
                    st.markdown(feedback)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Add chat interface
    st.write("### Chat with the System")
    user_question = st.text_input("Ask a question about the answers:")
    
    if user_question:
        # Create vectordb if not exists
        if "vectordb" not in st.session_state:
            texts = [suggested_answer]
            if student_text:
                texts.append(student_text)
            
            vectordb = Chroma.from_texts(
                texts,
                embeddings,
                metadatas=[{"source": i} for i in range(len(texts))]
            )
            st.session_state.vectordb = vectordb
        
        # Get response
        response = chat_with_rag(
            user_question, 
            st.session_state.vectordb,
            student_text=student_text,
            teacher_answer=suggested_answer
        )
        st.write(response)

if __name__ == "__main__":
    main()
