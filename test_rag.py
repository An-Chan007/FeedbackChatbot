from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

# Import necessary functions from main2.py
from main23 import compare_answers, process_student_upload, load_pdf_as_text

# Use the same paths as main2.py
TEACHER_ANSWER_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Suggested Answer.pdf"
STUDENT_UPLOADS_PATH =  r"C:\Users\Alcatraz\Desktop\Student Answer A.pdf"  # Updated with your student answer path
LICENCE_TO_OCCUPY_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Licence to Occupy.pdf"
QUESTION_CONTEXT_PATH = r"C:\Users\Alcatraz\Desktop\rag-tutorial-v2-main\data\Formative Question.pdf"

EVAL_PROMPT = """
Expected Feedback Structure:
1. Should have clear STRENGTHS section
2. Should have clear WEAKNESSES section
3. Should have clear SUGGESTIONS FOR IMPROVEMENT section
4. Should contain specific legal concepts and case citations
5. Should not repeat student's answer

Actual Response: {actual_response}
---
Evaluate the feedback quality (Answer with 'true' or 'false'):
1. Does it have all three required sections?
2. Does it provide specific legal feedback?
3. Does it avoid repeating the student's answer?
4. Is it properly formatted?

All criteria must be met for a 'true' response.
"""

def test_legal_feedback():
    """Test if the feedback system provides properly structured responses"""
    assert query_and_validate(
        test_pdf_path=STUDENT_UPLOADS_PATH,
        teacher_answer_path=TEACHER_ANSWER_PATH,
        question_context_path=QUESTION_CONTEXT_PATH
    )

def test_feedback_format():
    """Test if the feedback follows the required format"""
    assert query_and_validate(
        test_pdf_path=STUDENT_UPLOADS_PATH,
        teacher_answer_path=TEACHER_ANSWER_PATH,
        question_context_path=QUESTION_CONTEXT_PATH
    )

def query_and_validate(test_pdf_path: str, teacher_answer_path: str, question_context_path: str):
    # Load test files
    student_text = load_pdf_as_text(test_pdf_path)
    teacher_answer = load_pdf_as_text(teacher_answer_path)
    question_context = load_pdf_as_text(question_context_path)
    
    # Generate feedback
    response_text = compare_answers(question_context, teacher_answer, student_text)
    
    # Evaluate the response
    prompt = EVAL_PROMPT.format(actual_response=response_text)
    
    # Initialize the model (same as in main2.py)
    model = OpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        model="mistral-7b-instruct-v0.2",
        temperature=0.3
    )

    # Create evaluation chain
    eval_prompt = PromptTemplate.from_template(prompt)
    chain = eval_prompt | model
    evaluation_results = chain.invoke({})
    evaluation_results_cleaned = evaluation_results.strip().lower()

    print("\nTest Results:")
    print("-------------")
    print(f"Generated Feedback:\n{response_text}\n")
    print("Evaluation:")
    
    if "true" in evaluation_results_cleaned:
        print("\033[92m" + "✓ Feedback meets all quality criteria" + "\033[0m")
        return True
    elif "false" in evaluation_results_cleaned:
        print("\033[91m" + "✗ Feedback does not meet all quality criteria" + "\033[0m")
        return False
    else:
        raise ValueError("Invalid evaluation result")

if __name__ == "__main__":
    print("Running Legal Feedback System Tests...")
    try:
        test_legal_feedback()
        test_feedback_format()
        print("\033[92m" + "\nAll tests passed successfully!" + "\033[0m")
    except AssertionError:
        print("\033[91m" + "\nSome tests failed. Please check the output above." + "\033[0m")
