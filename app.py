from dotenv import load_dotenv
from openai import OpenAI
import os

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import streamlit as st
from backend.parser import extract_text_from_pdf, extract_text_from_txt
from backend.summarizer import summarize_text
from backend.qa_engine import build_vector_store, get_top_chunks
from backend.question_generator import generate_mcqs
import tempfile
import re
import textwrap

# Initialize session state for memory
if 'history' not in st.session_state:
    st.session_state['history'] = []

# App UI
st.set_page_config(page_title="SmartDocAI", layout="wide")
st.title("📄 SmartDocAI — Document-Aware Assistant")
st.caption("Upload a PDF or TXT document")

# Upload Section
uploaded_file = st.file_uploader("Drag and drop file here", type=["pdf", "txt"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Extract Text
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_file_path)
    else:
        text = extract_text_from_txt(tmp_file_path)

    os.remove(tmp_file_path)  # Cleanup

    # Summarize
    with st.spinner("Summarizing document..."):
        summary = summarize_text(text)
    st.subheader("📝 Auto Summary (≤150 words)")
    st.success(summary)

    # Chunk text for embedding
    text_chunks = re.split(r'\n{2,}|\.\s+', text)
    text_chunks = [chunk.strip() for chunk in text_chunks if len(chunk.strip()) > 30]
    index, embeddings = build_vector_store(text_chunks)

    # Interaction Mode
    st.subheader("🔍 Choose Interaction Mode:")
    mode = st.radio("Select a mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        st.subheader("💬 Ask Anything")
        user_question = st.text_input("Type your question:", key="question_input")

        if user_question:
            with st.spinner("Thinking..."):
                try:
                    # Get history context
                    history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state['history']])
                    full_prompt = f"{history_context}\nQ: {user_question}"

                    # Vector retrieval
                    top_chunks = get_top_chunks(full_prompt, text_chunks, index, embeddings)
                    context = "\n".join(top_chunks)

                    # OpenAI Chat Completion (new v1.0+ syntax)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who answers based only on the document content."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
                        ]
                    )

                    answer = response.choices[0].message.content

                except Exception as e:
                    answer = f"⚠️ Error occurred: {str(e)}"

                # Save to memory
                st.session_state['history'].append((user_question, answer))

            # Display answer
            st.subheader("📌 Answer:")
            st.info(answer)

        # Show past Q&A
        if st.session_state['history']:
            st.subheader("🕘 Previous Q&A")
            for i, (q, a) in enumerate(st.session_state['history']):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")

    elif mode == "Challenge Me":
        st.markdown("### 🧠 Let's test your understanding!")

        with st.spinner("Generating challenge questions..."):
            questions = generate_mcqs(text)

        for i, q in enumerate(questions, 1):
            st.write(f"**Q{i}:** {q['question']}")
            options = q['options']
            selected = st.radio("Choose one:", options, key=f"q{i}")

            if selected == q["answer"]:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. Correct answer: {q['answer']}")

            # Show justification if available
            justification = q.get("justification")
            if justification:
                st.caption(f"📖 Justification: {justification}")
