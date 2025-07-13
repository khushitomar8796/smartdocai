from backend import summarizer

text = """
SmartDocAI is an assistant that helps users understand documents by summarizing and answering questions.
It reads PDF and TXT files, extracts relevant information, and can even quiz the user on document logic.
This allows researchers, students, and professionals to save time reading large documents and get to the important parts faster.
"""

summary = summarizer.summarize_text(text)
print("SUMMARY:\n", summary)
