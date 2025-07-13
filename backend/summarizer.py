from transformers import pipeline

# Load the model once when the module loads
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_words=150):
    """
    Summarizes long text into a 150-word summary.
    """
    # Truncate input if it's too long for the model
    max_input_length = 1024  # tokens
    text = text[:3000]  # approx. 1024 tokens

    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    return summary.strip()
