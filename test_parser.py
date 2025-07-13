from backend import parser

print("Starting test...")

try:
    with open("coorporate event.pdf", "rb") as f:
        print("File opened ✅")
        text = parser.extract_text_from_pdf(f)
        print("Text extracted ✅")
        print(text[:1000])
except FileNotFoundError:
    print("❌ coorporate event.pdf not found")
except Exception as e:
    print("❌ Error occurred:", e)

