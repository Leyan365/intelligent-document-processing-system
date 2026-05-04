import pdfplumber
with pdfplumber.open("Production AI Tutorial.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
with open("pdf_text.txt", "w", encoding="utf-8") as out:
    out.write(text)
