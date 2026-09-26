from ocr import extract_text

text = extract_text("backend/uploads/test.png")

print("\n--- OCR RESULT ---\n")
print(text)