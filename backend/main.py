from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from backend.ocr import extract_text

app = FastAPI()

# Folder where uploaded files are temporarily stored
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "MediaGuideAI Backend is Running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Validate file type
    allowed_types = [
        "image/png",
        "image/jpeg",
        "application/pdf"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPEG, and PDF files are supported."
        )

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Extract text using Gemini
        extracted_text = extract_text(str(file_path))

        return {
            "filename": file.filename,
            "status": "success",
            "extracted_text": extracted_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed: {str(e)}"
        )