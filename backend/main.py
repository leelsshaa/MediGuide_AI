from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pathlib import Path
from backend.ocr import extract_text
from ai_module.app import generate_patient_plan

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
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form("English")
):
    if language not in ["English", "Tamil"]:
        raise HTTPException(
            status_code=400,
            detail="Language must be English or Tamil."
        )
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

        # Send OCR text to AI module
        patient_plan = generate_patient_plan(
            extracted_text,
            language=language
        )

        return {
            "filename": file.filename,
            "status": "success",
            "extracted_text": extracted_text,
            "patient_plan": patient_plan
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed: {str(e)}"
        )