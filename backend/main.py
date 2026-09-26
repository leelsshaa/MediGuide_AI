from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "MediaGuideAI Backend is Running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    contents = await file.read()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "message": "File uploaded successfully"
    }