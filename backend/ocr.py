import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from .env

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=api_key)


def extract_text(image_path):
    """
    Extract text from a discharge-summary image using Gemini.
    """

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),
            """
            Extract all visible text from this medical discharge summary.

            Important rules:
            - Preserve medicine names exactly.
            - Preserve dosage, frequency and duration.
            - Preserve dates and follow-up instructions.
            - Preserve food/diet instructions.
            - Do NOT guess unclear text.
            - Do NOT invent missing information.
            - Return ONLY the extracted text.
            """
        ]
    )

    return response.text or ""