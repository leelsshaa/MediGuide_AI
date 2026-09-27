
import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from gtts import gTTS


# ============================================================
# LOAD API KEY
# ============================================================

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to the .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# PATIENT NAME EXTRACTION
# ============================================================

def extract_patient_name(text):
    """
    Extract patient name locally before PII sanitization.

    The name is NOT sent to Gemini.
    """

    patterns = [
        r"(?im)^\s*Patient\s+Name\s*[:#-]\s*(.+?)\s*$",
        r"(?im)^\s*Name\s+of\s+Patient\s*[:#-]\s*(.+?)\s*$",
        r"(?im)^\s*Name\s*[:#-]\s*(.+?)\s*$"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(text))

        if match:
            name = match.group(1).strip()

            if name:
                return name

    return ""


# ============================================================
# PRIVACY / PII PROTECTION
# ============================================================

def sanitize_pii(text):
    """
    Remove unnecessary personally identifiable information
    before sending discharge-summary text to the AI model.

    Medical information required for processing is preserved.
    """

    sanitized = str(text)

    # --------------------------------------------------------
    # Email addresses
    # --------------------------------------------------------

    sanitized = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
        sanitized
    )

    # --------------------------------------------------------
    # Indian phone numbers
    # --------------------------------------------------------

    sanitized = re.sub(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        "[REDACTED_PHONE]",
        sanitized
    )

    # --------------------------------------------------------
    # Aadhaar-like 12 digit numbers
    # --------------------------------------------------------

    sanitized = re.sub(
        r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
        "[REDACTED_ID]",
        sanitized
    )

    # --------------------------------------------------------
    # Patient identifiers after common labels
    # --------------------------------------------------------

    patient_id_pattern = (
        r"(?im)"
        r"\b(?:patient\s+(?:id|no|number)|"
        r"patient\s+identifier|"
        r"uhid|"
        r"mrn|"
        r"medical\s+record\s+(?:no|number))"
        r"\s*[:#-]\s*[^\n,]+"
    )

    sanitized = re.sub(
        patient_id_pattern,
        lambda match: re.sub(
            r"(?i)([:#-]\s*).*$",
            r"\1[REDACTED_ID]",
            match.group(0)
        ),
        sanitized
    )

    # --------------------------------------------------------
    # Patient name after common labels
    # --------------------------------------------------------

    patient_name_pattern = (
        r"(?im)"
        r"\b(?:patient\s+name|"
        r"name\s+of\s+patient)"
        r"\s*[:#-]\s*[^\n,]+"
    )

    sanitized = re.sub(
        patient_name_pattern,
        lambda match: re.sub(
            r"(?i)([:#-]\s*).*$",
            r"\1[REDACTED_NAME]",
            match.group(0)
        ),
        sanitized
    )

    # --------------------------------------------------------
    # Address after common labels
    # --------------------------------------------------------

    address_pattern = (
        r"(?im)"
        r"\b(?:address|patient\s+address)"
        r"\s*[:#-]\s*[^\n]+"
    )

    sanitized = re.sub(
        address_pattern,
        lambda match: re.sub(
            r"(?i)([:#-]\s*).*$",
            r"\1[REDACTED_ADDRESS]",
            match.group(0)
        ),
        sanitized
    )

    return sanitized


# ============================================================
# FREQUENCY CONVERSION
# ============================================================

def get_patient_friendly_frequency(frequency):

    frequency_map = {
        "OD": "Once daily",
        "BD": "Twice daily",
        "TDS": "Three times daily",
        "QID": "Four times daily",
        "QD": "Once daily"
    }

    frequency_upper = str(frequency).strip().upper()

    return frequency_map.get(
        frequency_upper,
        frequency
    )


# ============================================================
# TAMIL FREQUENCY
# ============================================================

def get_tamil_frequency(frequency):

    frequency_map = {
        "OD": "நாளைக்கு ஒரு முறை",
        "BD": "நாளைக்கு இரண்டு முறை",
        "TDS": "நாளைக்கு மூன்று முறை",
        "QID": "நாளைக்கு நான்கு முறை",
        "QD": "நாளைக்கு ஒரு முறை"
    }

    frequency_upper = str(frequency).strip().upper()

    return frequency_map.get(
        frequency_upper,
        frequency
    )


# ============================================================
# DURATION CONVERSION
# ============================================================

def get_duration_days(duration):

    duration_text = str(duration).lower().strip()

    day_match = re.search(
        r"(\d+)\s*days?",
        duration_text
    )

    if day_match:
        return int(day_match.group(1))

    week_match = re.search(
        r"(\d+)\s*weeks?",
        duration_text
    )

    if week_match:
        return int(week_match.group(1)) * 7

    month_match = re.search(
        r"(\d+)\s*months?",
        duration_text
    )

    if month_match:
        return int(month_match.group(1)) * 30

    return None


# ============================================================
# TAMIL DURATION
# ============================================================

def get_tamil_duration(duration):

    duration_text = str(duration).lower().strip()

    day_match = re.search(
        r"(\d+)\s*days?",
        duration_text
    )

    if day_match:
        days = day_match.group(1)
        return f"{days} நாட்களுக்கு"

    week_match = re.search(
        r"(\d+)\s*weeks?",
        duration_text
    )

    if week_match:
        weeks = week_match.group(1)

        if weeks == "1":
            return "1 வாரத்திற்கு"

        return f"{weeks} வாரங்களுக்கு"

    month_match = re.search(
        r"(\d+)\s*months?",
        duration_text
    )

    if month_match:
        months = month_match.group(1)

        if months == "1":
            return "1 மாதத்திற்கு"

        return f"{months} மாதங்களுக்கு"

    return duration


# ============================================================
# TRANSLATE INSTRUCTIONS TO TAMIL
# ============================================================

def translate_instruction_to_tamil(instruction):

    text = str(instruction).lower().strip()

    translations = {
        "after breakfast": "காலை உணவுக்குப் பிறகு",
        "before breakfast": "காலை உணவுக்கு முன்",
        "after food": "உணவுக்குப் பிறகு",
        "before food": "உணவுக்கு முன்",
        "with meals": "உணவுடன்",
        "after lunch": "மதிய உணவுக்குப் பிறகு",
        "before lunch": "மதிய உணவுக்கு முன்",
        "after dinner": "இரவு உணவுக்குப் பிறகு",
        "before dinner": "இரவு உணவுக்கு முன்"
    }

    return translations.get(
        text,
        instruction
    )


# ============================================================
# TRANSLATE WARNING SIGNS TO TAMIL
# ============================================================

def translate_warning_to_tamil(warning):

    text = str(warning).lower().strip()

    replacements = [
        (
            "severe chest pain",
            "கடுமையான நெஞ்சு வலி"
        ),
        (
            "difficulty breathing",
            "மூச்சு விடுவதில் சிரமம்"
        ),
        (
            "severe weakness",
            "கடுமையான பலவீனம்"
        ),
        (
            "fainting",
            "மயக்கம்"
        ),
        (
            "weakness",
            "பலவீனம்"
        ),
        (
            "dizziness",
            "தலைச்சுற்றல்"
        ),
        (
            "vomiting",
            "வாந்தி"
        ),
        (
            "high fever",
            "அதிக காய்ச்சல்"
        )
    ]

    for english, tamil in replacements:
        text = text.replace(
            english,
            tamil
        )

    text = re.sub(
        r"\s+and\s+",
        " மற்றும் ",
        text,
        flags=re.IGNORECASE
    )

    return text


# ============================================================
# CREATE DAILY SCHEDULE
# ============================================================

def create_daily_schedule(result):

    for medicine in result.get("medicines", []):

        # ----------------------------------------------------
        # Do not create a schedule for an unsafe medicine.
        # ----------------------------------------------------

        if medicine.get("status") == "unclear":

            medicine["daily_schedule"] = (
                "Schedule not generated because medication "
                "information is unclear."
            )

            medicine["time_of_day"] = "Not generated"
            medicine["duration_days"] = None

            continue

        frequency = medicine.get(
            "frequency",
            ""
        )

        instructions = medicine.get(
            "instructions",
            ""
        )

        patient_frequency = (
            get_patient_friendly_frequency(
                frequency
            )
        )

        medicine["frequency_patient_friendly"] = (
            patient_frequency
        )

        duration_days = get_duration_days(
            medicine.get(
                "duration",
                ""
            )
        )

        medicine["duration_days"] = duration_days

        instruction_text = str(
            instructions
        ).lower()

        # ----------------------------------------------------
        # Prescribed time extraction
        # ----------------------------------------------------

        prescribed_times = re.findall(
            r"\b(?:at\s*)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
            instruction_text,
            flags=re.IGNORECASE
        )

        medicine["prescribed_times"] = [
            time.strip()
            for time in prescribed_times
        ]

        # ----------------------------------------------------
        # Food timing
        # ----------------------------------------------------

        if "after breakfast" in instruction_text:
            medicine["food_timing"] = "After breakfast"

        elif "before breakfast" in instruction_text:
            medicine["food_timing"] = "Before breakfast"

        elif "after lunch" in instruction_text:
            medicine["food_timing"] = "After lunch"

        elif "before lunch" in instruction_text:
            medicine["food_timing"] = "Before lunch"

        elif "after dinner" in instruction_text:
            medicine["food_timing"] = "After dinner"

        elif "before dinner" in instruction_text:
            medicine["food_timing"] = "Before dinner"

        elif "after food" in instruction_text:
            medicine["food_timing"] = "After food"

        elif "before food" in instruction_text:
            medicine["food_timing"] = "Before food"

        elif "with meals" in instruction_text:
            medicine["food_timing"] = "With meals"

        else:
            medicine["food_timing"] = ""

        # ----------------------------------------------------
        # Schedule
        # ----------------------------------------------------

        if "breakfast" in instruction_text:

            medicine["time_of_day"] = "Morning"

            medicine["daily_schedule"] = (
                f"Morning – {instructions}"
            )

        elif "lunch" in instruction_text:

            medicine["time_of_day"] = "Afternoon"

            medicine["daily_schedule"] = (
                f"Afternoon – {instructions}"
            )

        elif "dinner" in instruction_text:

            medicine["time_of_day"] = "Night"

            medicine["daily_schedule"] = (
                f"Night – {instructions}"
            )

        elif "morning" in instruction_text:

            medicine["time_of_day"] = "Morning"

            medicine["daily_schedule"] = (
                f"Morning – {instructions}"
            )

        elif "evening" in instruction_text:

            medicine["time_of_day"] = "Evening"

            medicine["daily_schedule"] = (
                f"Evening – {instructions}"
            )

        elif "night" in instruction_text:

            medicine["time_of_day"] = "Night"

            medicine["daily_schedule"] = (
                f"Night – {instructions}"
            )

        elif (
            "after food" in instruction_text
            or "before food" in instruction_text
        ):

            medicine["time_of_day"] = "Meal time"

            medicine["daily_schedule"] = (
                f"{patient_frequency} – "
                f"{instructions} "
                "(exact time not specified)"
            )

        else:

            medicine["time_of_day"] = (
                "Time not specified"
            )

            medicine["daily_schedule"] = (
                f"{patient_frequency} – "
                "exact time not specified"
            )

    return result


# ============================================================
# AI EXTRACTION + LANGUAGE SUPPORT
# ============================================================

def extract_discharge_info(raw_text, language):

    # --------------------------------------------------------
    # PRIVACY STEP
    # --------------------------------------------------------

    sanitized_text = sanitize_pii(
        raw_text
    )

    if language == "Tamil":

        language_instruction = """
Translate patient-facing information into simple, natural Tamil.

Do not change:
- medicine names
- dosage values
- original frequency abbreviations
- original duration values

Do not add medical information that is not present.
"""

    else:

        language_instruction = """
Use simple patient-friendly English.

Do not add medical information that is not present.
"""

    prompt = f"""
You are an information extraction system for hospital discharge summaries.

Extract ONLY information explicitly present in the provided text.

DO NOT:
- guess
- infer
- invent
- medically correct the source
- change a stated dose or frequency
- create a medication time that is not present

{language_instruction}

Return ONLY valid JSON.

Do not add explanations or markdown.

Use exactly this structure:

{{
    "medicines": [
        {{
            "name": "",
            "dosage": "",
            "frequency": "",
            "duration": "",
            "instructions": "",
            "status": ""
        }}
    ],
    "food_instructions": [],
    "warning_signs": [],
    "follow_up": {{
        "timing": "",
        "instructions": ""
    }},
    "patient_friendly_warning_signs": [],
    "patient_friendly_follow_up": {{
        "timing": "",
        "instructions": ""
    }},
    "safety_status": ""
}}

IMPORTANT:

Common medication abbreviations such as:

- OD = once daily
- BD = twice daily
- TDS = three times daily
- QID = four times daily

are CLEAR instructions and must NOT be marked as unclear.

Mark a medicine as "unclear" ONLY when the source itself contains
an ambiguous, incomplete, missing, or unreadable instruction.

Examples:

- ?D
- ??
- missing dose
- incomplete frequency
- unreadable handwriting

For clear medication information:

"status" = "clear"

For ambiguous medication information:

"status" = "unclear"

Set "safety_status" to:

- "CLEAR" when medication instructions are clear
- "UNCLEAR" when important medication information is ambiguous

Extract food instructions ONLY when explicitly present.

Extract warning signs ONLY when explicitly present.

Extract follow-up information ONLY when explicitly present.

Selected patient language:

{language}

DISCHARGE SUMMARY TEXT:

{sanitized_text}
"""

    try:

        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt
        )

        cleaned = response.text.strip()

        # ----------------------------------------------------
        # Remove markdown code fences if Gemini returns them.
        # ----------------------------------------------------

        cleaned = re.sub(
            r"^```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"^```\s*",
            "",
            cleaned
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

        return json.loads(cleaned)

    except Exception:

        print(
            "\nGemini temporarily unavailable."
        )

        print(
            "AI processing could not be completed safely.\n"
        )

        return {
            "medicines": [],
            "food_instructions": [],
            "warning_signs": [],
            "follow_up": {
                "timing": "",
                "instructions": ""
            },
            "patient_friendly_warning_signs": [],
            "patient_friendly_follow_up": {
                "timing": "",
                "instructions": ""
            },
            "safety_status": "AI_UNAVAILABLE"
        }


# ============================================================
# SAFETY REFUSAL
# ============================================================

def set_safe_refusal(result, language, reason):

    result["safety_status"] = (
        "REFUSED - " + reason
    )

    result["refusal"] = True

    if language == "Tamil":

        result["refusal_message"] = (
            "மருந்து பற்றிய தகவல் தெளிவாக இல்லை. "
            "மருத்துவமனையை தொடர்பு கொண்டு உறுதி செய்யவும்."
        )

    else:

        result["refusal_message"] = (
            "Medication information is unclear. "
            "Please contact the hospital to confirm it."
        )

    for medicine in result.get(
        "medicines",
        []
    ):

        if medicine.get("status") != "clear":

            medicine["status"] = "unclear"

            if language == "Tamil":

                medicine["patient_friendly"] = (
                    "மருந்து பற்றிய தகவல் தெளிவாக இல்லை. "
                    "மருத்துவமனையில் உறுதி செய்யவும்."
                )

            else:

                medicine["patient_friendly"] = (
                    "Medication information is unclear. "
                    "Please confirm with the hospital."
                )

            medicine["confidence"] = "low"

    return result


def safety_check(result, raw_text, language):

    text = str(raw_text).lower()

    # --------------------------------------------------------
    # AI itself could not process the document
    # --------------------------------------------------------

    if result.get("safety_status") == "AI_UNAVAILABLE":

        result["refusal"] = True

        if language == "Tamil":

            result["refusal_message"] = (
                "தகவலை பாதுகாப்பாக செயல்படுத்த முடியவில்லை. "
                "மருத்துவமனையை தொடர்பு கொள்ளவும்."
            )

        else:

            result["refusal_message"] = (
                "The information could not be processed safely. "
                "Please contact the hospital."
            )

        return result

    # --------------------------------------------------------
    # No medicine information extracted
    # --------------------------------------------------------

    medicines = result.get(
        "medicines",
        []
    )

    if not medicines:

        return set_safe_refusal(
            result,
            language,
            "No medication information could be reliably extracted."
        )

    # --------------------------------------------------------
    # Explicit ambiguity in original document
    # --------------------------------------------------------

    ambiguity_patterns = [
        r"\?d\b",
        r"\?\?",
        r"\?\s*mg\b",
        r"\bmg\s*\?",
        r"\btablet\s*\?",
        r"\btab\s*\?"
    ]

    for pattern in ambiguity_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            return set_safe_refusal(
                result,
                language,
                "Medication instruction is ambiguous."
            )

    # --------------------------------------------------------
    # Check every extracted medicine
    # --------------------------------------------------------

    for medicine in medicines:

        name = str(
            medicine.get(
                "name",
                ""
            )
        ).strip()

        dosage = str(
            medicine.get(
                "dosage",
                ""
            )
        ).strip()

        frequency = str(
            medicine.get(
                "frequency",
                ""
            )
        ).strip()

        duration = str(
            medicine.get(
                "duration",
                ""
            )
        ).strip()

        instructions = str(
            medicine.get(
                "instructions",
                ""
            )
        ).strip()

        status = str(
            medicine.get(
                "status",
                ""
            )
        ).lower().strip()

        # Critical medicine information must be present
        if not name or not dosage or not frequency or not duration:
            medicine["status"] = "unclear"
            medicine["confidence"] = "low"

            set_safe_refusal(
                result,
                language,
                "Medication information is incomplete or unclear. Please contact the hospital to confirm it."
            )

            result["safety_status"] = "REFUSED"
            result["refusal"] = True


            continue

        # ----------------------------------------------------
        # AI explicitly marked the medicine unclear
        # ----------------------------------------------------

        if status == "unclear":

            return set_safe_refusal(
                result,
                language,
                "Medication information is ambiguous."
            )

        # ----------------------------------------------------
        # Detect unclear words in important fields
        # ----------------------------------------------------

        important_fields = [
            dosage,
            frequency,
            duration,
            instructions
        ]

        if any(
            "unclear" in field.lower()
            or "ambiguous" in field.lower()
            or "unreadable" in field.lower()
            for field in important_fields
        ):

            return set_safe_refusal(
                result,
                language,
                "Medication information is ambiguous."
            )

        # ----------------------------------------------------
        # Missing medicine name
        # ----------------------------------------------------

        if not name:

            return set_safe_refusal(
                result,
                language,
                "Medication name could not be reliably read."
            )

        # ----------------------------------------------------
        # Missing dosage
        # ----------------------------------------------------

        if not dosage:

            return set_safe_refusal(
                result,
                language,
                "Medication dose could not be reliably read."
            )

        # ----------------------------------------------------
        # Missing frequency
        # ----------------------------------------------------

        if not frequency:

            return set_safe_refusal(
                result,
                language,
                "Medication frequency could not be reliably read."
            )

        # ----------------------------------------------------
        # Missing duration
        # ----------------------------------------------------

        if not duration:

            return set_safe_refusal(
                result,
                language,
                "Medication duration could not be reliably read."
            )

        # ----------------------------------------------------
        # Explicit question marks inside medication fields
        # ----------------------------------------------------

        if any(
            "?" in field
            for field in [
                name,
                dosage,
                frequency,
                duration,
                instructions
            ]
        ):

            return set_safe_refusal(
                result,
                language,
                "Medication information contains an ambiguous value."
            )

        medicine["status"] = "clear"

        # ----------------------------------------------------
        # Confidence is based on safety validation.
        # We do NOT let Gemini invent a confidence score.
        # ----------------------------------------------------

        medicine["confidence"] = "high"

    # --------------------------------------------------------
# Final safety validation
# --------------------------------------------------------

# If any medicine is unclear, the complete plan must be refused.
    if any(
        str(medicine.get("status", "")).lower().strip() == "unclear"
        for medicine in medicines
    ):
        result["safety_status"] = "REFUSED"
        result["refusal"] = True

        return result

    # All medicines passed the safety checks
    result["safety_status"] = "CLEAR"
    result["refusal"] = False

    result.pop(
        "refusal_message",
        None
    )

    return result


# ============================================================
# ADD PATIENT-FRIENDLY CONTENT
# ============================================================

def add_patient_friendly_content(result, language):

    # --------------------------------------------------------
    # Do not create a normal patient plan after refusal.
    # --------------------------------------------------------

    if result.get("refusal") is True:
        return result

    if language == "Tamil":

        for medicine in result.get(
            "medicines",
            []
        ):

            if medicine.get(
                "status"
            ) != "clear":
                continue

            name = medicine.get(
                "name",
                ""
            )

            dosage = medicine.get(
                "dosage",
                ""
            )

            frequency = medicine.get(
                "frequency",
                ""
            )

            duration = medicine.get(
                "duration",
                ""
            )

            instructions = medicine.get(
                "instructions",
                ""
            )

            tamil_frequency = (
                get_tamil_frequency(
                    frequency
                )
            )

            tamil_duration = (
                get_tamil_duration(
                    duration
                )
            )

            tamil_instruction = (
                translate_instruction_to_tamil(
                    instructions
                )
            )

            medicine["patient_friendly"] = (
                f"{name} {dosage}, "
                f"{tamil_frequency}, "
                f"{tamil_duration}, "
                f"{tamil_instruction} எடுத்துக்கொள்ளவும்."
            )

            medicine["frequency_patient_friendly"] = (
                tamil_frequency
            )

            instruction_text = str(
                instructions
            ).lower()

            if "breakfast" in instruction_text:

                tamil_schedule = (
                    f"காலை – {tamil_instruction}"
                )

            elif "lunch" in instruction_text:

                tamil_schedule = (
                    f"மதியம் – {tamil_instruction}"
                )

            elif "dinner" in instruction_text:

                tamil_schedule = (
                    f"இரவு – {tamil_instruction}"
                )

            elif "morning" in instruction_text:

                tamil_schedule = (
                    f"காலை – {tamil_instruction}"
                )

            elif "evening" in instruction_text:

                tamil_schedule = (
                    f"மாலை – {tamil_instruction}"
                )

            elif "night" in instruction_text:

                tamil_schedule = (
                    f"இரவு – {tamil_instruction}"
                )

            elif (
                "after food" in instruction_text
                or "before food" in instruction_text
            ):

                tamil_schedule = (
                    f"{tamil_frequency} – "
                    f"{tamil_instruction} "
                    "(சரியான நேரம் குறிப்பிடப்படவில்லை)"
                )

            else:

                tamil_schedule = (
                    f"{tamil_frequency} – "
                    "சரியான நேரம் குறிப்பிடப்படவில்லை"
                )

            medicine[
                "daily_schedule_patient_friendly"
            ] = tamil_schedule

        # ----------------------------------------------------
        # Food instructions
        # ----------------------------------------------------

        translated_food = []

        for instruction in result.get(
            "food_instructions",
            []
        ):

            translated_food.append(
                translate_instruction_to_tamil(
                    instruction
                )
            )

        result[
            "patient_friendly_food_instructions"
        ] = list(
            dict.fromkeys(
                translated_food
            )
        )

        # ----------------------------------------------------
        # Warning signs
        # ----------------------------------------------------

        translated_warnings = []

        for warning in result.get(
            "warning_signs",
            []
        ):

            translated_warnings.append(
                translate_warning_to_tamil(
                    warning
                )
            )

        result[
            "patient_friendly_warning_signs"
        ] = translated_warnings

        # ----------------------------------------------------
        # Follow-up
        # ----------------------------------------------------

        follow_up = result.get(
            "patient_friendly_follow_up",
            {}
        )

        result[
            "patient_friendly_follow_up"
        ] = follow_up

    else:

        for medicine in result.get(
            "medicines",
            []
        ):

            if medicine.get(
                "status"
            ) != "clear":
                continue

            name = medicine.get(
                "name",
                ""
            )

            dosage = medicine.get(
                "dosage",
                ""
            )

            frequency = medicine.get(
                "frequency",
                ""
            )

            duration = medicine.get(
                "duration",
                ""
            )

            instructions = medicine.get(
                "instructions",
                ""
            )

            patient_frequency = (
                get_patient_friendly_frequency(
                    frequency
                )
            )

            medicine["patient_friendly"] = (
                f"{name} {dosage}, "
                f"{patient_frequency}, "
                f"{duration}, "
                f"{instructions}"
            )

            medicine[
                "daily_schedule_patient_friendly"
            ] = medicine.get(
                "daily_schedule",
                ""
            )

        result[
            "patient_friendly_food_instructions"
        ] = result.get(
            "food_instructions",
            []
        )

        result[
            "patient_friendly_warning_signs"
        ] = result.get(
            "warning_signs",
            []
        )

    return result


# ============================================================
# CREATE AUDIO TEXT
# ============================================================

def create_audio_text(result, language):

    parts = []

    # --------------------------------------------------------
    # Refusal message
    # --------------------------------------------------------

    if result.get("refusal") is True:

        refusal_message = result.get(
            "refusal_message",
            ""
        )

        if refusal_message:

            parts.append(
                refusal_message
            )

        return " ".join(parts)

    # --------------------------------------------------------
    # Clear medicine information only
    # --------------------------------------------------------

    for index, medicine in enumerate(
        result.get(
            "medicines",
            []
        ),
        start=1
    ):

        if medicine.get(
            "status"
        ) != "clear":
            continue

        patient_friendly = medicine.get(
            "patient_friendly",
            ""
        )

        if not patient_friendly:
            continue

        if language == "Tamil":

            parts.append(
                f"மருந்து {index}. "
                f"{patient_friendly}"
            )

        else:

            parts.append(
                f"Medicine {index}. "
                f"{patient_friendly}"
            )

    # --------------------------------------------------------
    # Warning signs
    # --------------------------------------------------------

    warnings = result.get(
        "patient_friendly_warning_signs",
        []
    )

    if warnings:

        if language == "Tamil":

            parts.append(
                "எச்சரிக்கை அறிகுறிகள்: "
                + ", ".join(warnings)
            )

        else:

            parts.append(
                "Warning signs: "
                + ", ".join(warnings)
            )

    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    follow_up = result.get(
        "patient_friendly_follow_up",
        {}
    )

    follow_up_timing = follow_up.get(
        "timing",
        ""
    )

    follow_up_instructions = follow_up.get(
        "instructions",
        ""
    )

    if language == "Tamil":

        if follow_up_timing:

            parts.append(
                f"மீண்டும் மருத்துவமனைக்கு வர வேண்டிய நேரம்: "
                f"{follow_up_timing}."
            )

        if follow_up_instructions:

            parts.append(
                f"கொண்டு வர வேண்டியவை: "
                f"{follow_up_instructions}"
            )

    else:

        if follow_up_timing:

            parts.append(
                f"Follow-up: {follow_up_timing}."
            )

        if follow_up_instructions:

            parts.append(
                f"Bring: {follow_up_instructions}"
            )

    return " ".join(parts)


# ============================================================
# GENERATE AUDIO
# ============================================================

def generate_audio(result, language):

    audio_text = create_audio_text(
        result,
        language
    )

    if not audio_text.strip():
        return result

    os.makedirs(
        "audio_output",
        exist_ok=True
    )

    audio_file = os.path.join(
        "audio_output",
        "patient_plan_audio.mp3"
    )

    try:

        language_code = (
            "ta"
            if language == "Tamil"
            else "en"
        )

        tts = gTTS(
            text=audio_text,
            lang=language_code,
            slow=False
        )

        tts.save(
            audio_file
        )

        result["audio_status"] = "SUCCESS"
        result["audio_file"] = audio_file

    except Exception:

        result["audio_status"] = "AUDIO_ERROR"

    return result


# ============================================================
# BUILD FINAL FRONTEND JSON
# ============================================================

def build_frontend_json(
    result,
    patient_name,
    language
):
    """
    Convert the internal AI/safety result into the exact
    JSON contract expected by the frontend.

    Internal implementation fields are not exposed here.
    """

    medicines_output = []

    for medicine in result.get(
        "medicines",
        []
    ):

        medicines_output.append({
            "name": medicine.get(
                "name",
                ""
            ),
            "dosage": medicine.get(
                "dosage",
                ""
            ),
            "frequency": medicine.get(
                "frequency_patient_friendly",
                medicine.get(
                    "frequency",
                    ""
                )
            ),
            "duration": medicine.get(
                "duration",
                ""
            ),
            "food_timing": medicine.get(
                "food_timing",
                ""
            ),
            "prescribed_times": medicine.get(
                "prescribed_times",
                []
            ),
            "instructions": medicine.get(
                "instructions",
                ""
            ),
            "confidence": medicine.get(
                "confidence",
                "low"
            ),
            "status": medicine.get(
                "status",
                "unclear"
            )
        })

    # --------------------------------------------------------
    # Diet
    # --------------------------------------------------------

    if language == "Tamil":

        diet = result.get(
            "patient_friendly_food_instructions",
            []
        )

    else:

        diet = result.get(
            "food_instructions",
            []
        )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    if language == "Tamil":

        warnings = result.get(
            "patient_friendly_warning_signs",
            []
        )

    else:

        warnings = result.get(
            "warning_signs",
            []
        )

    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    if language == "Tamil":

        follow_up = result.get(
            "patient_friendly_follow_up",
            {}
        )

    else:

        follow_up = result.get(
            "follow_up",
            {}
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    if result.get("refusal") is True:

        verification_status = "REFUSED"

    elif result.get("safety_status") == "CLEAR":

        verification_status = "CLEAR"

    else:

        verification_status = "REFUSED"

    return {
        "patient_name": patient_name,

        "medicines": medicines_output,

        "diet": diet,

        "warnings": warnings,

        "follow_up": {
            "timing": follow_up.get(
                "timing",
                ""
            ),
            "instructions": follow_up.get(
                "instructions",
                ""
            )
        },

        "verification": {
            "status": verification_status,
            "refusal": result.get(
                "refusal",
                False
            ),
            "refusal_message": result.get(
                "refusal_message",
                ""
            )
        }
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

# print("\nChoose patient language:")

# print("1. English")
# print("2. Tamil")

# language_choice = input(
#     "\nEnter 1 or 2: "
# ).strip()

# if language_choice == "2":
#     language = "Tamil"
# else:
#     language = "English"


# raw_text = input(
#     "\nPaste discharge summary text:\n"
# )


# # ============================================================
# # EXTRACT PATIENT NAME LOCALLY
# # ============================================================

# patient_name = extract_patient_name(
#     raw_text
# )


# # ============================================================
# # AI PROCESSING
# # ============================================================

# result = extract_discharge_info(
#     raw_text,
#     language
# )

# result = safety_check(
#     result,
#     raw_text,
#     language
# )

# result = create_daily_schedule(
#     result
# )

# result = add_patient_friendly_content(
#     result,
#     language
# )

# result = generate_audio(
#     result,
#     language
# )


# # ============================================================
# # BUILD FRONTEND JSON
# # ============================================================

# frontend_result = build_frontend_json(
#     result,
#     patient_name,
#     language
# )


# # ============================================================
# # DISPLAY RESULT
# # ============================================================

# print("\nSELECTED LANGUAGE:")
# print(language)

# print("\nFRONTEND JSON:")

# print(
#     json.dumps(
#         frontend_result,
#         indent=2,
#         ensure_ascii=False
#     )
# )


# # ============================================================
# # STATUS
# # ============================================================

# if frontend_result["verification"]["refusal"] is True:

#     print("\n⚠️ SAFE REFUSAL:")

#     print(
#         frontend_result["verification"].get(
#             "refusal_message",
#             "Please confirm the medication information with the hospital."
#         )
#     )

# elif result.get(
#     "audio_status"
# ) == "SUCCESS":

#     print("\n🔊 AUDIO CREATED:")

#     print(
#         result["audio_file"]
#     )

# else:

#     print("\n🔊 AUDIO STATUS:")

#     print(
#         result.get(
#             "audio_status",
#             "NOT_CREATED"
#         )
#     )

def generate_patient_plan(raw_text, language="English"):
    """
    Complete AI pipeline:
    OCR text → AI extraction → safety check → schedule
    → patient-friendly content → audio → frontend JSON
    """

    # Extract patient name locally
    patient_name = extract_patient_name(raw_text)

    # AI extraction
    result = extract_discharge_info(raw_text, language)

    # Safety validation
    result = safety_check(result, raw_text, language)

    # Create medicine schedule
    result = create_daily_schedule(result)

    # Add patient-friendly English/Tamil content
    result = add_patient_friendly_content(result, language)

    # Generate audio
    result = generate_audio(result, language)

    # Build final JSON for frontend
    final_output = build_frontend_json(
        result,
        patient_name,
        language
    )

    return final_output

