import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load API Key
load_dotenv("D:/Video_Analysis/.env")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")


def extract_content(json_path):
    """Extract Video_Content and Audio_Content from the JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    slides = data.get("combine_json", {}).get("slides", [])
    combined_text = ""
    for slide in slides:
        video = slide.get("Video_Content", "")
        audio = " ".join(slide.get("Audio_Content", []))
        combined_text += f"{video} {audio} "

    return combined_text.strip()


def generate_mcqs(text):
    """Generate 50 MCQs from the given content using Gemini."""
    prompt = (
        "Generate exactly 50 multiple choice questions in JSON format ONLY based strictly on the content below. "
        "Format should be a JSON array of objects, where each object has this structure:\n"
        '{\n  "Question_No.": 1,\n  "Question": "Text?",\n  "(a)": "Option A",\n  "(b)": "Option B",\n  "(c)": "Option C",\n  "(d)": "Option D",\n  "Answer": "(a)"\n}\n\n'
        "Content:\n" + text
    )

    response = model.generate_content(prompt)
    return response.text


def clean_and_parse_mcqs(raw_text):
    """Clean and convert Gemini output into valid JSON array."""
    try:
        # Remove any accidental markdown or code fences
        clean_text = re.sub(r"```json|```", "", raw_text).strip()

        # Try loading as JSON
        data = json.loads(clean_text)

        # Validate it's a list of MCQs
        if isinstance(data, list) and all(isinstance(q, dict) for q in data):
            return data
        else:
            raise ValueError("Output is not a list of question dictionaries.")
    except Exception as e:
        print(f"❌ Error cleaning/parsing MCQs: {e}")
        return None


def main():
    json_path = input("Enter full path to your input JSON file: ").strip()
    if not os.path.isfile(json_path):
        print("❌ File not found.")
        return

    print("🔍 Extracting content...")
    content = extract_content(json_path)

    print("🤖 Generating MCQs with Gemini...")
    raw_mcqs = generate_mcqs(content)

    print("🧹 Cleaning and validating MCQs...")
    cleaned_mcqs = clean_and_parse_mcqs(raw_mcqs)
    if not cleaned_mcqs:
        print("❌ Failed to generate valid MCQs.")
        return

    output_path = os.path.join(os.path.dirname(json_path), "question.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_mcqs, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned MCQs saved to: {output_path}")


if __name__ == "__main__":
    main()
