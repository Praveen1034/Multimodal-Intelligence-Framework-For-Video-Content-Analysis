import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load Gemini API key from .env
load_dotenv("D:/Video_Analysis/.env")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment.")
genai.configure(api_key=api_key)

# Define all models
MODELS = {
    "Gemini-2.0-Flash": genai.GenerativeModel("gemini-2.0-flash"),
    "Gemini-1.5-Flash": genai.GenerativeModel("gemini-1.5-flash"),
    "Gemini-1.5-Pro": genai.GenerativeModel("gemini-1.5-pro"),
}

VALID_OPTIONS = {"(a)", "(b)", "(c)", "(d)"}

def ask_llm(model, question_block):
    qn = question_block["Question"]
    options = [f"(a) {question_block.get('(a)', '')}",
               f"(b) {question_block.get('(b)', '')}",
               f"(c) {question_block.get('(c)', '')}",
               f"(d) {question_block.get('(d)', '')}"]

    prompt = f"""
You are an expert question-answering assistant.

Read the following multiple choice question carefully. If you're confident, return the correct option in this format: (a), (b), (c), or (d).  
If you're unsure or the information is not clearly provided, reply exactly with: Don't Know

Question:
{qn}

Options:
{chr(10).join(options)}

Answer only with one of: (a), (b), (c), (d), or Don't Know. Do not explain.
"""
    try:
        response = model.generate_content(prompt)
        answer_raw = response.text.strip()
        normalized = answer_raw.lower()
        if normalized in VALID_OPTIONS:
            return answer_raw, answer_raw
        elif "don't know" in normalized:
            return "Don't Know", answer_raw
    except Exception as e:
        return "Don't Know", f"Error: {e}"

    return "Don't Know", answer_raw

def process_json(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    print("🔍 Starting LLM answer generation with 3 Gemini models...\n")

    for item in data:
        qno = item.get("Question_No.", "Unknown")
        print(f"➤ Processing Question {qno}: {item['Question']}")

        answers = {}
        for model_name, model in MODELS.items():
            llm_answer, _ = ask_llm(model, item)
            print(f"   ✅ {model_name} Answer: {llm_answer}")
            answers[model_name + "_Answer"] = llm_answer

        output.append({
            "Question_No.": qno,
            "Question": item["Question"],
            **answers
        })
        print()

    output_path = Path(input_path).parent / "Answers_By_LLM.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"✅ All questions processed.")
    print(f"📁 Output saved at: {output_path}")

if __name__ == "__main__":
    file_path = input("📥 Enter the path to the input JSON file: ").strip().strip('"')
    if not os.path.isfile(file_path):
        print("❌ File not found. Please check the path.")
    else:
        process_json(file_path)
