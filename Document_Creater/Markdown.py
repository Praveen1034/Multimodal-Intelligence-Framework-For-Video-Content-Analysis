import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv(dotenv_path="D:/Video_Analysis/.env")
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Configure Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")  # Or use "gemini-1.5-pro" for better quality

# Paths
json_path = r"D:\Video_Analysis\Output\Raymond James\process_json.json"
markdown_output_path = json_path.replace(".json", ".md")

# Load JSON content
with open(json_path, 'r', encoding='utf-8') as f:
    slides_data = json.load(f)

# Prepare content for the LLM prompt
slides = slides_data.get("combine_json", {}).get("slides", [])
compiled_content = ""
for idx, slide in enumerate(slides, 1):
    compiled_content += f"### Slide {idx}: {slide.get('Video_Content', '').strip()}\n"
    audio_lines = slide.get("Audio_Content", [])
    for line in audio_lines:
        compiled_content += f"- {line.strip()}\n"
    compiled_content += "\n"

# Define prompt
prompt = f"""You are an expert in creating clear and professional business documentation.
Below is slide and audio content from a Raymond James investor presentation.
Your task is to create a structured and clean Markdown document detail summarizing this presentation, including slide titles and bullet points for key points.

{compiled_content}

Generate a Markdown (.md) format summary now.
"""

# Generate markdown using Gemini
response = model.generate_content(prompt)
markdown_text = response.text

# Save markdown file
with open(markdown_output_path, 'w', encoding='utf-8') as md_file:
    md_file.write(markdown_text)

print(f"✅ Markdown file created at: {markdown_output_path}")
