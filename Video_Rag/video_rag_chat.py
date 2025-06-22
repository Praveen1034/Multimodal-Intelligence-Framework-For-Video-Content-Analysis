from mkv_Playback.mkv_playback import play_mkv_file
from Rag_time_extract.rag_time_extract import process_query
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path="D:/Video_Analysis/.env")

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

file_path = r"D:\Video_Analysis\Input_Data\Raymond James\Raymond James.mp4"
json_path = r"D:\Video_Analysis\Output\Raymond James\process_json.json"  # Replace with your actual JSON file path

user_question = "tell me about Raymand James"  # 👈 Replace with your actual user query
# Replace with your actual user query
start,end,text = process_query(json_path, user_question)
prompt = f"""You are an expert question answering system.
User query: {user_question}
extract text: {text}
You have to answer the user query from the text content of the video slides.
If the answer is not present in the text, return an empty string.'}}'"""
        
response = model.generate_content(prompt)
print(f"Answer: {response.text.strip()}")

play_mkv_file(file_path, start, end)
