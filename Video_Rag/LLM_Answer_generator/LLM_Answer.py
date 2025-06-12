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

json_path = r"D:\Video_Analysis\combine_json.json"  # 👈 Replace with the actual JSON file path
user_query = "tell me about Raymand James"  # 👈 Replace with the actual user query

with open(json_path, 'r', encoding='utf-8') as f:
        slides_json_str = f.read().strip()


        
prompt = f"""You are an expert question answering system.
Your task is to extract information from a JSON file containing slides from a video. Each slide has a start time, end time, and content. The content may include text and audio descriptions.
In answer you have to give details about the relevant slide that answers the user's query from audio content and video content of that slides. **Dont Mention start and end time of the slide in your answer**.
thier may be possibility that the answer is not present in the slides, in that case you have to return empty string in answer field and start time and end time of the slide as empty string Na.
If the answer is present in multiple slides, you have to return the all slide answer but time should be from first slide to second slide add both time.
Json_contest: {slides_json_str} User_query: {user_query}. 
Please provide a JSON response with the following structure:\n"
    "{{'Answer': 'The answer to the users query',\n"
    '"Start_time": "The start time of the relevant slide",\n'
    '"End_time": "The end time of the relevant slide check from the next slide start_time - 1",\n'
    '}}'"""
        
response = model.generate_content(prompt)
response_text = response.text.replace("```json", "").replace("```", "").strip()
response_json = json.loads(response_text)
if 'Answer' in response_json:
    answer = response_json['Answer']
    start_time = response_json.get('Start_time', 'Na')
    end_time = response_json.get('End_time', 'Na')
    