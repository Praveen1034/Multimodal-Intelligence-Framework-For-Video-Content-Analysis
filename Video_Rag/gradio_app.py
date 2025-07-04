import gradio as gr
import os
import tempfile
import subprocess
from Rag_time_extract.rag_time_extract import process_query
import google.generativeai as genai
from dotenv import load_dotenv

# Path to the video file (update as needed or make dynamic)
VIDEO_PATH = r"D:\Video_Analysis\Input_Data\Raymond James\Raymond James.mp4"
# Path to the process_json.json file
JSON_PATH = r"D:/Video_Analysis/Output/Raymond James/process_json.json"

# Load Gemini API key
load_dotenv(dotenv_path="D:/Video_Analysis/.env")
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

def extract_video_segment(input_path, start, end):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"segment_{start}_{end}.mp4")
    duration = end - start
    if not os.path.exists(input_path):
        return None, "Video file not found."
    if duration <= 0:
        return None, "Invalid segment duration."
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac", output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, f"FFmpeg error: {result.stderr.decode('utf-8')[:500]}"
        # Double-check file is readable and valid
        try:
            with open(output_path, "rb") as f:
                f.read(10)
        except Exception as e:
            return None, f"Segment file not readable: {e}"
        return os.path.abspath(output_path), None
    except FileNotFoundError:
        return None, "FFmpeg is not installed or not in PATH."

def qa_and_video(user_question):
    start, end, text = process_query(JSON_PATH, user_question)
    prompt = f"""You are an expert question answering system.\nUser query: {user_question}\nextract text: {text}\nYou have to answer the user query from the text content of the video slides.\nIf the answer is not present in the text, return an empty string."""
    response = model.generate_content(prompt)
    gemini_answer = response.text.strip()
    segment_path, video_error = extract_video_segment(VIDEO_PATH, start, end)
    if not segment_path:
        return None, f"<div style='padding:16px; background:#000; color:#fff; border-radius:8px;'><b>Gemini Answer:</b><br>{gemini_answer}</div>", f"Video segment could not be created. {video_error if video_error else ''}"
    return segment_path, f"<div style='padding:16px; background:#000; color:#fff; border-radius:8px;'><b>Gemini Answer:</b><br>{gemini_answer}</div>", ""

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
with gr.Blocks(theme=theme, title="Video QA & Playback App") as demo:
    gr.Markdown("""
    # Video QA & Playback App
    <p style='font-size:18px;'>Ask a question about the video below. The answer will be generated by Gemini and the relevant video segment will play.</p>
    """)
    with gr.Row():
        with gr.Column(scale=2):
            video_output = gr.Video(label="Video Segment", interactive=False)
        with gr.Column(scale=3):
            answer_output = gr.HTML(label="Gemini Answer")
            error_output = gr.Textbox(label="Error", visible=False)
    question_input = gr.Textbox(label="Ask a question about the video", placeholder="Type your question here...", lines=1)
    submit_btn = gr.Button("Get Answer & Play Video", variant="primary")
    submit_btn.click(qa_and_video, inputs=question_input, outputs=[video_output, answer_output, error_output])

demo.launch(share=True)


# give finanical report of raymond james?
# 
