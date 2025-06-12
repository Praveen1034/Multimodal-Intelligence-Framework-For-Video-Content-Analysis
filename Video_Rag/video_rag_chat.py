from mkv_Playback.mkv_playback import play_mkv_file
from Rag_time_extract.rag_time_extract import process_query

file_path = r"D:\Video_Analysis\Video_Slide_Detector\Example_Video\Raymond James.mp4"
json_path = r"D:\Video_Analysis\combine_json.json"  # Replace with your actual JSON file path


user_question = "forward-looking statements and risks"
start,end = process_query(json_path, user_question)

play_mkv_file(file_path, start, end)
