import json
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Utilities for time conversion
def timestamp_to_seconds(ts):
    # Support timestamps like '0.0:0.0:13' or '00:00:13'
    parts = ts.split(":")
    h = int(float(parts[0]))
    m = int(float(parts[1]))
    s = int(float(parts[2]))
    return h * 3600 + m * 60 + s

def seconds_to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}.0:0.0:{s:02d}"

# Load slides from JSON file
def load_slides(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["combine_json"]["slides"]

def time_str_to_seconds(time_str: str) -> int:
    """Convert a time string like '0.0:1.0:23' to total seconds."""
    try:
        parts = time_str.strip().split(':')
        if len(parts) != 3:
            raise ValueError("Invalid time format, expected 'H:M:S'")
        
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        total_seconds = int(hours * 3600 + minutes * 60 + seconds)
        return total_seconds
    except Exception as e:
        raise ValueError(f"Error parsing time string: {e}")



# Semantic search function
def get_slide_by_query(slides, user_query):
    slide_texts = [
        slide.get("Video_Content", "") + " " + " ".join(slide.get("Audio_Content", []))
        for slide in slides
    ]

    slide_embeddings = model.encode(slide_texts, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, slide_embeddings)[0]
    best_idx = int(torch.argmax(similarities))

    start_time = slides[best_idx]["start_time"]
    if best_idx + 1 < len(slides):
        next_start_sec = timestamp_to_seconds(slides[best_idx + 1]["start_time"])
        end_time = seconds_to_timestamp(next_start_sec - 1)
    else:
        end_time = 60
        
    start_sec = time_str_to_seconds(start_time)
    if end_time is not None:
        end_sec = time_str_to_seconds(end_time)
    else:
        end_sec = 60

    return start_sec, end_sec
# Main function
def process_query(json_file_path, user_query):
    slides = load_slides(json_file_path)
    result = get_slide_by_query(slides, user_query)
    return result

# Example usage
if __name__ == "__main__":
    json_path = r"D:\Video_Analysis\combine_json.json"  # 👈 Replace with the actual JSON file path
    query = "tell me about Raymand James"
    start,end = process_query(json_path, query)
    print("Start Time:", start)
    print("End Time:", end)
