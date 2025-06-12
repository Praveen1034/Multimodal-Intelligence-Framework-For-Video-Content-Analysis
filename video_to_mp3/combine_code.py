import json
import os

def parse_video_time(time_str):
    """
    Parse a video slide time string of the form "H.M:MM:SS" (e.g., "0.0:0.0:23")
    into total seconds.
    """
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Unexpected video time format: {time_str}")
    # The first two parts may be floats or ints, e.g. "0.0" or "0"
    hours = int(float(parts[0]))
    minutes = int(float(parts[1]))
    seconds = int(float(parts[2]))
    return hours * 3600 + minutes * 60 + seconds

def parse_audio_time(time_str):
    """
    Parse an audio transcription time string of the form "HH:MM:SS"
    into total seconds.
    """
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Unexpected audio time format: {time_str}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def combine_video_audio(video_json_path, audio_json_path, output_path):
    # Load video.json
    with open(video_json_path, 'r', encoding='utf-8') as f:
        video_data = json.load(f)

    # Load audio.json
    with open(audio_json_path, 'r', encoding='utf-8') as f:
        audio_data = json.load(f)

    # Extract list of slides and transcription entries
    slides = video_data.get('sequence', [])
    transcripts = audio_data.get('transcription', [])

    # Precompute slide start times (in seconds)
    slide_start_times = [parse_video_time(slide['start_time']) for slide in slides]

    # Precompute audio start times (in seconds) for each transcription entry
    audio_start_times = [parse_audio_time(entry['start_time']) for entry in transcripts]

    combined_slides = []
    num_slides = len(slides)

    for i, slide in enumerate(slides):
        this_slide_start = slide_start_times[i]
        # Determine end boundary for this slide:
        # If there is a next slide, its start time defines the next boundary.
        # Otherwise, use a very large number (include all remaining audio).
        if i + 1 < num_slides:
            next_slide_start = slide_start_times[i + 1]
        else:
            next_slide_start = float('inf')

        # Collect all audio entries whose start_time < next_slide_start
        # (i.e., belongs to this slide)
        this_slide_transcripts = []
        for j, entry in enumerate(transcripts):
            audio_start = audio_start_times[j]
            if this_slide_start <= audio_start < next_slide_start:
                this_slide_transcripts.append(entry)

        # Build the new format for the combined slide
        audio_content = [entry["text"] for entry in this_slide_transcripts]
        combined_slide = {
            "type": slide.get("type"),
            "start_time": slide.get("start_time"),
            "start_index": slide.get("start_index"),
            "offset": slide.get("offset"),
            "source": slide.get("source"),
            "Video_Content": slide.get("text_ocr"),
            "Audio_Content": audio_content
        }
        combined_slides.append(combined_slide)

    # Construct the final combined JSON structure
    combined_output = {
        "combine_json": {
            "slides": combined_slides
        }
    }

    # Write the combined JSON to output_path
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(combined_output, out_f, indent=2)

if __name__ == "__main__":
    # Paths to your input JSON files
    video_json_path = r"D:\Video_Analysis\Output\Raymond James\Raymond James_video.json"
    audio_json_path = r"D:\Video_Analysis\Output\Raymond James\Raymond James_audio.json"
    output_path = "combine_json.json"

    # Ensure the input files exist
    if not os.path.isfile(video_json_path):
        raise FileNotFoundError(f"Could not find {video_json_path}")
    if not os.path.isfile(audio_json_path):
        raise FileNotFoundError(f"Could not find {audio_json_path}")

    combine_video_audio(video_json_path, audio_json_path, output_path)
    print(f"Combined JSON written to {output_path}")
