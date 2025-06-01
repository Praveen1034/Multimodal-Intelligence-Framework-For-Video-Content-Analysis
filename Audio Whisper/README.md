# Audio Transcription with Whisper AI

## Overview
This project utilizes OpenAI's Whisper model to transcribe MP3 audio files into structured JSON format. The transcription includes timestamped segments with start and end times.

## Features
- Transcribes MP3 audio files into text.
- Supports multiple Whisper model sizes (`tiny`, `base`, `small`, `medium`, `large`).
- Provides timestamped segments for accurate transcription.
- Detects the language of the audio file.
- Saves transcriptions as JSON files.

## Prerequisites
Ensure you have the following dependencies installed:

### Python Packages
Install required dependencies using:
```bash
pip install -r requirements.txt
```
#### Required Packages:
- `torch`
- `whisper`
- `pydantic`
- `mutagen`
- `json`
- `os`

### Additional Requirements
- A compatible GPU with CUDA for improved performance (optional).

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio_transcriber.git
cd audio_transcriber
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Verify that Whisper is installed correctly by running:
```python
import whisper
model = whisper.load_model("tiny")
print("Model loaded successfully!")
```

## Usage
1. Update the `input_mp3_path` variable in the `Config` class with the path to your MP3 file.
2. Set the `output_json_dir` to the directory where transcriptions should be saved.
3. Run the script:
```bash
python transcriber.py
```
4. The script will:
   - Load the specified Whisper model.
   - Transcribe the MP3 file and extract timestamps.
   - Save the transcription in a JSON file in the specified directory.

## JSON Output Format
The transcription output will be saved as a JSON file in the following format:
```json
{
    "file_name": "example.mp3",
    "duration": 120.5,
    "language": "en",
    "transcription": [
        {
            "start_time": "00:00:05",
            "end_time": "00:00:10",
            "text": "Hello, welcome to our podcast."
        },
        {
            "start_time": "00:00:11",
            "end_time": "00:00:15",
            "text": "Today we are discussing AI advancements."
        }
    ]
}
```
## Customization
- Modify `model_size` in the `Config` class to choose a different Whisper model.
- Adjust the `output_json_dir` to store transcriptions in a different location.

## Troubleshooting
- **Issue: Whisper model takes too long to process.**
  - Try using a smaller model (`tiny` or `base`).
  - Use a GPU if available.
- **Issue: Transcriptions are inaccurate.**
  - Use a larger model (`medium` or `large`) for better accuracy.
- **Issue: `FileNotFoundError`**
  - Ensure the MP3 file path is correct and accessible.


