# vid2slides

## Overview
`vid2slides` is a Python-based tool designed to extract key slides from videos, such as those recorded during presentations, Zoom meetings, or Google Meet sessions. The tool processes the video to identify and extract keyframes, deduplicate slides, and generate a JSON file containing metadata about the slides, including their timestamps and OCR-extracted text.

## Features
- Extracts thumbnails and high-resolution frames from videos.
- Identifies key slides using heuristic and probabilistic methods.
- Performs OCR (Optical Character Recognition) on slides to extract text.
- Deduplicates slides based on visual similarity and text content.
- Outputs a JSON file with slide metadata, including timestamps and OCR text.

## Project Structure
```
vid2slides-main/
├── requirements.txt
├── vid2slides.py
├── Example_Video/
│   ├── Raymond James.mp4
│   ├── Summer Project Video.mp4
├── models/
│   └── haarcascade_frontalface_default.xml
├── Output/
│   ├── Raymond James/
│   │   ├── Raymond James.json
│   │   └── hi/
│   │       ├── thumb-0001.png
│   │       ├── thumb-0002.png
│   │       └── ...
│   ├── Summer Project Video/
│   │   ├── Summer Project Video.json
│   │   └── thumbnails/
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/vid2slides.git
   cd vid2slides-main
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - Download and install Tesseract OCR from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` path in `vid2slides.py` to point to your Tesseract installation directory.

## Usage

1. Place your video files in the `Example_Video/` directory.
2. Update the `target` variable in `vid2slides.py` to point to your video file. For example:
   ```python
   target = r"C:\Users\praveen choudhary\Downloads\vid2slides-main\vid2slides-main\Example_Video\Summer Project Video.mp4"
   ```
3. Run the script:
   ```bash
   python vid2slides.py
   ```
4. The output will be saved in the `Output/` directory, organized by video name. For example:
   ```
   Output/
   ├── Summer Project Video/
   │   ├── Summer Project Video.json
   │   └── thumbnails/
   ```

## JSON Output Format
The JSON file contains metadata about the extracted slides. Example structure:
```json
{
  "pip_location": [],
  "sequence": [
    {
      "type": "slide",
      "start_time": "00:00:05",
      "start_index": 0,
      "offset": 5.0,
      "source": "path/to/slide.png",
      "text_ocr": "Extracted text from slide"
    }
  ],
  "crop": [x, y, w, h]
}
```

## Key Functions

### `extract_keyframes_from_video`
- Extracts keyframes and generates a JSON file with slide metadata.

### `extract_thumbnails`
- Extracts low-resolution thumbnails from the video.

### `extract_frames`
- Extracts high-resolution frames for selected keyframes.

### `deduplicate_slides`
- Deduplicates slides based on visual similarity and OCR text.

### `extract_crop`
- Determines the crop region for slides based on content.

## Dependencies
- Python 3.7+
- OpenCV
- Decord
- FFmpeg
- Tesseract OCR
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- scikit-image

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [FFmpeg](https://ffmpeg.org/)