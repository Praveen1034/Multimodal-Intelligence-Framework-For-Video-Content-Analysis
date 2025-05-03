# MKV Video Playback Using Playwright

This project provides a Python script to play an MKV video file in a browser using Playwright. It allows users to specify a start and end time for playback.

## 🚀 Features
- Plays MKV videos using a Chromium browser.
- Allows users to specify start and end times.
- Uses Playwright for automation.
- Validates file existence and time inputs.
- Logs browser console messages for debugging.

## 📌 Requirements
- Python 3.8+
- Google Chrome installed
- Playwright installed with Chromium support

## 📥 Installation

1. **Create a virtual environment (Optional but recommended)**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

2. **Install Playwright browsers**:
   ```sh
   playwright install
   ```

## 🛠 Usage

1. **Run the script**:
   ```sh
   python mkv_playback.py
   ```

2. **Enter the required details when prompted**:
   ```sh
   Enter the absolute path to the MKV file: /path/to/video.mkv
   Enter start time in seconds: 0
   Enter end time in seconds: 10
   ```

3. **Playback starts in a Chromium browser.**

![image](https://github.com/user-attachments/assets/2d07d8f8-744e-479b-b67c-63f4b41cb5f4)
