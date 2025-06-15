# Author: Praveen Choudhary  
# Email: praveen.choudhary1034@gmail.com  
# Phone: +91 8619714798  
# Organization: DisruptiveNext  
# Supervisor: Mr. Prashant Mane  

# Description:  
# This script facilitates YouTube video downloads using the yt_dlp library,  
# ensuring high-quality audio and video retrieval in MP4 format. It validates user input,  
# sets up a structured logging mechanism, and ensures the availability of FFmpeg  
# for processing media files. The script incorporates automatic retries for  
# unstable network conditions and logs errors efficiently. It employs the Pydantic  
# data model for request validation and structured responses. Additionally, it  
# supports configurable download paths and integrates a command-line interface  
# for user interaction. The script is optimized for robustness, handling timeouts,  
# merging media streams, and ensuring a smooth download experience.  

# Copyright (c) 2025 Praveen Choudhary  

import os
import sys
import time
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, model_validator
import yt_dlp
import shutil
import re

# Set FFmpeg path
os.environ["PATH"] += os.pathsep + "C:\\ProgramData\\chocolatey\\bin"

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HARDCODED_DOWNLOAD_DIR = r"D:\Video_Analysis\Input_Data"
DEFAULT_OUTPUT_TEMPLATE = os.path.join(HARDCODED_DOWNLOAD_DIR, "%(title)s.%(ext)s")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120  # Extended timeout

class YouTubeDownloadRequest(BaseModel):
    """
    Data model representing a YouTube video download request.
    """
    video_url: str = Field(..., description="The URL of the YouTube video to download.")
    output_template: str = Field(DEFAULT_OUTPUT_TEMPLATE, description="File path template.")
    
    @model_validator(mode="before")
    def strip_fields(cls, values):
        """
        Strips whitespace from input fields and ensures video_url is not empty.
        """
        url = values.get("video_url", "").strip()
        template = values.get("output_template", "").strip() or DEFAULT_OUTPUT_TEMPLATE
        if not url:
            raise ValueError("video_url cannot be empty.")
        values["video_url"] = url
        values["output_template"] = template
        return values

class YouTubeDownloadResponse(BaseModel):
    """
    Data model representing the response of a YouTube video download attempt.
    """
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None


def check_ffmpeg_installed():
    """
    Checks if FFmpeg is installed and available in the system's PATH.
    """
    if shutil.which("ffmpeg") is None:
        print("FFmpeg is not installed. Please install FFmpeg for merging audio and video.")
        return False
    return True


def sanitize_folder_name(name: str) -> str:
    """
    Removes or replaces invalid characters for Windows folder names.
    """
    # Remove characters: < > : " / \ | ? *
    return re.sub(r'[<>:"/\\|?*]', '', name)


def download_youtube_video(video_url: str, output_template: str = DEFAULT_OUTPUT_TEMPLATE) -> YouTubeDownloadResponse:
    """
    Downloads a YouTube video using yt_dlp, ensuring the best video and audio quality.
    The video will be saved in D:\Video_Analysis\Video_Downloader\<Video Title>\<Video Title>.mp4
    
    Args:
        video_url (str): The URL of the YouTube video to download.
        output_template (str, optional): The output file template. Defaults to DEFAULT_OUTPUT_TEMPLATE.
    
    Returns:
        YouTubeDownloadResponse: A response indicating the success or failure of the download.
    """
    print(f"Starting download: {video_url}")

    # Step 1: Extract video title first
    try:
        with yt_dlp.YoutubeDL({}) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get('title', 'downloaded_video').strip()
            video_title = sanitize_folder_name(video_title)
    except Exception as e:
        print(f"Failed to fetch video info: {e}")
        return YouTubeDownloadResponse(success=False, error_message=f"Failed to fetch video info: {e}")

    # Step 2: Create folder with video title
    video_folder = os.path.join(HARDCODED_DOWNLOAD_DIR, video_title)
    os.makedirs(video_folder, exist_ok=True)

    # Step 3: Set output template to save video inside the folder with the title as filename
    output_template = os.path.join(video_folder, f"{video_title}.%(ext)s")

    # Validate Input
    try:
        request = YouTubeDownloadRequest(video_url=video_url, output_template=output_template)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return YouTubeDownloadResponse(success=False, error_message=str(e))

    # Ensure FFmpeg is installed
    if not check_ffmpeg_installed():
        return YouTubeDownloadResponse(success=False, error_message="FFmpeg is required but not installed.")

    ydl_opts = {
        'outtmpl': request.output_template,
        'format': 'bv*+ba/b',
        'merge_output_format': 'mp4',
        'ignoreerrors': True,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'
        }],
        'retries': 10,
        'socket_timeout': TIMEOUT_SECONDS,
        'noprogress': False,
    }

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(request.video_url, download=True)

                downloaded_files = [
                    file_info.get("filepath")
                    for file_info in info_dict.get("requested_downloads", [])
                    if file_info.get("filepath") and os.path.isfile(file_info.get("filepath"))
                ]

                if downloaded_files:
                    final_file = downloaded_files[-1]
                    print(f"Download successful: {final_file}")
                    return YouTubeDownloadResponse(success=True, file_path=final_file)
                else:
                    print("Download incomplete. Retrying...")
        
        except Exception as e:
            print(f"Download error: {e}")
            if "timed out" in str(e).lower():
                retry_count += 1
                print(f"Timeout occurred. Retrying {retry_count}/{MAX_RETRIES}...")
                time.sleep(5)
            else:
                return YouTubeDownloadResponse(success=False, error_message=str(e))
    
    return YouTubeDownloadResponse(success=False, error_message="Failed after multiple attempts.")


if __name__ == "__main__":
    """
    Main script execution to download a YouTube video based on user input.
    """
    video_url = "https://youtu.be/eGveWERbeY8?si=VInA7OkgLe3S-Qwu"
    output_template = r"D:\Video_Analysis\Input_Data"

    response = download_youtube_video(video_url, output_template)
    if response.success:
        print(f"✅ Downloaded: {response.file_path}")
    else:
        print(f"❌ Failed: {response.error_message}")