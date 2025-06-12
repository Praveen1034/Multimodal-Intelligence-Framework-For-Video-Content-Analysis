import subprocess
import os

def convert_mp4_to_mp3(video_path):
    if not os.path.exists(video_path):
        print("❌ File not found.")
        return None

    output_path = os.path.splitext(video_path)[0] + ".mp3"

    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_path
        ], check=True)
        print(f"✅ Converted successfully: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error: {e}")
        return None

if __name__ == "__main__":
    video_path = input("Enter the path to your .mp4 video file: ").strip()
    mp3_path = convert_mp4_to_mp3(video_path)
    if mp3_path:
        print(f"MP3 file saved at: {mp3_path}")
