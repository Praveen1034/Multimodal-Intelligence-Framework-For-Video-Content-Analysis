from video_to_mp3.video_to_mp3 import  convert_mp4_to_mp3
from video_to_mp3.combine_code import  combine_video_audio
from Audio_Whisper.audio_to_text import AudioTranscriber, Config
from Video_Slide_Detector.vid2slides import extract_keyframes_from_video
import os

video_path = r"D:\Video_Analysis\Video_Slide_Detector\Example_Video\Distillation_Robustifies_Unlearning.mp4"
mp3 = convert_mp4_to_mp3(video_path)
if mp3 and os.path.exists(mp3):
    print(f"MP3 file created at: {mp3}")
else:
    print("Failed to create MP3 file.")
    exit(1)

# audio to text
audio_path = mp3
if audio_path:
    # Set up config for AudioTranscriber
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_json_dir = os.path.join("Output", video_file_name)
    os.makedirs(output_json_dir, exist_ok=True)
    config = Config(
        input_mp3_path=audio_path,
        output_json_dir=output_json_dir,  # output under Output/<video_file_name>
        model_size="tiny"  # or another model size if desired
    )
    audio_transcriber = AudioTranscriber(config=config)
    transcription_data = audio_transcriber.transcribe_audio()
    if transcription_data:
        print(f"Transcription completed for {transcription_data.file_name}. Duration: {transcription_data.duration:.2f} seconds.")
        # Save the transcription to JSON
        audio_transcriber.save_to_json(transcription_data)
        audio_json_path = os.path.abspath(os.path.join(output_json_dir, f"{transcription_data.file_name.split('.')[0]}_audio.json"))
        print(f"Transcription data saved to: {audio_json_path}")
    else:
        print("Transcription failed or no data returned.")
        
# Extract keyframes from video
keyframes_output_dir = os.path.join("Output", video_file_name)
os.makedirs(keyframes_output_dir, exist_ok=True)
# create the output directory for the thumbnails
thumb_dir = os.path.join(keyframes_output_dir, 'thumbnails')
video_json_path = os.path.join(keyframes_output_dir, f"{video_file_name}_video.json")
video_json_path = extract_keyframes_from_video(video_path, video_json_path, thumb_dir)
print(f"Keyframes extracted and saved to: {video_json_path}")

# combine json
process_json_path = os.path.join(keyframes_output_dir, 'process_json.json')
combine_video_audio(video_json_path, audio_json_path, process_json_path)
print(f"process JSON created at: {process_json_path}")


