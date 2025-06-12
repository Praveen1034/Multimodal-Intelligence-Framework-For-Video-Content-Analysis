# Audio Transcription with Whisper AI
import os
import json
import whisper
import torch
from pydantic import BaseModel, Field, DirectoryPath, FilePath, ValidationError
from typing import Optional, List
from mutagen.mp3 import MP3


class Config(BaseModel):
    """Pydantic model for managing configuration paths."""
    input_mp3_path: FilePath = Field(..., description="Path to the MP3 file to be transcribed")
    output_json_dir: DirectoryPath = Field(..., description="Directory where JSON output will be saved")
    model_size: str = Field(default="tiny", description="Size of the Whisper model (tiny, base, small, medium, large)")

    class Config:
        arbitrary_types_allowed = True  # Allows validation of Path types


class TranscriptionSegment(BaseModel):
    """Pydantic model to store timestamped transcription segments."""
    start_time: str = Field(..., description="Start time of the speech segment")
    end_time: str = Field(..., description="End time of the speech segment")
    text: str = Field(..., description="Transcribed text segment")


class TranscriptionData(BaseModel):
    """Pydantic model to validate transcribed audio data."""
    file_name: str = Field(..., description="Name of the transcribed MP3 file")
    duration: Optional[float] = Field(None, description="Duration of the audio file in seconds")
    language: Optional[str] = Field(None, description="Detected language of the audio")
    transcription: List[TranscriptionSegment] = Field(default_factory=list, description="List of transcription segments")


class AudioTranscriber:
    """Class to handle MP3 transcription using Whisper AI."""

    def __init__(self, config: Config):
        """
        Initializes the Whisper AI model with GPU support if available.

        :param config: Configuration object containing input/output paths and model size.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = whisper.load_model(config.model_size).to(self.device)

    def get_audio_duration(self, file_path: str) -> float:
        """
        Get the duration of an MP3 file.

        :param file_path: Path to the MP3 file
        :return: Duration in seconds
        """
        audio = MP3(file_path)
        return audio.info.length

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def transcribe_audio(self) -> TranscriptionData:
        """
        Transcribes an MP3 file to text with timestamped segments.

        :return: TranscriptionData object
        :raises FileNotFoundError: If the file does not exist
        :raises ValueError: If the transcription is empty or invalid
        """
        file_path = self.config.input_mp3_path
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Error: File '{file_path}' not found.")

        duration = self.get_audio_duration(str(file_path))
        print(f"📢 Processing file: {file_path} (Duration: {duration:.2f} seconds)...")

        try:
            # Enable word timestamps
            result = self.model.transcribe(str(file_path), word_timestamps=True)
            segments = result.get("segments", [])
            language = result.get("language")

            if not segments:
                raise ValueError(f"⚠ No transcribed text found in {file_path}")

            formatted_segments = []
            for segment in segments:
                start_time = self.format_timestamp(segment["start"])
                end_time = self.format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                formatted_segments.append(TranscriptionSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))

            return TranscriptionData(
                file_name=file_path.name,
                duration=duration,
                language=language,
                transcription=formatted_segments
            )

        except Exception as e:
            raise RuntimeError(f"❌ Error during transcription: {str(e)}")

    def save_to_json(self, transcription: TranscriptionData) -> None:
        """
        Saves the transcribed text to a JSON file in the specified directory.

        :param transcription: TranscriptionData object
        """
        output_path = self.config.output_json_dir / f"{self.config.input_mp3_path.stem}_audio.json"
        try:
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(transcription.model_dump(), json_file, indent=4, ensure_ascii=False)
            print(f"✅ Transcription saved to: {output_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Error saving JSON file: {str(e)}")


def main():
    """Main function to automatically process the MP3 transcription."""

    # Configuration Setup
    try:
        config = Config(
            input_mp3_path=r"D:\Video_Analysis\Video-Slide-Detector\Example_Video\Raymond James.mp3",
            output_json_dir=r"D:\Video_Analysis\Audio Whisper\audio_to_text_jsons",
            model_size="tiny"  # Default to 'tiny' for faster processing
        )
        
    except ValidationError as e:
        print(f"❌ Config Validation Error: {e}")
        return

    # Ensure output directory exists
    os.makedirs(config.output_json_dir, exist_ok=True)

    try:
        transcriber = AudioTranscriber(config=config)
        transcription_data = transcriber.transcribe_audio()
        transcriber.save_to_json(transcription_data)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
