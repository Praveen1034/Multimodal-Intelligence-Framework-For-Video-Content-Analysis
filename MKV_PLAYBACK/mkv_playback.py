import os
import time
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, model_validator
from playwright.sync_api import sync_playwright

class MKVPlaybackRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the MKV file.")
    start_time: int = Field(..., description="Start time in seconds.")
    end_time: int = Field(..., description="End time in seconds.")

    @model_validator(mode="before")
    def validate_fields(cls, values):
        file_path = values.get("file_path", "").strip()
        start_time = values.get("start_time", 0)
        end_time = values.get("end_time", 0)
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")
        if start_time < 0 or end_time <= start_time:
            raise ValueError("Invalid start or end time.")
        values["file_path"] = file_path
        return values

class MKVPlaybackResponse(BaseModel):
    success: bool = Field(..., description="Indicates if playback started successfully.")
    error_message: Optional[str] = Field(None, description="Error message if playback failed.")

def play_mkv_file(file_path: str, start_time: int, end_time: int) -> MKVPlaybackResponse:
    try:
        request_obj = MKVPlaybackRequest(file_path=file_path, start_time=start_time, end_time=end_time)
    except ValidationError as e:
        return MKVPlaybackResponse(success=False, error_message=str(e))

    temp_html = None
    try:
        with sync_playwright() as p:
            # Generate temporary HTML with multiple source types
            file_uri = Path(request_obj.file_path).as_uri()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                html_content = f"""
                <html>
                    <head>
                        <style>body {{ margin: 0; }}</style>
                    </head>
                    <body>
                        <video id="player" controls width="100%" height="100%">
                            <source src="{file_uri}" type="video/x-matroska">
                            <source src="{file_uri}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </body>
                </html>
                """
                f.write(html_content)
                temp_html = f.name

            # Launch Chromium browser
            browser = p.chromium.launch(
                headless=False,  # Ensure headless mode is disabled
                channel="chrome",
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-features=OutOfBlinkCors",
                    "--enable-features=PlatformHEVCDecoderSupport",
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream"
                ]
            )

            context = browser.new_context(viewport={"width": 1280, "height": 720})
            page = context.new_page()

            # Enable console logging for debugging
            def console_handler(msg):
                print(f"Browser Console: {msg.text}")

            page.on("console", console_handler)

            page.goto(f"file://{temp_html}", wait_until="networkidle")

            # Ensure the video element is present
            page.wait_for_selector("video#player")
            
            # Check for video load errors
            video_error = page.evaluate("""() => {
                const video = document.querySelector('video#player');
                return video.error ? video.error.message : null;
            }""")
            if video_error:
                return MKVPlaybackResponse(success=False, error_message=f"Video Error: {video_error}")

            # Get video duration and validate time ranges
            duration = page.evaluate("""() => {
                const video = document.querySelector('video#player');
                return isNaN(video.duration) ? null : video.duration;
            }""")
            if duration is None or duration <= 0:
                return MKVPlaybackResponse(success=False, error_message="Invalid video duration detected")
            if request_obj.start_time > duration:
                return MKVPlaybackResponse(success=False, error_message="Start time exceeds video duration")
            if request_obj.end_time > duration:
                return MKVPlaybackResponse(success=False, error_message="End time exceeds video duration")

            # Seek to the start time and wait for the seek to complete
            page.evaluate(f"""() => {{
                const video = document.querySelector('video#player');
                video.currentTime = {request_obj.start_time};
                return new Promise((resolve) => {{
                    video.addEventListener('seeked', resolve, {{ once: true }});
                }});
            }}""")
            
            # Start video playback and ensure it begins
            page.evaluate("""
                () => {
                    const video = document.querySelector('video#player');
                    video.play();
                    return new Promise((resolve) => {
                        video.addEventListener('play', resolve, { once: true });
                    });
                }
            """)

            # Wait until the video’s currentTime reaches the requested end_time.
            expected_playback_duration = request_obj.end_time - request_obj.start_time
            timeout_ms = expected_playback_duration * 1000 + 5000  # buffer of 5 seconds
            try:
                page.wait_for_function(
                    f"() => document.querySelector('video#player').currentTime >= {request_obj.end_time}",
                    timeout=timeout_ms
                )
            except Exception as e:
                return MKVPlaybackResponse(success=False, error_message=f"Timeout waiting for video to reach the end time: {e}")

            # Once reached, pause the video
            page.evaluate("""() => {
                document.querySelector('video#player').pause();
            }""")

            browser.close()
            return MKVPlaybackResponse(success=True)

    except Exception as e:
        return MKVPlaybackResponse(success=False, error_message=str(e))
    finally:
        if temp_html and os.path.exists(temp_html):
            try:
                os.unlink(temp_html)
            except Exception:
                pass

if __name__ == "__main__":
    try:
        file_path = input("Enter the absolute path to the MKV file: ").strip()
        start_time = int(input("Enter start time in seconds: ").strip())
        end_time = int(input("Enter end time in seconds: ").strip())
        response = play_mkv_file(file_path, start_time, end_time)
        if response.success:
            print("\n✅ Playback started successfully!")
        else:
            print(f"\n❌ Playback failed. Reason: {response.error_message}")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
