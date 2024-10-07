import os
import tempfile
import requests
import subprocess
from utils.logger import logger
from .audio_moderation import moderate_text_content
from .image_moderation import extract_text_from_image
from .image_moderation import moderate_image_content
from PIL import Image
import openai

def moderate_video(item, policies=None, sensitivity='medium'):
    """
    Moderates the video by extracting audio, frames, and text, and analyzing them.

    Args:
        item (dict): The video item to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        tuple: A tuple containing the status ('Approved' or 'Rejected'), reason, and tags.
    """
    video_url = item.get('video_url', {}).get('url')
    if not video_url:
        return "Rejected", "No video URL provided", []
    logger.info(f"Processing video URL: {video_url}")
    response = requests.get(video_url)
    if response.status_code != 200:
        return "Rejected", "Unable to download video", []

    video_data = response.content

    # Save video data to a temporary file with the correct extension
    file_extension = os.path.splitext(video_url)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_video_file:
        temp_video_file.write(video_data)
        temp_video_path = temp_video_file.name

    try:
        tags = []
        temp_audio_path = None
        temp_frames_dir = None

        # Extract audio and transcribe
        temp_audio_path = temp_video_path + ".mp3"
        command = f"ffmpeg -i \"{temp_video_path}\" -q:a 0 -map a \"{temp_audio_path}\" -y"
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg output: {result.stdout}")
            logger.info(f"FFmpeg error output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e.stderr}")
            return "Rejected", f"Error extracting audio: {e.stderr}", []

        if not os.path.exists(temp_audio_path):
            logger.error(f"Audio file not created: {temp_audio_path}")
            return "Rejected", f"Audio file not created: {temp_audio_path}", []

        with open(temp_audio_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file, response_format="verbose_json")
        text_audio = transcription['text']
        language = transcription['language']
        logger.info(f"Detected language in video audio: {language}")

        # Moderate the transcribed audio text
        status_audio, reason_audio, tags_audio = moderate_text_content(text_audio, policies, sensitivity)
        tags.extend(tags_audio)
        if status_audio == "Rejected":
            return status_audio, reason_audio, tags

        # Extract frames
        temp_frames_dir = tempfile.mkdtemp()
        command = f"ffmpeg -i \"{temp_video_path}\" -vf fps=1 \"{temp_frames_dir}/frame%04d.jpg\""
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg output: {result.stdout}")
            logger.info(f"FFmpeg error output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e.stderr}")
            return "Rejected", f"Error extracting frames: {e.stderr}", []

        if not os.listdir(temp_frames_dir):
            logger.error("No frames were extracted from the video")
            return "Rejected", "No frames were extracted from the video", []

        tags_images = []
        # Moderate each frame
        for frame_file in sorted(os.listdir(temp_frames_dir)):
            frame_path = os.path.join(temp_frames_dir, frame_file)
            with open(frame_path, 'rb') as image_file:
                image_data = image_file.read()

            # Open image
            image = Image.open(frame_path)

            # Moderate the image content
            status_img, reason_img, tags_img = moderate_image_content(image, policies, sensitivity)
            tags.extend(tags_img)
            if status_img == "Rejected":
                return status_img, reason_img, tags

            # Extract text from the image/frame
            extracted_text = extract_text_from_image(image)

            if extracted_text:
                # Moderate the extracted text
                status_text, reason_text, tags_text = moderate_text_content(extracted_text, policies, sensitivity)
                tags.extend(tags_text)
                if status_text == "Rejected":
                    return status_text, reason_text, tags

        # All frames and audio approved
        return "Approved", "Content is appropriate", tags

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return "Rejected", f"Error processing video: {str(e)}", []

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if temp_frames_dir and os.path.exists(temp_frames_dir):
            for f in os.listdir(temp_frames_dir):
                os.unlink(os.path.join(temp_frames_dir, f))
            os.rmdir(temp_frames_dir)
