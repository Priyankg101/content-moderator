import os
import tempfile
import requests
import openai
from openai import OpenAI
from utils.logger import logger
from utils.config import OPENAI_API_KEY
from utils.language_detection import detect_language

client = OpenAI(api_key=OPENAI_API_KEY)

def moderate_audio(item, policies=None, sensitivity='medium'):
    """
    Moderates the audio by transcribing and analyzing the text with multi-language support and customizable policies.

    Args:
        item (dict): The audio item to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        tuple: A tuple containing the status ('Approved' or 'Rejected'), reason, and tags.
    """
    audio_url = item.get('audio_url', {}).get('url')
    if not audio_url:
        return "Rejected", "No audio URL provided", []

    response = requests.get(audio_url)
    if response.status_code != 200:
        return "Rejected", "Unable to download audio", []

    audio_data = response.content

    # Save audio data to a temporary file with the correct extension
    file_extension = os.path.splitext(audio_url)[1] or '.mp3'
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_path = temp_audio_file.name

    try:
        # Transcribe audio using OpenAI's Whisper API with language detection
        with open(temp_audio_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file, response_format="verbose_json")
        text = transcription['text']
        language = transcription['language']
        logger.info(f"Detected language: {language}")

        # Moderate the transcribed text
        status, reason, tags = moderate_text_content(text, policies, sensitivity)
        return status, reason, tags

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return "Rejected", "Error transcribing audio", []

    finally:
        os.unlink(temp_audio_path)

def moderate_text_content(text, policies, sensitivity):
    """
    Moderates the given text with customizable policies and sensitivity.

    Args:
        text (str): The text to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        tuple: A tuple containing the status, reason, and tags.
    """
    # Prepare policy instructions
    policy_instructions = create_policy_instructions(policies, sensitivity)

    # Use GPT-4 for moderation, requesting the response in English
    prompt = f"""As an AI content moderation assistant, analyze the following transcribed audio text for compliance with community guidelines. {policy_instructions} Identify any issues related to disallowed content such as harassment, hate speech, explicit content, privacy violations, and misinformation. Provide a decision ('Approved' or 'Rejected'), reasons, and relevant tags. The response should be in English, regardless of the text's language.

Transcribed Text:
{text}

Response:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )

    content = response.choices[0].message.content.strip()

    # Parse the response
    status, reason, tags = parse_moderation_response(content)
    return status, reason, tags

def parse_moderation_response(response_text):
    """
    Parses the GPT-4 moderation response.

    Args:
        response_text (str): The GPT-4 response text.

    Returns:
        tuple: A tuple containing the status, reason, and tags.
    """
    # Reuse the function from text_moderation.py
    from moderation.text_moderation import parse_moderation_response
    return parse_moderation_response(response_text)

def create_policy_instructions(policies, sensitivity):
    """
    Creates policy instructions for the GPT-4 prompt based on custom policies and sensitivity level.

    Args:
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        str: Policy instructions for the prompt.
    """
    # Reuse the function from text_moderation.py
    from moderation.text_moderation import create_policy_instructions
    return create_policy_instructions(policies, sensitivity)
