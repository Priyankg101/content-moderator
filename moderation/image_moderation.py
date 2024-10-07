import io
import requests
import base64
import openai
from openai import OpenAI
from PIL import Image
import pytesseract
import subprocess
from utils.logger import logger
from utils.config import OPENAI_API_KEY
from utils.language_detection import detect_language
from moderation.text_moderation import moderate_text_content

client = OpenAI(api_key=OPENAI_API_KEY)

def moderate_image(item, policies=None, sensitivity='medium'):
    """
    Moderates the image using OpenAI's Image Moderation API and OCR for text detection.

    Args:
        item (dict): The image item to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        tuple: A tuple containing the status ('Approved' or 'Rejected'), reason, and tags.
    """
    image_url = item.get('image_url', {}).get('url')
    if not image_url:
        return "Rejected", "No image URL provided", []

    response = requests.get(image_url)
    if response.status_code != 200:
        return "Rejected", "Unable to download image", []

    image_data = response.content

    # Open image
    image = Image.open(io.BytesIO(image_data))

    # Moderate the image content
    status, reason, tags = moderate_image_content(image, policies, sensitivity)
    print("image moderation status, reason, tags", status, reason, tags)
    if status == "Rejected":
        return status, reason, tags

    # Check if Tesseract is installed before attempting OCR
    if is_tesseract_installed():
        extracted_text = extract_text_from_image(image)
        if extracted_text:
            # Moderate the extracted text
            status_text, reason_text, tags_text = moderate_text_content(extracted_text, policies, sensitivity)
            tags.extend(tags_text)
            if status_text == "Rejected":
                return status_text, reason_text, tags
    else:
        logger.warning("Tesseract is not installed. Skipping OCR and text moderation.")

    return "Approved", "Content is appropriate", tags

def moderate_image_content(image, policies, sensitivity):
    """
    Moderates the visual content of the image using DALL-E's latest version.

    Args:
        image (PIL.Image): The image to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level (not used in this implementation).

    Returns:
        tuple: A tuple containing the status, reason, and tags.
    """
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    try:
        # Attempt to edit the image using DALL-E
        response = client.images.edit(
            image=image_bytes,
            prompt="check and return an error if image is not appropriate with I want you to blur the image that are not appropriate",
            n=1,
            size="1024x1024"
        )
        print("DALL-E response", response)
        # If we reach this point, DALL-E accepted the image
        return "Approved", "Image content is appropriate", []
    
    except openai.BadRequestError as e:
        # DALL-E rejected the image
        error_message = str(e)
        return "Rejected", f"DALL-E rejected the image: {error_message}", ["DALL-E rejection"]

def extract_text_from_image(image):
    """
    Extracts text from an image using OCR.

    Args:
        image (PIL.Image): The image to extract text from.

    Returns:
        str: The extracted text, or None if extraction failed.
    """
    try:
        text = pytesseract.image_to_string(image)
        text = text.strip()
        return text if text else None
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not in your PATH. OCR functionality is disabled.")
        return None
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return None

# Add this function to check if Tesseract is installed
def is_tesseract_installed():
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False
