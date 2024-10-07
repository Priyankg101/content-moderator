# Unchanged except for adding moderate_text_content function for reuse

import openai
from openai import OpenAI
from utils.logger import logger
from utils.config import OPENAI_API_KEY
from utils.language_detection import detect_language
import json

client = OpenAI(api_key=OPENAI_API_KEY)

def moderate_text(item, policies=None, sensitivity='medium'):
    """
    Moderates the given text with multi-language support.

    Args:
        item (dict): The text item to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level ('low', 'medium', 'high').

    Returns:
        tuple: A tuple containing the status ('Approved' or 'Rejected'), reason, and tags.
    """
    text_content = item.get('text')
    language = item.get('language')

    # If language is not provided, detect it
    if not language:
        language = detect_language(text_content)
    logger.info(f"Detected language: {language}")

    # Moderate the text content
    status, reason, tags = moderate_text_content(text_content, policies, sensitivity)
    return status, reason, tags

def moderate_text_content(text, policies=None, sensitivity='medium'):
    """
    Moderates the given text content using text-moderation-latest and GPT-4 if needed.

    Args:
        text (str): The text content to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        tuple: A tuple containing the status, reason, and tags.
    """
    # First, use text-moderation-latest
    moderation_response = client.moderations.create(input=text)
    result = moderation_response.results[0]
    print(result)
    return use_gpt4_for_moderation(text, policies, sensitivity)
    # if result.flagged:
    #     # If flagged by text-moderation-latest, use GPT-4 for detailed analysis
    #     return use_gpt4_for_moderation(text, policies, sensitivity)
    # else:
    #     # If not flagged, return approved status
    #     return "Approved", "Content does not violate community guidelines", []

def use_gpt4_for_moderation(text, policies, sensitivity):
    policy_instructions = create_policy_instructions(policies, sensitivity)

    prompt = f"""As an AI content moderation assistant, analyze the following text for compliance with community guidelines. {policy_instructions}

Consider the context and use of idiomatic expressions. Do not flag content that uses figurative language or common expressions unless they genuinely promote disallowed content. Focus on the overall intent and meaning of the text.

Identify any issues related to disallowed content such as harassment, hate speech, explicit content, privacy violations, and misinformation. Provide a decision ('Approved' or 'Rejected'), reasons, and relevant tags.

The response should be in English, regardless of the text's language.

Please return your response in the following JSON format:

{{
    "decision": "Approved" or "Rejected",
    "reason": "Brief explanation of the decision",
    "tags": ["tag1", "tag2", "tag3"]
}}

Text:
{text}

Response:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )

    content = response.choices[0].message.content.strip()
    logger.info(f"GPT-4 response: {content}")

    try:
        moderation_result = json.loads(content)
        status = moderation_result['decision']
        reason = moderation_result['reason']
        tags = moderation_result['tags']
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response from GPT-4")
        status = "Rejected"
        reason = "Error in moderation process"
        tags = []

    return status, reason, tags

def create_policy_instructions(policies, sensitivity):
    """
    Creates policy instructions for the GPT-4 prompt based on custom policies and sensitivity level.

    Args:
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level.

    Returns:
        str: Policy instructions for the prompt.
    """
    instructions = ""

    # Sensitivity level adjustments
    sensitivity_levels = {
        'low': "Be lenient in your analysis, only flag severe violations.",
        'medium': "Apply standard moderation guidelines.",
        'high': "Be strict in your analysis, flag even minor violations."
    }

    instructions += sensitivity_levels.get(sensitivity, "Apply standard moderation guidelines.")

    # Custom policies
    if policies:
        disallowed = policies.get('disallowed_categories', [])
        allowed = policies.get('allowed_categories', [])

        if disallowed:
            instructions += f" Disallowed content categories include: {', '.join(disallowed)}."
        if allowed:
            instructions += f" Allowed content categories include: {', '.join(allowed)}."

    return instructions
