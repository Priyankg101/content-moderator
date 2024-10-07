import json
import datetime
from moderation.text_moderation import moderate_text
from moderation.image_moderation import moderate_image
from moderation.audio_moderation import moderate_audio
from moderation.video_moderation import moderate_video
from utils.logger import logger

def moderate_content(input_data, policies=None, sensitivity='medium'):
    """
    Moderates a list of content items with multi-language support and customizable policies.

    Args:
        input_data (list): A list of content items to moderate.
        policies (dict): Custom moderation policies.
        sensitivity (str): Sensitivity level ('low', 'medium', 'high').

    Returns:
        dict: A dictionary containing the moderation status, reason, tags, and timestamp.
    """
    overall_status = "Approved"
    overall_reason = "Content is appropriate"
    tags = []

    for item in input_data:
        item_type = item.get('type')
        try:
            if item_type == 'text':
                status, reason, item_tags = moderate_text(item, policies, sensitivity)
            elif item_type == 'image_url':
                status, reason, item_tags = moderate_image(item, policies, sensitivity)
            elif item_type == 'audio_url':
                status, reason, item_tags = moderate_audio(item, policies, sensitivity)
            elif item_type == 'video_url':
                status, reason, item_tags = moderate_video(item, policies, sensitivity)
            else:
                status = "Rejected"
                reason = "Unsupported content type"
                item_tags = []

            tags.extend(item_tags)

            if status == 'Rejected':
                overall_status = 'Rejected'
                overall_reason = reason
                break

        except Exception as e:
            logger.error(f"Error moderating {item_type}: {e}")
            overall_status = 'Rejected'
            overall_reason = f"Error processing {item_type}"
            break

    current_time = datetime.datetime.now().isoformat()

    output = {
        "Status": overall_status,
        "Reason": overall_reason,
        "Tags": list(set(tags)),
        "Time": current_time
    }
    return output

if __name__ == "__main__":
    # Sample input data
    input_data = [
        # {
        #     "type": "image_url",
        #     "image_url": {
        #         "url": "https://api.url2png.com/v6/P4DF2F8BC83648/189f62d5d9da7d7308982fc5650fa4b3/png/?thumbnail_max_width=851&url=pornhub.com&viewport=1280x2000"
        #     }
        # },
        {
            "type": "video_url",
            "video_url": {
                "url": "https://www.redgifs.com/watch/aridserpentineacouchi"
            }
        }
        # Include other content items as needed
    ]

    # Custom policies
    custom_policies = {
        "disallowed_categories": ["harassment", "hate speech", "violence", "explicit_nudity"],
        "allowed_categories": ["mild_language"],
        # Additional policy configurations can be added here
    }

    # Sensitivity level
    sensitivity_level = 'high'  # Options: 'low', 'medium', 'high'

    output = moderate_content(input_data, policies=custom_policies, sensitivity=sensitivity_level)
    print(json.dumps(output, indent=4))
