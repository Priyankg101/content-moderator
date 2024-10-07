import logging

# Configure logging
logger = logging.getLogger('content_moderation')
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(ch)
