import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from logging_utils import get_logger

# Initialize logger
logger = get_logger("MODEL_LOADER")


def load_model(model_name: str):
    """
    Load a model using its correct model factory based on the model name.

    Args:
        model_name (str): Name of the model to load

    Returns:
        Model instance or None if model cannot be loaded
    """
    # Map model names to their respective platforms and types
    model_mapping = {
        "gpt-4.1": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4_1,
        },
        "claude-3.5-sonnet": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": ModelType.CLAUDE_3_5_SONNET,
        },
        "claude-sonnet-4": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": ModelType.CLAUDE_SONNET_4,
        },
        "qwen3-coder-480b-a35b-instruct": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "qwen3-coder-480b-a35b-instruct",
        },
        "deepseek-v3.1": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "deepseek-v3.1",
        },
        "devstral-small-2507": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "devstral-small-2507",
        },
        "gemini-2.5-pro": {
            "platform": ModelPlatformType.GEMINI,
            "type": ModelType.GEMINI_2_5_PRO,
        },
        "gemini-2.5-flash": {
            "platform": ModelPlatformType.GEMINI,
            "type": ModelType.GEMINI_2_5_FLASH,
        },
        "local": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "local",
        },
    }

    if model_name not in model_mapping:
        logger.warning(f"Unknown model: {model_name}")
        return None

    model_info = model_mapping[model_name]

    try:
        if model_name == "gpt-4.1":
            if os.environ.get("OPENAI_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                )
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return None

        if "claude" in model_name:
            if os.environ.get("ANTHROPIC_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            return None

        if "gemini" in model_name:
            if os.environ.get("GEMINI_API_KEY"):
                return ModelFactory.create(
                    model_platform=model_info["platform"],
                    model_type=model_info["type"],
                    model_config_dict={
                        "stream": True,
                    },
                )
            logger.warning("GEMINI_API_KEY not found in environment variables")
            return None

        # OPENAI compatible / local
        if os.environ.get("LOCAL_API_KEY") and os.environ.get("LOCAL_API_HOST"):
            model_config_dict = {}
            if model_name == "qwen3-coder-480b-a35b-instruct":
                model_config_dict = {
                    "stream": True,
                    "temperature": 0.6,
                }
            elif model_name == "deepseek-v3.1":
                model_config_dict = {
                    "stream": True,
                    "temperature": 0.6,
                }
            elif model_name == "devstral-small-2507":
                model_config_dict = {
                    "stream": True,
                    "temperature": 0.15,
                }
            elif model_name == "local":
                model_config_dict = {
                    # Set the correct temperature for the local model
                    # "temperature": 0.5,
                    "stream": True,
                }

            return ModelFactory.create(
                model_platform=model_info["platform"],
                model_type=model_info["type"],
                api_key=os.environ.get("LOCAL_API_KEY"),
                url=os.environ.get("LOCAL_API_HOST"),
                model_config_dict=model_config_dict,
            )

        logger.warning("LOCAL_API_KEY or LOCAL_API_HOST not found in environment variables")
        return None

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None
