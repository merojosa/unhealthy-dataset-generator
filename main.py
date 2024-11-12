import json
from src.generator import generate_dataset
from typing import Any, Dict


class ConfigValidationError(Exception):
    pass


def validate_config(
    user_config: Dict[str, Any], default_config: Dict[str, Any], path: str = ""
) -> None:
    """
    Recursively validate that user_config matches the structure and types of default_config.
    Raises ConfigValidationError if validation fails.
    """
    for key, default_value in default_config.items():
        # Check if required key exists
        if key not in user_config:
            raise ConfigValidationError(
                f"Missing required key '{path}{key}' in config.json"
            )

        user_value = user_config[key]

        # Check if types match
        if type(default_value) != type(user_value):
            raise ConfigValidationError(
                f"Invalid type for '{path}{key}': "
                f"expected {type(default_value).__name__}, "
                f"got {type(user_value).__name__} in config.json"
            )

        # Recursively validate nested dictionaries
        if isinstance(default_value, dict):
            validate_config(user_value, default_value, f"{path}{key}.")


def load_config():
    # Load default config first
    try:
        with open("default_config.json") as f:
            default_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("default_config.json not found!")
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Invalid JSON in default_config.json: {e}")

    try:
        with open("config.json") as f:
            user_config = json.load(f)

            # Validate structure and types
            validate_config(user_config, default_config)

            print("Using config.json")
            return user_config
    except FileNotFoundError:
        print("Using default_config.json")
        return default_config
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Invalid JSON in config.json: {e}")


def main():
    generate_dataset(load_config())


if __name__ == "__main__":
    main()
