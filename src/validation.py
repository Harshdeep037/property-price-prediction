"""Check the data sent by the website before it reaches the model."""
import math

from src import config


def validate_input(data):
    """Validate a dict of raw inputs.

    Returns (clean_values, errors). If `errors` is not empty, show it to the user
    (it maps a field name to a helpful message).
    """
    if not isinstance(data, dict):
        return None, {"_form": "Send the inputs as a JSON object."}

    clean, errors = {}, {}

    for field, (low, high) in config.INPUT_LIMITS.items():
        raw = data.get(field)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors[field] = "This field is required."
            continue
        if isinstance(raw, bool):
            errors[field] = "Enter a number."
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[field] = "Enter a valid number."
            continue
        if not math.isfinite(value):
            errors[field] = "Enter a valid number."
        elif not low <= value <= high:
            errors[field] = f"Must be between {low:g} and {high:g}."
        else:
            clean[field] = value

    ocean = data.get("ocean_proximity")
    if ocean is None or (isinstance(ocean, str) and ocean.strip() == ""):
        errors["ocean_proximity"] = "This field is required."
    elif ocean not in config.OCEAN_CATEGORIES:
        errors["ocean_proximity"] = "Choose one of: " + ", ".join(config.OCEAN_CATEGORIES)
    else:
        clean["ocean_proximity"] = ocean

    # Cross-field check (only if both numbers were individually valid)
    if "total_rooms" in clean and "total_bedrooms" in clean:
        if clean["total_bedrooms"] > clean["total_rooms"]:
            errors["total_bedrooms"] = "Bedrooms cannot be more than total rooms."

    return (None, errors) if errors else (clean, {})
