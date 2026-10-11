"""Folder names for photos on disk."""
import re


def sanitize_folder_name(name):
    """Folder name only. Spaces become underscores. This is not the Firestore name."""
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Ñ": "N", "ñ": "n",
    }
    for original, replacement in replacements.items():
        name = name.replace(original, replacement)

    name = re.sub(r"[^\w\s-]", "", name)
    name = name.replace(" ", "_")
    name = name.upper()
    return name
