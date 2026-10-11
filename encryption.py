"""Fernet encryption for the biometric photos."""
import base64
import os
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from constants import ENCRYPTION_KEY_FILE, ENCRYPTION_SALT_FILE, PHOTO_DIR_NAME


def load_or_create_key():
    """Load the existing key. A new key is created only when both files are missing."""
    if os.path.exists(ENCRYPTION_KEY_FILE) and os.path.exists(ENCRYPTION_SALT_FILE):
        with open(ENCRYPTION_KEY_FILE, "rb") as handle:
            key = handle.read()
        with open(ENCRYPTION_SALT_FILE, "rb") as handle:
            salt = handle.read()
        print("✔ Clave de encriptación cargada")
        return key, salt

    salt = os.urandom(16)
    password = os.urandom(32)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )

    key = base64.urlsafe_b64encode(kdf.derive(password))

    with open(ENCRYPTION_KEY_FILE, "wb") as handle:
        handle.write(key)
    with open(ENCRYPTION_SALT_FILE, "wb") as handle:
        handle.write(salt)

    print("✔ Nueva clave de encriptación generada")
    print(f"⚠️ IMPORTANTE: Guarda '{ENCRYPTION_KEY_FILE}' y '{ENCRYPTION_SALT_FILE}' en lugar seguro")
    return key, salt


def cipher():
    """Fernet object. The salt is stored and is not used to decrypt."""
    key, _salt = load_or_create_key()
    return Fernet(key)


def encrypt_file(file_path):
    """Replace a file with its ciphertext, keeping the same path."""
    try:
        fernet = cipher()
        with open(file_path, "rb") as handle:
            data = handle.read()
        encrypted = fernet.encrypt(data)
        with open(file_path + ".enc", "wb") as handle:
            handle.write(encrypted)
        os.remove(file_path)
        os.rename(file_path + ".enc", file_path)
        return True
    except Exception as error:
        print(f"Error encriptando el archivo {file_path}: {error}")
        return False


def decrypt_file(file_path):
    """Return the decrypted bytes of a photo."""
    try:
        fernet = cipher()
        with open(file_path, "rb") as handle:
            encrypted = handle.read()
        return fernet.decrypt(encrypted)
    except Exception as error:
        print(f"Error desencriptando el archivo {file_path}: {error}")
        return None


def encrypt_student_folder(folder_name):
    """Encrypt every jpg in one student folder."""
    folder_path = Path(PHOTO_DIR_NAME) / folder_name
    if not folder_path.exists():
        return False

    encrypted_count = 0
    for image_file in folder_path.glob("*.jpg"):
        if encrypt_file(str(image_file)):
            encrypted_count += 1

    print(f"✔ {encrypted_count} imágenes encriptadas en {folder_name}")
    return True
