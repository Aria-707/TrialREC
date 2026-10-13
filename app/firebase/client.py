import base64
import json
import os

import firebase_admin
from firebase_admin import credentials, firestore

import config  # loads .env before the credential is chosen

SERVICE_ACCOUNT_FILE = "asistenciaconreconocimiento-firebase-adminsdk.json"


def _credential():
    encoded = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
    if not encoded:
        print("✔ Cuenta de servicio desde el archivo local.")
        return credentials.Certificate(SERVICE_ACCOUNT_FILE)
    try:
        account = json.loads(base64.b64decode(encoded))
    except Exception:
        print("✖ FIREBASE_SERVICE_ACCOUNT no se pudo leer. Revisa el base64 del .env.")
        raise
    print("✔ Cuenta de servicio desde .env.")
    return credentials.Certificate(account)


if not firebase_admin._apps:
    firebase_admin.initialize_app(_credential())

db = firestore.client()
