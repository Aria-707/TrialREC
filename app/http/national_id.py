"""The enrollment id is the cédula, digits only."""


def national_id_from_request(data):
    if not data:
        return None
    national_id = str(data.get("national_id", "")).strip()
    if national_id.isdigit():
        return national_id
    return None
