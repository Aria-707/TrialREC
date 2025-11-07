"""
seguridad_config.py
Sistema de encriptación para proteger datos biométricos según Ley 1581
"""

import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64

# Archivo donde se almacena la clave de encriptación
CLAVE_FILE = 'clave_encriptacion.key'
SALT_FILE = 'salt_encriptacion.key'

def generar_o_cargar_clave():
    """
    Genera una nueva clave de encriptación o carga la existente.
    IMPORTANTE: Guarda estos archivos en un lugar seguro y NO los subas a Git.
    """
    if os.path.exists(CLAVE_FILE) and os.path.exists(SALT_FILE):
        # Cargar clave existente
        with open(CLAVE_FILE, 'rb') as f:
            clave = f.read()
        with open(SALT_FILE, 'rb') as f:
            salt = f.read()
        print("✔ Clave de encriptación cargada")
        return clave, salt
    else:
        # Generar nueva clave
        salt = os.urandom(16)
        password = os.urandom(32)  # Contraseña aleatoria
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        clave = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Guardar clave y salt
        with open(CLAVE_FILE, 'wb') as f:
            f.write(clave)
        with open(SALT_FILE, 'wb') as f:
            f.write(salt)
        
        print("✔ Nueva clave de encriptación generada")
        print("⚠️ IMPORTANTE: Guarda 'clave_encriptacion.key' y 'salt_encriptacion.key' en lugar seguro")
        
        return clave, salt

def obtener_cipher():
    """Retorna el objeto cipher para encriptar/desencriptar"""
    clave, _ = generar_o_cargar_clave()
    return Fernet(clave)

def encriptar_archivo(ruta_archivo):
    """
    Encripta un archivo y lo reemplaza con la versión encriptada.
    """
    try:
        cipher = obtener_cipher()
        
        # Leer archivo
        with open(ruta_archivo, 'rb') as f:
            datos = f.read()
        
        # Encriptar
        datos_encriptados = cipher.encrypt(datos)
        
        # Guardar archivo encriptado
        with open(ruta_archivo + '.enc', 'wb') as f:
            f.write(datos_encriptados)
        
        # Eliminar archivo original
        os.remove(ruta_archivo)
        
        # Renombrar archivo encriptado
        os.rename(ruta_archivo + '.enc', ruta_archivo)
        
        return True
    except Exception as e:
        print(f"Error encriptando {ruta_archivo}: {e}")
        return False

def desencriptar_archivo(ruta_archivo):
    """
    Desencripta un archivo y retorna los datos.
    """
    try:
        cipher = obtener_cipher()
        
        # Leer archivo encriptado
        with open(ruta_archivo, 'rb') as f:
            datos_encriptados = f.read()
        
        # Desencriptar
        datos = cipher.decrypt(datos_encriptados)
        
        return datos
    except Exception as e:
        print(f"Error desencriptando {ruta_archivo}: {e}")
        return None

def encriptar_carpeta_estudiante(nombre_carpeta):
    """
    Encripta todas las imágenes en la carpeta de un estudiante.
    """
    from pathlib import Path
    
    carpeta_path = Path('Data') / nombre_carpeta
    
    if not carpeta_path.exists():
        return False
    
    archivos_encriptados = 0
    for archivo in carpeta_path.glob('*.jpg'):
        if encriptar_archivo(str(archivo)):
            archivos_encriptados += 1
    
    print(f"✔ {archivos_encriptados} imágenes encriptadas en {nombre_carpeta}")
    return True