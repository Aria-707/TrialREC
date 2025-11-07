"""
auditoria.py
Sistema de registro de auditoría para cumplir con Ley 1581
"""

import json
from datetime import datetime
import os

AUDITORIA_FILE = 'logs_auditoria.json'

def registrar_evento(tipo, descripcion, usuario=None, datos_adicionales=None):
    """
    Registra un evento en el log de auditoría.
    
    Tipos de eventos:
    - ACCESO_DATOS: Acceso a datos personales
    - REGISTRO_ESTUDIANTE: Nuevo registro de estudiante
    - MODIFICACION_DATOS: Modificación de datos
    - ELIMINACION_CONSENTIMIENTO: Revocación de consentimiento
    - RECONOCIMIENTO_FACIAL: Reconocimiento exitoso
    """
    try:
        evento = {
            'timestamp': datetime.now().isoformat(),
            'tipo': tipo,
            'descripcion': descripcion,
            'usuario': usuario,
            'datos_adicionales': datos_adicionales or {}
        }
        
        # Cargar logs existentes
        if os.path.exists(AUDITORIA_FILE):
            with open(AUDITORIA_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Agregar nuevo evento
        logs.append(evento)
        
        # Guardar
        with open(AUDITORIA_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Evento registrado: {tipo}")
        
    except Exception as e:
        print(f"⚠️ Error registrando auditoría: {e}")

def obtener_logs(filtro_tipo=None, limite=100):
    """Obtiene los últimos logs de auditoría"""
    try:
        if not os.path.exists(AUDITORIA_FILE):
            return []
        
        with open(AUDITORIA_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        if filtro_tipo:
            logs = [log for log in logs if log['tipo'] == filtro_tipo]
        
        return logs[-limite:]
        
    except Exception as e:
        print(f"Error obteniendo logs: {e}")
        return []