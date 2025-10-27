from flask import Flask, render_template, request, jsonify, redirect, Response, send_file
import cv2
import os
import numpy as np
import time
import base64
import re
import requests
import json
from flask_cors import CORS
from firebase_config import db
from datetime import datetime
import sys
import scheduler_asistencia


app = Flask(__name__)
CORS(app)

# Rutas
dataPath = os.path.join(os.path.dirname(__file__), 'Data')
model_path = os.path.join('backend', 'modeloLBPHReconocimientoOpencv.xml')

# Asegurar que existe la carpeta Data
os.makedirs(dataPath, exist_ok=True)

# Clasificadores
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carga del modelo
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    imagePaths = os.listdir(dataPath)
    label_dict = {name: idx for idx, name in enumerate(imagePaths)}
    next_label = len(imagePaths)
else:
    imagePaths = []
    label_dict = {}
    next_label = 0

print("Model loaded. Persons:", imagePaths)
print("Label dict:", label_dict, " Next label:", next_label)

# Para evitar registros duplicados
cap = None
duracion_reconocimiento = 3
estudiantes_reconocidos = set()
tiempos_reconocimiento = {}
salon_anterior = None  # Para detectar cambios de salón

# ==================== MAPEO DE DÍAS ====================
DIAS_ESPANOL_A_INGLES = {
    'Lunes': 'Monday',
    'Martes': 'Tuesday',
    'Miércoles': 'Wednesday',
    'Jueves': 'Thursday',
    'Viernes': 'Friday',
    'Sábado': 'Saturday',
    'Domingo': 'Sunday'
}

DIAS_INGLES_A_ESPANOL = {v: k for k, v in DIAS_ESPANOL_A_INGLES.items()}

# ==================== FUNCIÓN: NORMALIZAR NOMBRE ====================
def normalizar_nombre(nombre):
    """
    Normaliza un nombre para búsqueda en Firebase.
    - Quita espacios extras
    - Convierte a mayúsculas
    - Quita tildes y acentos
    - Mantiene espacios simples entre palabras
    """
    # Quitar espacios extras
    nombre = nombre.strip()
    nombre = ' '.join(nombre.split())
    
    # Quitar tildes y acentos
    replacements = {
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Ñ': 'N', 'ñ': 'n'
    }
    for orig, repl in replacements.items():
        nombre = nombre.replace(orig, repl)
    
    # Convertir a mayúsculas
    nombre = nombre.upper()
    
    return nombre

# ==================== FUNCIÓN: SANITIZAR PARA FILESYSTEM ====================
def sanitizar_nombre_filesystem(nombre):
    """
    Sanitiza un nombre SOLO para guardar en el filesystem.
    Mantiene la esencia del nombre para coincidir con Firebase después.
    """
    # Quitar tildes comunes
    replacements = {
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Ñ': 'N', 'ñ': 'n'
    }
    for orig, repl in replacements.items():
        nombre = nombre.replace(orig, repl)
    
    # Quitar caracteres especiales EXCEPTO espacios
    nombre = re.sub(r'[^\w\s-]', '', nombre)
    
    # Espacios a guiones bajos
    nombre = nombre.replace(' ', '_')
    
    # Todo a mayúsculas
    nombre = nombre.upper()
    
    return nombre

# ==================== FUNCIÓN: OBTENER CURSO ACTIVO CON VENTANA ====================
def obtener_curso_activo(profesor_id=None):
    """
    Obtiene el curso activo según el horario actual.
    
    VENTANA DE REGISTRO:
    - Desde 5 min ANTES del inicio
    - Hasta 15 min ANTES del final
    
    Ejemplo: Clase 07:00-09:00
    - Registro permitido: 06:55 - 08:45
    
    Returns:
        tuple: (curso_id, hora_inicio_real) o (None, None) si no hay curso
    """
    try:
        from datetime import timedelta
        
        ahora = datetime.now()
        dia_ingles = ahora.strftime('%A')
        dia_espanol = DIAS_INGLES_A_ESPANOL.get(dia_ingles, dia_ingles)
        hora_actual_str = ahora.strftime('%H:%M')
        
        print(f"\n=== BUSCANDO CURSO ACTIVO ===")
        print(f"Día (español): {dia_espanol}")
        print(f"Día (inglés): {dia_ingles}")
        print(f"Hora actual: {hora_actual_str}")
        
        cursos_ref = db.collection('courses')
        if profesor_id:
            cursos_ref = cursos_ref.where('profesorID', '==', profesor_id)
        
        cursos = cursos_ref.get()
        
        for curso_doc in cursos:
            curso_id = curso_doc.id
            curso_data = curso_doc.to_dict()
            schedule = curso_data.get('schedule', [])
            
            print(f"\n  Verificando curso: {curso_id}")
            print(f"  Nombre: {curso_data.get('nameCourse')}")
            
            for horario in schedule:
                dia_horario = horario.get('day', '')
                hora_inicio_str = horario.get('iniTime', '00:00')
                hora_fin_str = horario.get('endTime', '23:59')
                
                print(f"    Schedule: {dia_horario} {hora_inicio_str}-{hora_fin_str}")
                
                # Comparar día
                if dia_horario == dia_espanol or dia_horario == dia_ingles:
                    print(f"      ✓ Día coincide")
                    
                    # Convertir strings a datetime para cálculos
                    hora_inicio = datetime.strptime(hora_inicio_str, '%H:%M')
                    hora_fin = datetime.strptime(hora_fin_str, '%H:%M')
                    hora_actual = datetime.strptime(hora_actual_str, '%H:%M')
                    
                    # VENTANA DE REGISTRO:
                    # - Inicio: 5 min ANTES del inicio de clase
                    # - Fin: 15 min ANTES del final de clase
                    ventana_inicio = hora_inicio - timedelta(minutes=5)
                    ventana_fin = hora_fin - timedelta(minutes=15)
                    
                    print(f"      Ventana de registro: {ventana_inicio.strftime('%H:%M')} - {ventana_fin.strftime('%H:%M')}")
                    
                    # Verificar si está dentro de la ventana
                    if ventana_inicio <= hora_actual <= ventana_fin:
                        print(f"      ✓ Hora dentro del rango de registro")
                        print(f"  [✔] ¡CURSO ACTIVO ENCONTRADO: {curso_id}!")
                        return (curso_id, hora_inicio_str)
                    else:
                        if hora_actual < ventana_inicio:
                            print(f"      ✗ Demasiado temprano (antes de {ventana_inicio.strftime('%H:%M')})")
                        else:
                            print(f"      ✗ Demasiado tarde (después de {ventana_fin.strftime('%H:%M')})")
        
        print(f"\n[!] No se encontró curso activo para {dia_espanol} a las {hora_actual_str}")
        print(f"    NO se registrará asistencia (no hay cursos por defecto)")
        return (None, None)
        
    except Exception as e:
        print(f"[✖] ERROR obteniendo curso activo: {e}")
        import traceback
        traceback.print_exc()
        return (None, None)


# ==================== FUNCIÓN: REGISTRAR ASISTENCIA CON SALÓN ====================
def registrar_asistencia(nombre_estudiante, courseID=None, hora_inicio_clase=None):
    """
    Actualiza la asistencia de un estudiante de 'Ausente' a 'Presente'.
    Si el documento no existe, lo crea automáticamente.
    """
    try:
        from datetime import timedelta
        
        # Obtener salón configurado
        salon_actual = obtener_salon_actual()
        
        # Si no se proporciona courseID, obtenerlo automáticamente CON SALÓN
        if not courseID:
            courseID, hora_inicio_clase = obtener_curso_activo_con_salon(salon_requerido=salon_actual)
        
        # VALIDACIÓN: Si no hay curso activo, NO registrar
        if not courseID:
            print(f"[!] NO SE REGISTRA ASISTENCIA: No hay curso activo")
            return False
        
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        hora_actual_str = datetime.now().strftime('%H:%M')
        
        print(f"\n=== ACTUALIZANDO ASISTENCIA ===")
        print(f"Estudiante: '{nombre_estudiante}'")
        
        # Normalizar el nombre
        nombre_normalizado = normalizar_nombre(nombre_estudiante)
        print(f"Normalizado: '{nombre_normalizado}'")
        print(f"Fecha: {fecha_hoy}")
        print(f"Hora: {hora_actual_str}")
        print(f"Curso: {courseID}")
        
        # Buscar estudiante en Firebase
        personas_ref = db.collection('person')
        query = personas_ref.where('type', '==', 'Estudiante').get()
        
        estudiante_doc = None
        for doc in query:
            data = doc.to_dict()
            nombre_db = data.get('namePerson', '')
            nombre_db_normalizado = normalizar_nombre(nombre_db)
            
            if nombre_db_normalizado == nombre_normalizado:
                estudiante_doc = doc
                break
        
        if not estudiante_doc:
            print(f"[✖] ERROR: Estudiante no encontrado")
            return False
        
        estudianteID = estudiante_doc.id
        print(f"ID encontrado: {estudianteID}")
        
        # ========== CALCULAR TARDANZA ==========
        llegada_tarde = False
        if hora_inicio_clase:
            try:
                hora_inicio = datetime.strptime(hora_inicio_clase, '%H:%M')
                hora_actual_dt = datetime.strptime(hora_actual_str, '%H:%M')
                diferencia = (hora_actual_dt - hora_inicio).total_seconds() / 60
                
                if diferencia > 15:
                    llegada_tarde = True
                    print(f"⚠️ TARDE: {int(diferencia)} minutos")
                else:
                    print(f"✓ A TIEMPO")
            except:
                pass
        
        # Referencia al documento
        asistencia_ref = db.collection('courses').document(courseID).collection('assistances').document(fecha_hoy)
        asistencia_doc = asistencia_ref.get()
        
        # ========== NUEVO: CREAR DOCUMENTO SI NO EXISTE ==========
        if not asistencia_doc.exists:
            print(f"[!] Documento no existe - Creando automáticamente...")
            
            # Obtener estudiantes del curso
            curso_ref = db.collection('courses').document(courseID)
            curso_doc = curso_ref.get()
            
            if not curso_doc.exists:
                print(f"[✖] ERROR: Curso no existe")
                return False
            
            curso_data = curso_doc.to_dict()
            estudiantes_ids = curso_data.get('estudianteID', [])
            
            # Crear documento con todos los estudiantes en "Ausente"
            datos_iniciales = {}
            for est_id in estudiantes_ids:
                datos_iniciales[est_id] = {
                    'estadoAsistencia': 'Ausente',
                    'horaRegistro': None,
                    'late': False
                }
            
            asistencia_ref.set(datos_iniciales)
            print(f"[✔] Documento creado con {len(estudiantes_ids)} estudiantes")
            
            # Obtener el documento recién creado
            asistencia_doc = asistencia_ref.get()
        
        # ========== CONTINUAR CON REGISTRO NORMAL ==========
        datos_existentes = asistencia_doc.to_dict() or {}
        
        # Verificar si el estudiante está en el documento
        if estudianteID not in datos_existentes:
            print(f"[!] ADVERTENCIA: Estudiante no está registrado en este curso")
             
            # ⚠️ CÓDIGO TEMPORAL - ELIMINAR DESPUÉS DE LA PRESENTACIÓN
            # Agregar automáticamente al estudiante al registro de asistencia
            print(f"🔧 AGREGANDO ESTUDIANTE AL DOCUMENTO (TEMPORAL)")
            
            # Verificar si el estudiante tiene el curso "00000"
            estudiante_ref = db.collection('person').document(estudianteID)
            estudiante_doc = estudiante_ref.get()
            
            if estudiante_doc.exists:
                estudiante_data = estudiante_doc.to_dict()
                cursos_estudiante = estudiante_data.get('courses', [])
                
                # Si tiene el curso "00000", lo agregamos al registro
                if '00000' in cursos_estudiante:
                    print(f"   ✓ Estudiante tiene curso de prueba (00000)")
                    print(f"   ✓ Agregando al registro de asistencia...")
                    
                    # Agregar el estudiante al documento con estado "Ausente"
                    datos_existentes[estudianteID] = {
                        'estadoAsistencia': 'Ausente',
                        'horaRegistro': None,
                        'late': False
                    }
                    
                    # Actualizar el documento en Firebase
                    asistencia_ref.update({
                        estudianteID: datos_existentes[estudianteID]
                    })
                    
                    print(f"   ✅ Estudiante agregado al registro")
                    print(f"   ℹ️ Continuando con actualización de asistencia...")
                else:
                    print(f"   ✗ Estudiante no tiene curso de prueba")
                    return False
            else:
                print(f"   ✗ No se encontró el documento del estudiante")
                return False
            # ⚠️ FIN CÓDIGO TEMPORAL
            
        registro_actual = datos_existentes[estudianteID]
        estado_actual = registro_actual.get('estadoAsistencia')
        
        print(f"Estado actual: {estado_actual}")
        
        # Solo actualizar si está en "Ausente"
        if estado_actual == 'Ausente':
            # ACTUALIZAR de Ausente a Presente
            asistencia_ref.update({
                estudianteID: {
                    'estadoAsistencia': 'Presente',
                    'horaRegistro': hora_actual_str,
                    'late': llegada_tarde
                }
            })
            
            print(f"[✔] ACTUALIZADO: Ausente → Presente")
            print(f"    Hora: {hora_actual_str}")
            print(f"    Tarde: {'Sí' if llegada_tarde else 'No'}")
            print(f"=== ACTUALIZACIÓN EXITOSA ===\n")
            return True
        else:
            print(f"[!] Ya registrado como: {estado_actual}")
            return True
        
    except Exception as e:
        print(f"[✖] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== FUNCIONES DE ENTRENAMIENTO ====================
def entrenar_incremental(nuevos_registros):
    """Entrenamiento incremental del modelo."""
    global face_recognizer, label_dict, next_label, imagePaths

    facesData, labels = [], []

    for persona, rutas in nuevos_registros.items():
        if persona not in label_dict:
            label_dict[persona] = next_label
            imagePaths.append(persona)
            next_label += 1

        lbl = label_dict[persona]
        for img_path in rutas:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    facesData.append(img)
                    labels.append(lbl)
            except Exception as e:
                print(f"Error leyendo imagen {img_path}: {e}")

    if facesData:
        face_recognizer.update(facesData, np.array(labels))
        face_recognizer.write(model_path)
        print(f"Entrenamiento incremental: {len(facesData)} imágenes añadidas.")
        return True
    else:
        print("No hay imágenes nuevas para entrenar.")
        return False


# ==================== FUNCIÓN MEJORADA: DETECTAR ROSTRO ====================
def detectar_rostro_mejorado(imagen_gray):
    """
    Detecta rostros con múltiples estrategias.
    Retorna (faces, metodo_usado) o (None, None) si falla.
    """
    try:
        # Estrategia 1: Detección normal
        faces = faceClassif.detectMultiScale(
            imagen_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return faces, "normal"
        
        # Estrategia 2: Más permisivo
        faces = faceClassif.detectMultiScale(
            imagen_gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20)
        )
        if len(faces) > 0:
            return faces, "permisivo"
        
        # Estrategia 3: Ecualizar histograma
        imagen_eq = cv2.equalizeHist(imagen_gray)
        faces = faceClassif.detectMultiScale(
            imagen_eq,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return faces, "ecualizado"
        
    except Exception as e:
        print(f"Error en detección: {e}")
    
    return None, None

# Variable global para almacenar el salón configurado (persistente)
salon_configurado = None
def obtener_salon_configurado_para_scheduler():
     """
     Función callback para que el scheduler obtenga el salón actual.
     """
     return obtener_salon_actual()
SALON_CONFIG_FILE = 'salon_config.txt'

def cargar_salon_persistente():
    """Carga el salón configurado desde archivo."""
    global salon_configurado
    try:
        if os.path.exists(SALON_CONFIG_FILE):
            with open(SALON_CONFIG_FILE, 'r', encoding='utf-8') as f:
                salon_configurado = f.read().strip()
                print(f"✔ Salón cargado desde archivo: '{salon_configurado}'")
                return salon_configurado
    except Exception as e:
        print(f"⚠️ Error cargando salón: {e}")
    return None

def guardar_salon_persistente(salon):
    """Guarda el salón configurado en archivo."""
    try:
        with open(SALON_CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write(salon)
        print(f"💾 Salón guardado: '{salon}'")
        return True
    except Exception as e:
        print(f"❌ Error guardando salón: {e}")
        return False

# ==================== FUNCIÓN: OBTENER TODOS LOS SALONES ====================
def obtener_salones_disponibles():
    """
    Extrae todos los salones únicos de la colección 'courses'.
    Maneja dos casos:
    1. Cursos con subcolección 'groups'
    2. Cursos con campo 'schedule' directo
    
    Returns:
        list: Lista de salones únicos disponibles
    """
    try:
        salones = set()  # Usar set para evitar duplicados
        
        print("\n=== EXTRAYENDO SALONES DISPONIBLES ===")
        
        # Obtener todos los cursos
        cursos_ref = db.collection('courses')
        cursos = cursos_ref.get()
        
        for curso_doc in cursos:
            curso_id = curso_doc.id
            curso_data = curso_doc.to_dict()
            
            print(f"\nProcesando curso: {curso_id}")
            print(f"  Nombre: {curso_data.get('nameCourse', 'Sin nombre')}")
            
            # CASO 1: Verificar si tiene subcolección 'groups'
            try:
                groups_ref = db.collection('courses').document(curso_id).collection('groups')
                groups = groups_ref.get()
                
                if groups:  # Si hay grupos
                    print(f"  ✓ Tiene {len(groups)} grupos")
                    
                    for group_doc in groups:
                        group_id = group_doc.id
                        group_data = group_doc.to_dict()
                        schedule = group_data.get('schedule', [])
                        
                        print(f"    Grupo: {group_id}")
                        
                        for horario in schedule:
                            classroom = horario.get('classroom', '').strip()
                            if classroom:
                                salones.add(classroom)
                                print(f"      → Salón: {classroom}")
                    
                    continue  # Si procesó grupos, pasar al siguiente curso
                    
            except Exception as e:
                print(f"  [!] No tiene subcolección groups o error: {e}")
            
            # CASO 2: Campo 'schedule' directo en el curso
            schedule = curso_data.get('schedule', [])
            
            if schedule:
                print(f"  ✓ Tiene schedule directo con {len(schedule)} horarios")
                
                for horario in schedule:
                    classroom = horario.get('classroom', '').strip()
                    if classroom:
                        salones.add(classroom)
                        print(f"    → Salón: {classroom}")
        
        salones_lista = sorted(list(salones))  # Convertir a lista ordenada
        
        print(f"\n=== SALONES ENCONTRADOS: {len(salones_lista)} ===")
        for salon in salones_lista:
            print(f"  • {salon}")
        print("=" * 50 + "\n")
        
        return salones_lista
        
    except Exception as e:
        print(f"[✖] ERROR obteniendo salones: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==================== FUNCIÓN: CONFIGURAR SALÓN ====================
def configurar_salon(nombre_salon):
    """
    Configura el salón activo para el sistema y lo persiste.
    
    Args:
        nombre_salon: Nombre del salón a configurar
        
    Returns:
        bool: True si se configuró exitosamente
    """
    global salon_configurado
    
    try:
        salon_configurado = nombre_salon.strip()
        guardar_salon_persistente(salon_configurado)
        print(f"\n✅ SALÓN CONFIGURADO: '{salon_configurado}'")
        return True
    except Exception as e:
        print(f"[✖] ERROR configurando salón: {e}")
        return False


# ==================== FUNCIÓN: OBTENER SALÓN ACTUAL ====================
def obtener_salon_actual():
    """
    Retorna el salón actualmente configurado.
    
    Returns:
        str: Nombre del salón configurado o None
    """
    return salon_configurado


# ==================== FUNCIÓN MODIFICADA: OBTENER CURSO ACTIVO CON SALÓN ====================
def obtener_curso_activo_con_salon(profesor_id=None, salon_requerido=None):
    """
    Obtiene el curso activo según el horario actual Y el salón configurado.
    
    Args:
        profesor_id: ID del profesor (opcional)
        salon_requerido: Salón en el que se toma asistencia
    
    Returns:
        tuple: (curso_id, hora_inicio_real) o (None, None) si no hay curso
    """
    try:
        from datetime import timedelta
        
        ahora = datetime.now()
        dia_ingles = ahora.strftime('%A')
        dia_espanol = DIAS_INGLES_A_ESPANOL.get(dia_ingles, dia_ingles)
        hora_actual_str = ahora.strftime('%H:%M')
        
        print(f"\n=== BUSCANDO CURSO ACTIVO CON SALÓN ===")
        print(f"Día (español): {dia_espanol}")
        print(f"Hora actual: {hora_actual_str}")
        print(f"Salón requerido: {salon_requerido}")
        
        if not salon_requerido:
            print(f"[!] No hay salón configurado - no se puede buscar curso")
            return (None, None)
        
        cursos_ref = db.collection('courses')
        if profesor_id:
            cursos_ref = cursos_ref.where('profesorID', '==', profesor_id)
        
        cursos = cursos_ref.get()
        
        for curso_doc in cursos:
            curso_id = curso_doc.id
            curso_data = curso_doc.to_dict()
            
            print(f"\n  Verificando curso: {curso_id}")
            print(f"  Nombre: {curso_data.get('nameCourse')}")
            
            # CASO 1: Verificar grupos (subcolección)
            try:
                groups_ref = db.collection('courses').document(curso_id).collection('groups')
                groups = groups_ref.get()
                
                if groups:
                    print(f"    → Buscando en {len(groups)} grupos...")
                    
                    for group_doc in groups:
                        group_id = group_doc.id
                        group_data = group_doc.to_dict()
                        schedule = group_data.get('schedule', [])
                        
                        resultado = verificar_horario_salon(
                            schedule, 
                            dia_espanol, 
                            dia_ingles, 
                            hora_actual_str, 
                            salon_requerido,
                            f"Grupo {group_id}"
                        )
                        
                        if resultado:
                            print(f"  [✔] ¡CURSO ACTIVO ENCONTRADO: {curso_id} - Grupo {group_id}!")
                            return (curso_id, resultado)
                    
                    continue  # Siguiente curso si ya procesó grupos
                    
            except Exception as e:
                print(f"    [!] No tiene grupos: {e}")
            
            # CASO 2: Schedule directo
            schedule = curso_data.get('schedule', [])
            
            if schedule:
                resultado = verificar_horario_salon(
                    schedule, 
                    dia_espanol, 
                    dia_ingles, 
                    hora_actual_str, 
                    salon_requerido,
                    "Schedule directo"
                )
                
                if resultado:
                    print(f"  [✔] ¡CURSO ACTIVO ENCONTRADO: {curso_id}!")
                    return (curso_id, resultado)
        
        print(f"\n[!] No se encontró curso activo para:")
        print(f"    • Día: {dia_espanol}")
        print(f"    • Hora: {hora_actual_str}")
        print(f"    • Salón: {salon_requerido}")
        return (None, None)
        
    except Exception as e:
        print(f"[✖] ERROR obteniendo curso activo: {e}")
        import traceback
        traceback.print_exc()
        return (None, None)


# ==================== FUNCIÓN AUXILIAR: VERIFICAR HORARIO Y SALÓN ====================
def verificar_horario_salon(schedule, dia_espanol, dia_ingles, hora_actual_str, salon_requerido, origen):
    """
    Verifica si algún horario coincide con día, hora y salón.
    
    Returns:
        str: Hora de inicio si coincide, None si no
    """
    for horario in schedule:
        dia_horario = horario.get('day', '')
        hora_inicio_str = horario.get('iniTime', '00:00')
        hora_fin_str = horario.get('endTime', '23:59')
        classroom = horario.get('classroom', '').strip()
        
        print(f"      [{origen}] {dia_horario} {hora_inicio_str}-{hora_fin_str} @ {classroom}")
        
        # 1. Verificar salón
        if classroom != salon_requerido:
            print(f"        ✗ Salón no coincide (esperado: {salon_requerido})")
            continue
        
        print(f"        ✓ Salón coincide")
        
        # 2. Verificar día
        if dia_horario != dia_espanol and dia_horario != dia_ingles:
            print(f"        ✗ Día no coincide")
            continue
        
        print(f"        ✓ Día coincide")
        
        # 3. Verificar hora
        from datetime import timedelta
        
        hora_inicio = datetime.strptime(hora_inicio_str, '%H:%M')
        hora_fin = datetime.strptime(hora_fin_str, '%H:%M')
        hora_actual = datetime.strptime(hora_actual_str, '%H:%M')
        
        ventana_inicio = hora_inicio - timedelta(minutes=5)
        ventana_fin = hora_fin - timedelta(minutes=15)
        
        print(f"        Ventana: {ventana_inicio.strftime('%H:%M')} - {ventana_fin.strftime('%H:%M')}")
        
        if ventana_inicio <= hora_actual <= ventana_fin:
            print(f"        ✓ Hora dentro del rango")
            return hora_inicio_str
        else:
            print(f"        ✗ Hora fuera del rango")
    
    return None


# ==================== RUTAS ====================
@app.route('/')
def index():
    """
    Ruta principal.
    Carga el salón persistente automáticamente.
    """
    # Cargar salón desde archivo
    salon_actual = cargar_salon_persistente()
    
    if not salon_actual:
        # No hay salón configurado, ir a configuración
        print("⚠️ No hay salón configurado. Redirigiendo a /configuracion")
        return redirect('/configuracion')
    
    # Ya hay salón configurado, mostrar página principal
    print(f"✔ Salón configurado: {salon_actual}")
    return render_template('index.html')

@app.route('/registrar')
def registrar():
    return render_template('registrar.html')

def calcular_categoria_llegada(hora_inicio_str):
    """
    Calcula la categoría de llegada del estudiante basado en la hora actual
    vs la hora de inicio de la clase.
    
    Categorías:
    - "temprano": 5 a 1 minuto antes
    - "justo_a_tiempo": exactamente a la hora o hasta 1 minuto después
    - "llego": 1 a 15 minutos después
    - "tarde": más de 15 minutos después
    
    Returns:
        str: Categoría de llegada
    """
    try:
        from datetime import datetime, timedelta
        
        ahora = datetime.now()
        hora_actual_str = ahora.strftime('%H:%M')
        
        hora_inicio = datetime.strptime(hora_inicio_str, '%H:%M')
        hora_actual = datetime.strptime(hora_actual_str, '%H:%M')
        
        # Diferencia en minutos (positivo = después, negativo = antes)
        diferencia = (hora_actual - hora_inicio).total_seconds() / 60
        
        print(f"📊 Diferencia de tiempo: {diferencia:.1f} minutos")
        
        if diferencia < -1:  # Más de 1 minuto antes
            return "temprano"
        elif -1 <= diferencia <= 1:  # Entre 1 min antes y 1 min después
            return "justo_a_tiempo"
        elif 1 < diferencia <= 15:  # Entre 1 y 15 minutos después
            return "llego"
        else:  # Más de 15 minutos después
            return "tarde"
            
    except Exception as e:
        print(f"Error calculando categoría: {e}")
        return "llego"  # Por defecto

@app.route('/registro', methods=['POST'])
def registro():
    """Endpoint para reconocimiento en tiempo real CON SALÓN"""
    global estudiantes_reconocidos, tiempos_reconocimiento, salon_anterior
    
    try:
        # Verificar que haya salón configurado
        salon_actual = obtener_salon_actual()
        
        if not salon_actual:
            return jsonify({
                "estado": "error",
                "mensaje": "No hay salón configurado. Configura el salón primero."
            }), 400
        
        # DETECTAR CAMBIO DE SALÓN Y LIMPIAR REGISTROS
        if salon_anterior is not None and salon_anterior != salon_actual:
            print(f"\n🔄 CAMBIO DE SALÓN DETECTADO:")
            print(f"   Anterior: '{salon_anterior}'")
            print(f"   Nuevo: '{salon_actual}'")
            print(f"   Limpiando registros previos...")
            estudiantes_reconocidos.clear()
            tiempos_reconocimiento.clear()
            print(f"   ✅ Registros limpiados - se reintentará asistencia\n")
        
        # Actualizar salón anterior
        salon_anterior = salon_actual
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"estado": "error", "mensaje": "No se recibió imagen"}), 400

        image_data = re.sub(r'^data:image/.+;base64,', '', data['image'])
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"estado": "error", "mensaje": "Imagen inválida"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces, metodo = detectar_rostro_mejorado(gray)

        if faces is None or len(faces) == 0:
            return jsonify({"estado": "sin_rostro"})

        x, y, w, h = faces[0]
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        label, confianza = face_recognizer.predict(rostro)

        box = [int(x), int(y), int(w), int(h)]
        
        if confianza < 70 and label < len(imagePaths):
            nombre_carpeta = imagePaths[label]
            nombre_estudiante = nombre_carpeta.replace('_', ' ')
            
            if nombre_estudiante not in tiempos_reconocimiento:
                tiempos_reconocimiento[nombre_estudiante] = time.time()
            elif time.time() - tiempos_reconocimiento[nombre_estudiante] >= duracion_reconocimiento:
                if nombre_estudiante not in estudiantes_reconocidos:
                    estudiantes_reconocidos.add(nombre_estudiante)
                    
                    # Obtener curso activo CON SALÓN
                    courseID, hora_inicio = obtener_curso_activo_con_salon(salon_requerido=salon_actual)
                    
                    if courseID:
                        registrado = registrar_asistencia(nombre_estudiante, courseID, hora_inicio)
                        
                        # CALCULAR CATEGORÍA DE LLEGADA
                        categoria = calcular_categoria_llegada(hora_inicio)
                        
                        return jsonify({
                            "estado": "reconocido",
                            "estudiante": nombre_estudiante,
                            "confianza": float(confianza),
                            "box": box,
                            "salon": salon_actual,
                            "registrado": registrado,
                            "categoria_llegada": categoria  # ← NUEVO
                        })
                    else:
                        print(f"[!] Reconocido '{nombre_estudiante}' pero NO hay curso activo en {salon_actual}")
            
            return jsonify({
                "estado": "reconocido",
                "estudiante": nombre_estudiante,
                "confianza": float(confianza),
                "box": box,
                "salon": salon_actual
            })
        else:
            return jsonify({
                "estado": "desconocido",
                "confianza": float(confianza),
                "box": box
            })
    except Exception as e:
        print(f"Error en /registro: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"estado": "error", "mensaje": str(e)}), 500


@app.route('/detectar_rostro', methods=['POST'])
def detectar_rostro():
    """Detecta si hay un rostro en la imagen."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"rostro_detectado": False}), 200

        image_data = re.sub(r'^data:image/.+;base64,', '', data['image'])
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"rostro_detectado": False}), 200

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, metodo = detectar_rostro_mejorado(gray)

        if faces is not None and len(faces) > 0:
            x, y, w, h = faces[0]
            return jsonify({
                "rostro_detectado": True,
                "box": [int(x), int(y), int(w), int(h)],
                "metodo": metodo
            }), 200
        else:
            return jsonify({"rostro_detectado": False}), 200

    except Exception as e:
        print(f"Error en /detectar_rostro: {e}")
        return jsonify({"rostro_detectado": False, "error": str(e)}), 200


@app.route('/guardar_foto', methods=['POST'])
def guardar_foto():
    """Guarda foto con nombre sanitizado SOLO para filesystem."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "No se recibió data"}), 400
        
        nombre_original = data.get('estudiante', '').strip()
        foto_b64 = data.get('foto', '')
        
        if not nombre_original:
            return jsonify({"ok": False, "error": "Nombre requerido"}), 400
        
        if not foto_b64:
            return jsonify({"ok": False, "error": "Foto requerida"}), 400

        # Sanitizar SOLO para el filesystem
        nombre_filesystem = sanitizar_nombre_filesystem(nombre_original)
        
        print(f"\n💾 Guardando foto:")
        print(f"   Nombre original: '{nombre_original}'")
        print(f"   Nombre filesystem: '{nombre_filesystem}'")
        
        personPath = os.path.join(dataPath, nombre_filesystem)
        
        try:
            os.makedirs(personPath, exist_ok=True)
        except Exception as e:
            print(f"❌ Error creando carpeta: {e}")
            return jsonify({"ok": False, "error": f"Error creando carpeta: {str(e)}"}), 500

        # Decodificar imagen
        try:
            if ',' in foto_b64:
                header, encoded = foto_b64.split(',', 1)
            else:
                encoded = foto_b64
                
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"ok": False, "error": "Imagen inválida"}), 400
            
        except Exception as e:
            print(f"❌ Error decodificando imagen: {e}")
            return jsonify({"ok": False, "error": f"Error decodificando: {str(e)}"}), 500

        # Convertir a escala de grises
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Error en conversión: {str(e)}"}), 500

        # Detectar rostro
        faces = None
        metodo = "ninguno"
        try:
            faces, metodo = detectar_rostro_mejorado(gray)
        except Exception as e:
            print(f"⚠️ Error en detección (continuando): {e}")

        # Preparar imagen
        timestamp = int(time.time() * 1000000)
        filename = f'rostro_{timestamp}.jpg'
        ruta = os.path.join(personPath, filename)

        # Guardar
        try:
            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0]
                rostro = gray[y:y+h, x:x+w]
                rostro_final = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                tipo = "recorte"
            else:
                rostro_final = cv2.resize(gray, (150, 150), interpolation=cv2.INTER_CUBIC)
                tipo = "completa"
            
            success = cv2.imwrite(ruta, rostro_final)
            
            if not success or not os.path.exists(ruta):
                return jsonify({"ok": False, "error": "Error al guardar archivo"}), 500
            
            file_size = os.path.getsize(ruta)
            print(f"✅ Foto guardada: {file_size} bytes ({tipo})")
            
            return jsonify({
                "ok": True, 
                "ruta": ruta,
                "tipo": tipo,
                "metodo": metodo,
                "size": file_size
            }), 200
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"Error guardando: {str(e)}"}), 500

    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Error general: {str(e)}"}), 500


@app.route('/entrenar', methods=['POST'])
def entrenar():
    """
    Entrena el modelo SIN eliminar las carpetas de Data.
    Las fotos individuales se eliminan pero la carpeta permanece.
    """
    try:
        data = request.get_json()
        nombre_original = data.get('estudiante', '').strip()
        
        if not nombre_original:
            return jsonify({"success": False, "error": "Nombre requerido"}), 400

        # Sanitizar igual que en guardar_foto
        nombre_filesystem = sanitizar_nombre_filesystem(nombre_original)
        
        personPath = os.path.join(dataPath, nombre_filesystem)
        
        print(f"\n{'='*60}")
        print(f"🤖 ENTRENANDO MODELO")
        print(f"{'='*60}")
        print(f"Nombre original: '{nombre_original}'")
        print(f"Nombre filesystem: '{nombre_filesystem}'")
        print(f"Carpeta: {personPath}")
        
        if not os.path.exists(personPath):
            return jsonify({
                "success": False, 
                "error": f"No se encontró la carpeta para {nombre_original}"
            }), 404
        
        archivos = [f for f in os.listdir(personPath) if f.endswith('.jpg')]
        
        if len(archivos) == 0:
            return jsonify({
                "success": False,
                "error": "No hay imágenes para entrenar"
            }), 400
        
        print(f"Imágenes encontradas: {len(archivos)}")
        
        nuevas_rutas = [os.path.join(personPath, f) for f in archivos]
        
        # Entrenar con el nombre de la carpeta (nombre_filesystem)
        exito = entrenar_incremental({nombre_filesystem: nuevas_rutas})
        
        if not exito:
            return jsonify({
                "success": False,
                "error": "Falló el entrenamiento"
            }), 500
        
        print(f"✔ Modelo entrenado con {len(nuevas_rutas)} imágenes")
        
        # CAMBIO: NO eliminar las fotos - mantenerlas por seguridad
        print(f"💾 Fotos mantenidas en: {personPath}")
        print(f"   Total de imágenes: {len(archivos)}")
        print(f"   Esto permite reentrenar o mejorar el modelo en el futuro")
        
        print(f"{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   • Carpeta: '{nombre_filesystem}'")
        print(f"   • Imágenes preservadas: {len(archivos)}")
        print(f"   • Modelo guardado en: {model_path}")
        print(f"{'='*60}\n")
        
        # ⚠️ CÓDIGO TEMPORAL - ELIMINAR DESPUÉS DE LA PRESENTACIÓN
        # Registrar estudiante en Firebase automáticamente
        print(f"\n🔥 INICIANDO REGISTRO EN FIREBASE (TEMPORAL)...")
        estudiante_id = registrar_estudiante_en_firebase(nombre_original)
        # ⚠️ FIN CÓDIGO TEMPORAL

        return jsonify({
            "success": True,
            "mensaje": f"Modelo entrenado con {len(nuevas_rutas)} imágenes",
            "imagenes_entrenadas": len(nuevas_rutas),
            "carpeta": nombre_filesystem,
            "firebase_id": estudiante_id  # ⚠️ TEMPORAL - ID generado en Firebase
        }), 200

        
    except Exception as e:
        print(f"✖ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==================== FUNCIÓN TEMPORAL PARA PRUEBAS ====================
# ⚠️ ELIMINAR DESPUÉS DE LA PRESENTACIÓN
# Esta función registra automáticamente estudiantes en Firebase
# con datos de prueba durante el registro facial

def registrar_estudiante_en_firebase(nombre_estudiante):
    """
    FUNCIÓN TEMPORAL - SOLO PARA PRUEBAS
    Registra un estudiante en la colección 'person' de Firebase
    con un ID aleatorio y curso por defecto "00000"
    """
    try:
        import random
        
        # Normalizar el nombre para Firebase
        nombre_normalizado = normalizar_nombre(nombre_estudiante)
        
        print(f"\n{'='*60}")
        print(f"🔥 REGISTRANDO ESTUDIANTE EN FIREBASE (TEMPORAL)")
        print(f"{'='*60}")
        print(f"Nombre original: '{nombre_estudiante}'")
        print(f"Nombre normalizado: '{nombre_normalizado}'")
        
        # Verificar si ya existe
        personas_ref = db.collection('person')
        query = personas_ref.where('type', '==', 'Estudiante').get()
        
        for doc in query:
            data = doc.to_dict()
            nombre_db = data.get('namePerson', '')
            nombre_db_normalizado = normalizar_nombre(nombre_db)
            
            if nombre_db_normalizado == nombre_normalizado:
                print(f"[!] Estudiante ya existe en Firebase con ID: {doc.id}")
                print(f"{'='*60}\n")
                return doc.id
        
        # Generar ID aleatorio entre 200000000000 y 2999999999
        nuevo_id = str(random.randint(20000000000, 2999999999))
        
        # Verificar que el ID no exista (muy raro, pero por seguridad)
        while personas_ref.document(nuevo_id).get().exists:
            nuevo_id = str(random.randint(20000000000, 2999999999))
        
        # Crear el documento
        datos_estudiante = {
            'namePerson': nombre_normalizado,
            'type': 'Estudiante',
            'courses': ['00000']  # Curso por defecto para pruebas
        }
        
        personas_ref.document(nuevo_id).set(datos_estudiante)
        
        print(f"✅ ESTUDIANTE REGISTRADO EXITOSAMENTE")
        print(f"   ID: {nuevo_id}")
        print(f"   Nombre: {nombre_normalizado}")
        print(f"   Tipo: Estudiante")
        print(f"   Cursos: ['00000']")
        print(f"{'='*60}\n")
        
        return nuevo_id
        
    except Exception as e:
        print(f"[✖] ERROR registrando estudiante en Firebase: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== FIN FUNCIÓN TEMPORAL ====================


@app.route('/test_curso')
def test_curso():
    """Endpoint para probar la detección del curso activo CON SALÓN"""
    salon_actual = obtener_salon_actual()
    
    if not salon_actual:
        return jsonify({
            "success": False,
            "mensaje": "No hay salón configurado",
            "salon": None
        })
    
    curso_id, hora_inicio = obtener_curso_activo_con_salon(salon_requerido=salon_actual)
    
    if not curso_id:
        return jsonify({
            "success": False,
            "mensaje": f"No hay curso activo en salón '{salon_actual}'",
            "courseID": None,
            "hora_inicio": None,
            "salon": salon_actual
        })
    
    curso_ref = db.collection('courses').document(curso_id)
    curso_doc = curso_ref.get()
    
    if curso_doc.exists:
        curso_data = curso_doc.to_dict()
        return jsonify({
            "success": True,
            "courseID": curso_id,
            "hora_inicio": hora_inicio,
            "salon": salon_actual,
            "curso": curso_data,
            "mensaje": f"Curso activo encontrado en {salon_actual}"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Curso encontrado pero no existe en BD",
            "courseID": curso_id,
            "salon": salon_actual
        })


def obtener_proximo_curso(salon_requerido):
    """
    Obtiene información del próximo curso que iniciará en el salón configurado.
    
    Returns:
        dict: Información del próximo curso o None
    """
    try:
        from datetime import timedelta
        
        ahora = datetime.now()
        dia_ingles = ahora.strftime('%A')
        dia_espanol = DIAS_INGLES_A_ESPANOL.get(dia_ingles, dia_ingles)
        hora_actual = ahora.strftime('%H:%M')
        
        print(f"\n=== BUSCANDO PRÓXIMO CURSO ===")
        print(f"Día: {dia_espanol}")
        print(f"Hora actual: {hora_actual}")
        print(f"Salón: {salon_requerido}")
        
        cursos_ref = db.collection('courses')
        cursos = cursos_ref.get()
        
        proximos_cursos = []
        
        for curso_doc in cursos:
            curso_id = curso_doc.id
            curso_data = curso_doc.to_dict()
            
            # CASO 1: Verificar grupos
            try:
                groups_ref = db.collection('courses').document(curso_id).collection('groups')
                groups = groups_ref.get()
                
                if groups:
                    for group_doc in groups:
                        group_data = group_doc.to_dict()
                        schedule = group_data.get('schedule', [])
                        
                        for horario in schedule:
                            resultado = buscar_proximo_horario(
                                horario, dia_espanol, dia_ingles,
                                hora_actual, salon_requerido,
                                curso_id, curso_data
                            )
                            if resultado:
                                proximos_cursos.append(resultado)
                    continue
            except:
                pass
            
            # CASO 2: Schedule directo
            schedule = curso_data.get('schedule', [])
            for horario in schedule:
                resultado = buscar_proximo_horario(
                    horario, dia_espanol, dia_ingles,
                    hora_actual, salon_requerido,
                    curso_id, curso_data
                )
                if resultado:
                    proximos_cursos.append(resultado)
        
        if proximos_cursos:
            # Ordenar por tiempo de espera (más cercano primero)
            proximos_cursos.sort(key=lambda x: x['minutos_para_inicio'])
            proximo = proximos_cursos[0]
            
            print(f"✔ Próximo curso encontrado:")
            print(f"   Curso: {proximo['nombre_curso']}")
            print(f"   Inicia: {proximo['hora_inicio']}")
            print(f"   En: {proximo['minutos_para_inicio']} minutos")
            
            return proximo
        
        print(f"[!] No hay más cursos hoy en {salon_requerido}")
        return None
        
    except Exception as e:
        print(f"[✖] ERROR buscando próximo curso: {e}")
        import traceback
        traceback.print_exc()
        return None


def buscar_proximo_horario(horario, dia_espanol, dia_ingles, hora_actual_str, salon_requerido, curso_id, curso_data):
    """
    Busca si un horario es el próximo a iniciar.
    Retorna información del curso si es posterior a la hora actual.
    """
    try:
        dia_horario = horario.get('day', '')
        hora_inicio_str = horario.get('iniTime', '00:00')
        classroom = horario.get('classroom', '').strip()
        
        # Verificar salón
        if classroom != salon_requerido:
            return None
        
        # Verificar día
        if dia_horario != dia_espanol and dia_horario != dia_ingles:
            return None
        
        # Verificar si es posterior a la hora actual
        hora_actual = datetime.strptime(hora_actual_str, '%H:%M')
        hora_inicio = datetime.strptime(hora_inicio_str, '%H:%M')
        
        # Solo considerar cursos futuros (después de la hora actual)
        if hora_inicio <= hora_actual:
            return None
        
        # Calcular minutos para que inicie
        diferencia = (hora_inicio - hora_actual).total_seconds() / 60
        
        return {
            'curso_id': curso_id,
            'nombre_curso': curso_data.get('nameCourse', 'Sin nombre'),
            'hora_inicio': hora_inicio_str,
            'minutos_para_inicio': int(diferencia),
            'salon': salon_requerido
        }
        
    except Exception as e:
        return None
     
@app.route('/api/verificar_curso_activo', methods=['GET'])
def api_verificar_curso_activo():
    """
    Verifica si hay un curso activo en este momento.
    Retorna información sobre si el sistema debe estar en modo espera o reconocimiento.
    """
    try:
        salon_actual = obtener_salon_actual()
        
        if not salon_actual:
            return jsonify({
                "curso_activo": False,
                "salon_configurado": False,
                "mensaje": "No hay salón configurado"
            }), 200
        
        curso_id, hora_inicio = obtener_curso_activo_con_salon(salon_requerido=salon_actual)
        
        if curso_id:
            # Hay curso activo
            curso_ref = db.collection('courses').document(curso_id)
            curso_doc = curso_ref.get()
            
            if curso_doc.exists:
                curso_data = curso_doc.to_dict()
                return jsonify({
                    "curso_activo": True,
                    "salon_configurado": True,
                    "courseID": curso_id,
                    "hora_inicio": hora_inicio,
                    "salon": salon_actual,
                    "nombre_curso": curso_data.get('nameCourse', 'Sin nombre')
                }), 200
        
        # No hay curso activo - calcular próximo curso
        proximo_curso = obtener_proximo_curso(salon_actual)
        
        return jsonify({
            "curso_activo": False,
            "salon_configurado": True,
            "salon": salon_actual,
            "proximo_curso": proximo_curso
        }), 200
        
    except Exception as e:
        print(f"[✖] ERROR en /api/verificar_curso_activo: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "curso_activo": False,
            "error": str(e)
        }), 500
    
@app.route('/configuracion')
def configuracion():
    """Página de configuración inicial del salón"""
    return render_template('configSalon.html')


@app.route('/api/salones', methods=['GET'])
def api_obtener_salones():
    """
    Endpoint para obtener la lista de salones disponibles.
    
    Returns:
        JSON con lista de salones
    """
    try:
        salones = obtener_salones_disponibles()
        
        return jsonify({
            "success": True,
            "salones": salones,
            "total": len(salones)
        }), 200
        
    except Exception as e:
        print(f"[✖] ERROR en /api/salones: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/configurar_salon', methods=['POST'])
def api_configurar_salon():
    """
    Endpoint para configurar el salón activo.
    
    Body:
        {
            "salon": "nombre_del_salon"
        }
    
    Returns:
        JSON con confirmación
    """
    try:
        data = request.get_json()
        salon = data.get('salon', '').strip()
        
        if not salon:
            return jsonify({
                "success": False,
                "error": "Nombre de salón requerido"
            }), 400
        
        # Verificar que el salón existe
        salones_disponibles = obtener_salones_disponibles()
        
        if salon not in salones_disponibles:
            return jsonify({
                "success": False,
                "error": f"El salón '{salon}' no existe en el sistema"
            }), 404
        
        # Configurar salón
        exito = configurar_salon(salon)
        
        if exito:
            return jsonify({
                "success": True,
                "salon": salon,
                "mensaje": f"Salón configurado: {salon}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Error al configurar salón"
            }), 500
            
    except Exception as e:
        print(f"[✖] ERROR en /api/configurar_salon: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/salon_actual', methods=['GET'])
def api_salon_actual():
    """
    Endpoint para obtener el salón actualmente configurado.
    
    Returns:
        JSON con el salón actual
    """
    try:
        salon = obtener_salon_actual()
        
        return jsonify({
            "success": True,
            "salon": salon,
            "configurado": salon is not None
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/limpiar_registros', methods=['POST'])
def api_limpiar_registros():
    """
    Limpia los registros de estudiantes reconocidos.
    Se llama cuando se cambia de salón.
    """
    global estudiantes_reconocidos, tiempos_reconocimiento
    
    try:
        estudiantes_reconocidos.clear()
        tiempos_reconocimiento.clear()
        
        print(f"\n🧹 REGISTROS LIMPIADOS MANUALMENTE")
        print(f"   Se reintentará el registro de asistencia")
        
        return jsonify({
            "success": True,
            "mensaje": "Registros limpiados correctamente"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route("/firebase-config")
def firebase_config():
    return send_file("./firebase_config_public.json")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 INICIANDO SERVIDOR FLASK")
    print("="*60)
    print(f"📁 Data Path: {dataPath}")
    print(f"🤖 Model Path: {model_path}")
    print(f"👥 Personas cargadas: {len(imagePaths)}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Cargar salón persistente al inicio
    salon_inicial = cargar_salon_persistente()
    if salon_inicial:
        print(f"🏫 Salón configurado: {salon_inicial}")
    else:
        print(f"⚠️ No hay salón configurado")
    
    print("="*60 + "\n")
    
    # Iniciar scheduler de asistencia
    print("\n" + "="*60)
    print("⏰ INICIANDO SCHEDULER DE ASISTENCIA")
    print("="*60)
    scheduler_thread = scheduler_asistencia.iniciar_scheduler(obtener_salon_configurado_para_scheduler)
    print("✅ Scheduler iniciado en hilo separado")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)