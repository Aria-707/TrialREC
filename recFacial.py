from flask import Flask, render_template, request, jsonify, redirect, Response
import cv2
import os
import numpy as np
import time
import base64
import re
import requests
import json
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import sys

# Inicializar Firebase
cred = credentials.Certificate('asistenciaconreconocimiento-firebase-adminsdk.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

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


# ==================== FUNCIÓN: REGISTRAR ASISTENCIA MEJORADA ====================
def registrar_asistencia(nombre_estudiante, courseID=None, hora_inicio_clase=None):
    """
    Registra la asistencia en Firestore.
    MEJORADO: 
    - Maneja caso sin curso activo
    - Detecta llegadas tarde (>30 min después del inicio)
    - Agrega campo 'late' booleano
    
    Args:
        nombre_estudiante: Nombre del estudiante
        courseID: ID del curso (si es None, se busca automáticamente)
        hora_inicio_clase: Hora de inicio real de la clase (formato "HH:MM")
    """
    try:
        from datetime import timedelta
        
        # Si no se proporciona courseID, obtenerlo automáticamente
        if not courseID:
            courseID, hora_inicio_clase = obtener_curso_activo()
        
        # VALIDACIÓN: Si no hay curso activo, NO registrar
        if not courseID:
            print(f"[!] NO SE REGISTRA ASISTENCIA: No hay curso activo en este momento")
            print(f"    El reconocimiento seguirá funcionando pero no guardará registros")
            return False
        
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        hora_actual_str = datetime.now().strftime('%H:%M')
        hora_actual = datetime.now()
        
        print(f"\n=== REGISTRANDO ASISTENCIA ===")
        print(f"Estudiante recibido: '{nombre_estudiante}'")
        
        # Normalizar el nombre recibido
        nombre_normalizado = normalizar_nombre(nombre_estudiante)
        print(f"Nombre normalizado: '{nombre_normalizado}'")
        print(f"Fecha: {fecha_hoy}")
        print(f"Hora registro: {hora_actual_str}")
        print(f"Curso: {courseID}")
        print(f"Hora inicio clase: {hora_inicio_clase}")
        
        # Buscar estudiante en Firebase
        personas_ref = db.collection('person')
        query = personas_ref.where('type', '==', 'Estudiante').get()
        
        estudiante_doc = None
        for doc in query:
            data = doc.to_dict()
            nombre_db = data.get('namePerson', '')
            nombre_db_normalizado = normalizar_nombre(nombre_db)
            
            print(f"  Comparando:")
            print(f"    DB: '{nombre_db}' → '{nombre_db_normalizado}'")
            print(f"    Buscado: '{nombre_estudiante}' → '{nombre_normalizado}'")
            
            if nombre_db_normalizado == nombre_normalizado:
                estudiante_doc = doc
                print(f"  ✔ ¡COINCIDENCIA ENCONTRADA!")
                break
        
        if not estudiante_doc:
            print(f"[✖] ERROR: No se encontró estudiante '{nombre_estudiante}'")
            print(f"    Nombre normalizado buscado: '{nombre_normalizado}'")
            return False
        
        estudianteID = estudiante_doc.id
        estudiante_data = estudiante_doc.to_dict()
        nombre_real = estudiante_data.get('namePerson', '')
        
        print(f"EstudianteID encontrado: {estudianteID}")
        print(f"Nombre real en Firebase: '{nombre_real}'")
        
        # Verificar si está inscrito en el curso
        curso_ref = db.collection('courses').document(courseID)
        curso_doc = curso_ref.get()
        
        if curso_doc.exists:
            curso_data = curso_doc.to_dict()
            estudiantes_curso = curso_data.get('estudianteID', [])
            
            if estudianteID not in estudiantes_curso:
                print(f"[!] ADVERTENCIA: Estudiante {estudianteID} no inscrito en curso {courseID}")
                print(f"    Estudiantes del curso: {estudiantes_curso}")
        
        # ========== CALCULAR SI LLEGÓ TARDE ==========
        # REGLA: Se considera TARDE si llegó más de 30 min después del inicio
        # Ejemplo: Clase a las 07:00
        #   - A tiempo: 06:55 - 07:30
        #   - Tarde: 07:31 en adelante
        
        llegada_tarde = False
        
        if hora_inicio_clase:
            try:
                # Convertir hora de inicio a datetime
                hora_inicio = datetime.strptime(hora_inicio_clase, '%H:%M')
                hora_actual_dt = datetime.strptime(hora_actual_str, '%H:%M')
                
                # Calcular diferencia en minutos
                diferencia = (hora_actual_dt - hora_inicio).total_seconds() / 60
                
                # Se considera tarde si llegó MÁS DE 30 minutos después del inicio
                if diferencia > 30:
                    llegada_tarde = True
                    print(f"⚠️  LLEGADA TARDE DETECTADA")
                    print(f"    Hora inicio clase: {hora_inicio_clase}")
                    print(f"    Hora llegada: {hora_actual_str}")
                    print(f"    Diferencia: {int(diferencia)} minutos después del inicio")
                    print(f"    Límite puntualidad: 30 minutos")
                else:
                    # Llegó a tiempo o incluso antes
                    if diferencia < 0:
                        print(f"✓ Llegada ANTICIPADA")
                        print(f"    Llegó {int(abs(diferencia))} minutos ANTES del inicio")
                    else:
                        print(f"✓ Llegada A TIEMPO")
                        print(f"    Llegó {int(diferencia)} minutos después del inicio (dentro del límite)")
                    
            except Exception as e:
                print(f"⚠️ Error calculando tardanza: {e}")
                # Si hay error, por defecto no marca como tarde
                llegada_tarde = False
        
        # Referencia al documento de asistencia
        asistencia_ref = db.collection('courses').document(courseID).collection('assistances').document(fecha_hoy)
        asistencia_doc = asistencia_ref.get()
        
        # ========== DATOS DE ASISTENCIA CON CAMPO "late" ==========
        datos_asistencia = {
            estudianteID: {
                'estadoAsistencia': 'Presente',
                'horaRegistro': hora_actual_str,
                'late': llegada_tarde  # ← NUEVO CAMPO
            }
        }
        
        if asistencia_doc.exists:
            datos_existentes = asistencia_doc.to_dict() or {}
            
            if estudianteID in datos_existentes:
                print(f"[!] El estudiante ya tiene asistencia registrada")
                print(f"    Registro existente: {datos_existentes[estudianteID]}")
                return True
            
            # Actualizar documento existente
            asistencia_ref.update(datos_asistencia)
            print(f"[✔] Asistencia ACTUALIZADA")
        else:
            # Crear nuevo documento
            asistencia_ref.set(datos_asistencia)
            print(f"[✔] Asistencia CREADA")
        
        print(f"    Ruta: courses/{courseID}/assistances/{fecha_hoy}/{estudianteID}")
        print(f"    Estado: Presente")
        print(f"    Tarde: {'Sí ⚠️' if llegada_tarde else 'No ✓'}")
        print(f"=== REGISTRO EXITOSO ===\n")
        return True
        
    except Exception as e:
        print(f"[✖] ERROR registrando asistencia: {e}")
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


# ==================== RUTAS ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registrar')
def registrar():
    return render_template('registrar.html')

@app.route('/registro', methods=['POST'])
def registro():
    """Endpoint para reconocimiento en tiempo real"""
    try:
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
            # Obtener nombre de la carpeta (sanitizado)
            nombre_carpeta = imagePaths[label]
            
            # Convertir nombre de carpeta a nombre real
            nombre_estudiante = nombre_carpeta.replace('_', ' ')
            
            if nombre_estudiante not in tiempos_reconocimiento:
                tiempos_reconocimiento[nombre_estudiante] = time.time()
            elif time.time() - tiempos_reconocimiento[nombre_estudiante] >= duracion_reconocimiento:
                if nombre_estudiante not in estudiantes_reconocidos:
                    estudiantes_reconocidos.add(nombre_estudiante)
                    
                    # Obtener curso activo y hora de inicio
                    courseID, hora_inicio = obtener_curso_activo()
                    
                    # Solo registrar si hay curso activo
                    if courseID:
                        registrar_asistencia(nombre_estudiante, courseID, hora_inicio)
                    else:
                        print(f"[!] Reconocido '{nombre_estudiante}' pero NO hay curso activo - no se registra")
            
            return jsonify({
                "estado": "reconocido",
                "estudiante": nombre_estudiante,
                "confianza": float(confianza),
                "box": box
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
        
        return jsonify({
            "success": True,
            "mensaje": f"Modelo entrenado con {len(nuevas_rutas)} imágenes",
            "imagenes_entrenadas": len(nuevas_rutas),
            "carpeta": nombre_filesystem
        }), 200
        
    except Exception as e:
        print(f"✖ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/test_curso')
def test_curso():
    """Endpoint para probar la detección del curso activo"""
    curso_id, hora_inicio = obtener_curso_activo()
    
    if not curso_id:
        return jsonify({
            "success": False,
            "mensaje": "No hay curso activo en este momento",
            "courseID": None,
            "hora_inicio": None
        })
    
    curso_ref = db.collection('courses').document(curso_id)
    curso_doc = curso_ref.get()
    
    if curso_doc.exists:
        curso_data = curso_doc.to_dict()
        return jsonify({
            "success": True,
            "courseID": curso_id,
            "hora_inicio": hora_inicio,
            "curso": curso_data,
            "mensaje": "Curso activo encontrado"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Curso encontrado pero no existe en BD",
            "courseID": curso_id
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 INICIANDO SERVIDOR FLASK")
    print("="*60)
    print(f"📁 Data Path: {dataPath}")
    print(f"🤖 Model Path: {model_path}")
    print(f"👥 Personas cargadas: {len(imagePaths)}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)