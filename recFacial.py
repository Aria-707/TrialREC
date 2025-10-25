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

# Inicializar Firebase
cred = credentials.Certificate('asistenciaconreconocimiento-firebase-adminsdk.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
CORS(app)

# Rutas
dataPath = os.path.join(os.path.dirname(__file__), 'Data')
model_path = os.path.join('backend', 'modeloLBPHReconocimientoOpencv.xml')

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

# ==================== FUNCIÓN: OBTENER CURSO ACTIVO ====================
def obtener_curso_activo(profesor_id=None):
    """
    Obtiene el curso activo según el horario actual.
    Compatible con tu estructura de schedule como array.
    """
    try:
        ahora = datetime.now()
        # Obtener día en inglés (Monday, Tuesday, etc.)
        dia_ingles = ahora.strftime('%A')
        # Convertir a español
        dia_espanol = DIAS_INGLES_A_ESPANOL.get(dia_ingles, dia_ingles)
        hora_actual = ahora.strftime('%H:%M')
        
        print(f"\n=== BUSCANDO CURSO ACTIVO ===")
        print(f"Día (español): {dia_espanol}")
        print(f"Día (inglés): {dia_ingles}")
        print(f"Hora: {hora_actual}")
        
        # Consultar cursos
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
            print(f"  Schedule: {schedule}")
            
            # schedule es un array de objetos
            for horario in schedule:
                dia_horario = horario.get('day', '')
                hora_inicio = horario.get('iniTime', '00:00')
                hora_fin = horario.get('endTime', '23:59')
                
                print(f"    - Día horario: {dia_horario}, {hora_inicio} - {hora_fin}")
                
                # Comparar día (acepta tanto español como inglés)
                if dia_horario == dia_espanol or dia_horario == dia_ingles:
                    # Comparar hora
                    if hora_inicio <= hora_actual <= hora_fin:
                        print(f"  [✔] ¡CURSO ACTIVO ENCONTRADO!")
                        print(f"  ID: {curso_id}")
                        print(f"  Horario: {dia_horario} {hora_inicio}-{hora_fin}")
                        print(f"=== FIN BÚSQUEDA ===\n")
                        return curso_id
        
        print(f"[!] No se encontró curso activo para {dia_espanol} a las {hora_actual}")
        print(f"    Usando curso por defecto: '0000'")
        print(f"=== FIN BÚSQUEDA ===\n")
        return '0000'  # Tu curso de prueba
        
    except Exception as e:
        print(f"[✖] ERROR obteniendo curso activo: {e}")
        import traceback
        traceback.print_exc()
        return '0000'


# ==================== FUNCIÓN: REGISTRAR ASISTENCIA ====================
def registrar_asistencia(nombre_estudiante, courseID=None):
    """
    Registra la asistencia en: courses/{courseID}/assistances/{YYYY-MM-DD}/{estudianteID}
    Compatible con tu estructura donde person tiene ID numérico como documento.
    """
    try:
        # Si no se proporciona courseID, obtenerlo automáticamente
        if not courseID:
            courseID = obtener_curso_activo()
        
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        hora_actual = datetime.now().strftime('%H:%M')
        
        print(f"\n=== REGISTRANDO ASISTENCIA ===")
        print(f"Estudiante: {nombre_estudiante}")
        print(f"Fecha: {fecha_hoy}")
        print(f"Hora: {hora_actual}")
        print(f"Curso: {courseID}")
        
        # IMPORTANTE: Buscar estudiante por nombre
        # Ajusta 'namePerson' al nombre exacto del campo en tu colección person
        personas_ref = db.collection('person')
        query = personas_ref.where('namePerson', '==', nombre_estudiante).limit(1)
        resultados = query.get()
        
        if not resultados:
            print(f"[✖] ERROR: No se encontró estudiante '{nombre_estudiante}' en colección 'person'")
            print(f"    Verifica que el nombre coincida exactamente con el campo 'namePerson'")
            return False
        
        estudiante_doc = resultados[0]
        estudianteID = estudiante_doc.id
        estudiante_data = estudiante_doc.to_dict()
        
        print(f"EstudianteID encontrado: {estudianteID}")
        
        # Verificar si el curso tiene al estudiante inscrito
        # Tu estructura tiene estudianteID en el curso, no courses en person
        curso_ref = db.collection('courses').document(courseID)
        curso_doc = curso_ref.get()
        
        if curso_doc.exists:
            curso_data = curso_doc.to_dict()
            estudiantes_curso = curso_data.get('estudianteID', [])
            
            if estudianteID not in estudiantes_curso:
                print(f"[!] ADVERTENCIA: Estudiante {estudianteID} no está en estudianteID del curso {courseID}")
                print(f"    Estudiantes del curso: {estudiantes_curso}")
        
        # Referencia al documento de asistencia
        asistencia_ref = db.collection('courses').document(courseID).collection('assistances').document(fecha_hoy)
        asistencia_doc = asistencia_ref.get()
        
        # Datos de asistencia
        datos_asistencia = {
            estudianteID: {
                'estadoAsistencia': 'Presente',
                'horaRegistro': hora_actual
            }
        }
        
        if asistencia_doc.exists:
            datos_existentes = asistencia_doc.to_dict() or {}
            
            if estudianteID in datos_existentes:
                print(f"[!] El estudiante ya tiene asistencia registrada hoy")
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
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                facesData.append(img)
                labels.append(lbl)

    if facesData:
        face_recognizer.update(facesData, np.array(labels))
        face_recognizer.write(model_path)
        print(f"Entrenamiento incremental: {len(facesData)} imágenes añadidas.")
    else:
        print("No hay imágenes nuevas para entrenar.")


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
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"estado": "error", "mensaje": "No se recibió imagen"}), 400

    # Decodificar imagen
    image_data = re.sub(r'^data:image/.+;base64,', '', data['image'])
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"estado": "sin_rostro"})

    # Procesar primer rostro
    x, y, w, h = faces[0]
    rostro = auxFrame[y:y+h, x:x+w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
    label, confianza = face_recognizer.predict(rostro)

    box = [int(x), int(y), int(w), int(h)]
    
    if confianza < 70 and label < len(imagePaths):
        nombre = imagePaths[label]
        
        # Lógica de registro único
        if nombre not in tiempos_reconocimiento:
            tiempos_reconocimiento[nombre] = time.time()
        elif time.time() - tiempos_reconocimiento[nombre] >= duracion_reconocimiento:
            if nombre not in estudiantes_reconocidos:
                estudiantes_reconocidos.add(nombre)
                registrar_asistencia(nombre)  # USA LA NUEVA FUNCIÓN
        
        return jsonify({
            "estado": "reconocido",
            "estudiante": nombre,
            "confianza": float(confianza),
            "box": box
        })
    else:
        return jsonify({
            "estado": "desconocido",
            "confianza": float(confianza),
            "box": box
        })

@app.route('/detectar_rostro', methods=['POST'])
def detectar_rostro():
    """
    Detecta si hay un rostro en la imagen sin guardarla.
    Usado para feedback visual en tiempo real.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"rostro_detectado": False}), 200

        # Decodificar imagen
        image_data = re.sub(r'^data:image/.+;base64,', '', data['image'])
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"rostro_detectado": False}), 200

        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            return jsonify({
                "rostro_detectado": True,
                "box": [int(x), int(y), int(w), int(h)]
            }), 200
        else:
            return jsonify({"rostro_detectado": False}), 200

    except Exception as e:
        return jsonify({"rostro_detectado": False, "error": str(e)}), 200


@app.route('/guardar_foto', methods=['POST'])
def guardar_foto():
    """
    Guarda foto detectando y recortando la cara.
    CORREGIDO: Ahora guarda correctamente en disco.
    """
    try:
        data = request.get_json()
        nombre = data.get('estudiante', '').strip()
        foto_b64 = data.get('foto', '')
        
        if not nombre or not foto_b64:
            return jsonify({"ok": False, "error": "Faltan datos"}), 400

        # Sanitizar nombre
        nombre = re.sub(r'[^\w\s-]', '', nombre)  # Quitar caracteres especiales
        nombre = nombre.replace(' ', '_')  # Espacios a guiones bajos
        
        personPath = os.path.join(dataPath, nombre)
        os.makedirs(personPath, exist_ok=True)

        # Decodificar imagen
        if ',' in foto_b64:
            header, encoded = foto_b64.split(',', 1)
        else:
            encoded = foto_b64
            
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"ok": False, "error": "Imagen inválida"}), 400

        # Detectar cara
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"ok": False, "msg": "no_face"}), 200

        # Recortar primera cara
        x, y, w, h = faces[0]
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Guardar con timestamp único
        timestamp = int(time.time() * 1000000)  # Microsegundos para más unicidad
        ruta = os.path.join(personPath, f'rostro_{timestamp}.jpg')
        
        # GUARDAR EN DISCO
        success = cv2.imwrite(ruta, rostro)
        
        if success:
            print(f"  ✔ Foto guardada: {ruta}")
            return jsonify({"ok": True, "ruta": ruta}), 200
        else:
            print(f"  ✖ Error al guardar: {ruta}")
            return jsonify({"ok": False, "error": "Error al escribir archivo"}), 500

    except Exception as e:
        print(f"  ✖ Error en guardar_foto: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/entrenar', methods=['POST'])
def entrenar():
    """
    Entrena el modelo con las fotos capturadas del estudiante.
    Limpia las fotos después del entrenamiento.
    """
    try:
        data = request.get_json()
        nombre = data.get('estudiante', '').strip()
        
        if not nombre:
            return jsonify({"success": False, "error": "Nombre requerido"}), 400

        # Sanitizar nombre (igual que en guardar_foto)
        nombre_sanitizado = re.sub(r'[^\w\s-]', '', nombre)
        nombre_sanitizado = nombre_sanitizado.replace(' ', '_')
        
        personPath = os.path.join(dataPath, nombre_sanitizado)
        
        print(f"\n{'='*60}")
        print(f"🤖 ENTRENANDO MODELO PARA: {nombre}")
        print(f"{'='*60}")
        print(f"Carpeta: {personPath}")
        
        if not os.path.exists(personPath):
            return jsonify({
                "success": False, 
                "error": f"No se encontraron fotos para {nombre}"
            }), 404
        
        # Obtener todas las imágenes
        archivos = [f for f in os.listdir(personPath) if f.endswith('.jpg')]
        
        if len(archivos) == 0:
            return jsonify({
                "success": False,
                "error": "No hay imágenes para entrenar"
            }), 400
        
        print(f"Imágenes encontradas: {len(archivos)}")
        
        # Preparar rutas completas
        nuevas_rutas = [os.path.join(personPath, f) for f in archivos]
        
        # Entrenar incrementalmente
        entrenar_incremental({nombre: nuevas_rutas})
        
        print(f"✔ Modelo entrenado con {len(nuevas_rutas)} imágenes")
        
        # Limpiar imágenes temporales
        print(f"🗑️  Limpiando archivos temporales...")
        for ruta in nuevas_rutas:
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
            except Exception as e:
                print(f"  Error eliminando {ruta}: {e}")
        
        # Eliminar carpeta si está vacía
        try:
            if os.path.exists(personPath) and len(os.listdir(personPath)) == 0:
                os.rmdir(personPath)
                print(f"✔ Carpeta temporal eliminada")
        except Exception as e:
            print(f"  Error eliminando carpeta: {e}")
        
        print(f"{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO PARA: {nombre}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "success": True,
            "mensaje": f"Modelo entrenado con {len(nuevas_rutas)} imágenes",
            "imagenes_entrenadas": len(nuevas_rutas)
        }), 200
        
    except Exception as e:
        print(f"✖ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



# ==================== TEST ENDPOINT ====================
@app.route('/test_curso')
def test_curso():
    """Endpoint para probar la detección del curso activo"""
    curso_id = obtener_curso_activo()
    
    # Obtener info del curso
    curso_ref = db.collection('courses').document(curso_id)
    curso_doc = curso_ref.get()
    
    if curso_doc.exists:
        curso_data = curso_doc.to_dict()
        return jsonify({
            "success": True,
            "courseID": curso_id,
            "curso": curso_data
        })
    else:
        return jsonify({
            "success": False,
            "error": "Curso no encontrado",
            "courseID": curso_id
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 INICIANDO SERVIDOR FLASK")
    print("="*60)
    print(f"📁 Data Path: {dataPath}")
    print(f"🤖 Model Path: {model_path}")
    print(f"👥 Personas cargadas: {len(imagePaths)}")
    print("="*60 + "\n")
    
    app.run(debug=True)