"""
scheduler_asistencia.py
Sistema de inicialización automática de documentos de asistencia.
Verifica cada minuto si debe crear documentos de asistencia para cursos activos.
"""

import time
import threading
from datetime import datetime, timedelta
from firebase_config import db


# Mapeo de días
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

# Diccionario para rastrear qué documentos ya fueron inicializados
documentos_inicializados = set()  # Formato: "courseID_fecha"


def obtener_cursos_proximos_a_iniciar(salon_configurado):
    """
    Obtiene los cursos que están a punto de iniciar (5 minutos antes).
    
    Args:
        salon_configurado: Salón actualmente configurado
        
    Returns:
        list: Lista de tuplas (courseID, fecha, hora_inicio)
    """
    try:
        ahora = datetime.now()
        dia_ingles = ahora.strftime('%A')
        dia_espanol = DIAS_INGLES_A_ESPANOL.get(dia_ingles, dia_ingles)
        hora_actual = ahora.strftime('%H:%M')
        fecha_actual = ahora.strftime('%Y-%m-%d')
        
        # Calcular ventana de 5 minutos antes
        hora_en_5_min = (ahora + timedelta(minutes=5)).strftime('%H:%M')
        
        cursos_a_inicializar = []
        
        print(f"\n⏰ [{ahora.strftime('%H:%M:%S')}] Verificando cursos próximos a iniciar...")
        print(f"   Día: {dia_espanol} ({dia_ingles})")
        print(f"   Hora actual: {hora_actual}")
        print(f"   Hora en 5 min: {hora_en_5_min}")
        print(f"   Salón configurado: {salon_configurado}")
        
        if not salon_configurado:
            print("   [!] No hay salón configurado - saltando verificación")
            return []
        
        # Obtener todos los cursos
        cursos_ref = db.collection('courses')
        cursos = cursos_ref.get()
        
        for curso_doc in cursos:
            curso_id = curso_doc.id
            curso_data = curso_doc.to_dict()
            
            # CASO 1: Verificar grupos (subcolección)
            try:
                groups_ref = db.collection('courses').document(curso_id).collection('groups')
                groups = groups_ref.get()
                
                if groups:
                    for group_doc in groups:
                        group_data = group_doc.to_dict()
                        schedule = group_data.get('schedule', [])
                        
                        for horario in schedule:
                            resultado = verificar_horario_inicio(
                                horario, dia_espanol, dia_ingles, 
                                hora_actual, hora_en_5_min, salon_configurado
                            )
                            
                            if resultado:
                                cursos_a_inicializar.append((curso_id, fecha_actual, resultado))
                                print(f"   ✅ Curso a inicializar: {curso_id} a las {resultado}")
                    
                    continue
            except Exception as e:
                pass
            
            # CASO 2: Schedule directo
            schedule = curso_data.get('schedule', [])
            
            for horario in schedule:
                resultado = verificar_horario_inicio(
                    horario, dia_espanol, dia_ingles,
                    hora_actual, hora_en_5_min, salon_configurado
                )
                
                if resultado:
                    cursos_a_inicializar.append((curso_id, fecha_actual, resultado))
                    print(f"   ✅ Curso a inicializar: {curso_id} a las {resultado}")
        
        if not cursos_a_inicializar:
            print(f"   ℹ️ No hay cursos próximos a iniciar")
        
        return cursos_a_inicializar
        
    except Exception as e:
        print(f"   ❌ ERROR obteniendo cursos: {e}")
        return []


def verificar_horario_inicio(horario, dia_espanol, dia_ingles, hora_actual, hora_en_5_min, salon_requerido):
    """
    Verifica si un horario está a punto de iniciar (exactamente 5 minutos antes).
    
    Returns:
        str: Hora de inicio si coincide, None si no
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
        
        # Verificar si estamos exactamente 5 minutos antes
        # Comparar si hora_en_5_min coincide con hora_inicio
        hora_inicio = datetime.strptime(hora_inicio_str, '%H:%M')
        hora_5min = datetime.strptime(hora_en_5_min, '%H:%M')
        hora_act = datetime.strptime(hora_actual, '%H:%M')
        
        # Ventana: estamos entre 5 y 6 minutos antes del inicio
        diferencia = (hora_inicio - hora_act).total_seconds() / 60
        
        if 4 <= diferencia <= 6:  # Ventana de 2 minutos para capturar
            return hora_inicio_str
        
        return None
        
    except Exception as e:
        return None


def inicializar_documento_asistencia(course_id, fecha):
    """
    Crea el documento de asistencia con todos los estudiantes del curso en estado 'Ausente'.
    
    Args:
        course_id: ID del curso
        fecha: Fecha en formato YYYY-MM-DD
        
    Returns:
        bool: True si se inicializó correctamente
    """
    try:
        print(f"\n{'='*70}")
        print(f"📋 INICIALIZANDO DOCUMENTO DE ASISTENCIA")
        print(f"{'='*70}")
        print(f"   Curso: {course_id}")
        print(f"   Fecha: {fecha}")
        
        # Verificar si ya fue inicializado
        clave_documento = f"{course_id}_{fecha}"
        if clave_documento in documentos_inicializados:
            print(f"   [!] Ya fue inicializado previamente en esta sesión")
            return True
        
        # Referencia al documento de asistencia
        asistencia_ref = db.collection('courses').document(course_id).collection('assistances').document(fecha)
        
        # Verificar si ya existe en Firebase
        asistencia_doc = asistencia_ref.get()
        if asistencia_doc.exists:
            print(f"   [!] El documento ya existe en Firebase")
            documentos_inicializados.add(clave_documento)
            return True
        
        # Obtener información del curso
        curso_ref = db.collection('courses').document(course_id)
        curso_doc = curso_ref.get()
        
        if not curso_doc.exists:
            print(f"   ❌ ERROR: Curso {course_id} no existe")
            return False
        
        curso_data = curso_doc.to_dict()
        estudiantes_ids = curso_data.get('estudianteID', [])
        
        if not estudiantes_ids or len(estudiantes_ids) == 0:
            print(f"   ⚠️ ADVERTENCIA: No hay estudiantes inscritos en el curso")
            print(f"   ℹ️ Creando documento vacío")
            asistencia_ref.set({})
            documentos_inicializados.add(clave_documento)
            return True
        
        print(f"   📚 Estudiantes inscritos: {len(estudiantes_ids)}")
        
        # Crear diccionario con todos los estudiantes en estado "Ausente"
        datos_asistencia = {}
        
        for estudiante_id in estudiantes_ids:
            datos_asistencia[estudiante_id] = {
                'estadoAsistencia': 'Ausente',
                'horaRegistro': None,
                'late': False
            }
        
        # Crear el documento en Firebase
        asistencia_ref.set(datos_asistencia)
        
        # Marcar como inicializado
        documentos_inicializados.add(clave_documento)
        
        print(f"   ✅ Documento creado exitosamente")
        print(f"   📊 Total de estudiantes: {len(estudiantes_ids)}")
        print(f"   📝 Estado inicial: Ausente")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR inicializando documento: {e}")
        import traceback
        traceback.print_exc()
        return False


def limpiar_registros_antiguos():
    """
    Limpia el registro de documentos inicializados de días anteriores.
    Se ejecuta una vez al día a la medianoche.
    """
    global documentos_inicializados
    fecha_actual = datetime.now().strftime('%Y-%m-%d')
    
    # Filtrar solo los documentos del día actual
    documentos_inicializados = {
        doc for doc in documentos_inicializados 
        if doc.endswith(fecha_actual)
    }
    
    print(f"\n🧹 Limpieza de registros antiguos completada")
    print(f"   Documentos en memoria: {len(documentos_inicializados)}")


def tarea_programada(salon_configurado_callback):
    """
    Tarea que se ejecuta cada minuto para verificar cursos a inicializar.
    
    Args:
        salon_configurado_callback: Función que retorna el salón actual configurado
    """
    print("\n" + "="*70)
    print("🚀 SCHEDULER DE ASISTENCIA INICIADO")
    print("="*70)
    print("   ⏰ Verificación: Cada 60 segundos")
    print("   📋 Inicialización: 5 minutos antes de cada clase")
    print("   🧹 Limpieza: Diariamente a la medianoche")
    print("="*70 + "\n")
    
    ultima_limpieza = datetime.now().date()
    
    while True:
        try:
            # Verificar si es un nuevo día para limpiar registros
            hoy = datetime.now().date()
            if hoy > ultima_limpieza:
                limpiar_registros_antiguos()
                ultima_limpieza = hoy
            
            # Obtener salón configurado
            salon_actual = salon_configurado_callback()
            
            if salon_actual:
                # Buscar cursos próximos a iniciar
                cursos = obtener_cursos_proximos_a_iniciar(salon_actual)
                
                # Inicializar documentos
                for curso_id, fecha, hora_inicio in cursos:
                    inicializar_documento_asistencia(curso_id, fecha)
            
        except Exception as e:
            print(f"\n❌ ERROR en tarea programada: {e}")
            import traceback
            traceback.print_exc()
        
        # Esperar 60 segundos antes de la siguiente verificación
        time.sleep(60)


def iniciar_scheduler(salon_configurado_callback):
    """
    Inicia el scheduler en un hilo separado.
    
    Args:
        salon_configurado_callback: Función que retorna el salón actual
    """
    thread = threading.Thread(
        target=tarea_programada, 
        args=(salon_configurado_callback,),
        daemon=True  # Se cierra cuando termina el programa principal
    )
    thread.start()
    return thread