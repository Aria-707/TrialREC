const admin = require("../../netlify/functions/firebaseAdmin");

class AsistenciaControlador {
  constructor() {
    this.db = admin.firestore();
    this.ingresar = this.ingresar.bind(this);
  }

  async ingresar(req, res) {
    try {
      let body = req.body;

      if (Buffer.isBuffer(body)) {
        body = JSON.parse(body.toString("utf8"));
      }

      const { estudiante, estadoAsistencia, courseID = 'default_course' } = body;

      if (!estudiante || !estadoAsistencia) {
        return res.status(400).json({
          error: "Faltan datos requeridos: estudiante y estadoAsistencia"
        });
      }

      console.log("\n=== REGISTRANDO ASISTENCIA (Node.js) ===");
      console.log("Estudiante:", estudiante);
      console.log("Estado:", estadoAsistencia);
      console.log("Curso:", courseID);

      // 1. Obtener fecha y hora actual
      const now = new Date();
      const fechaHoy = now.toISOString().split('T')[0]; // YYYY-MM-DD
      const horaActual = now.toTimeString().slice(0, 5); // HH:mm

      console.log("Fecha:", fechaHoy);
      console.log("Hora:", horaActual);

      // 2. Buscar el ID del estudiante por nombre
      const personasRef = this.db.collection('person');
      const query = personasRef
        .where('namePerson', '==', estudiante)
        .where('type', '==', 'Estudiante')
        .limit(1);
      
      const snapshot = await query.get();

      if (snapshot.empty) {
        console.log(`[✖] ERROR: No se encontró estudiante '${estudiante}'`);
        return res.status(404).json({
          error: `No se encontró el estudiante '${estudiante}' en la base de datos`
        });
      }

      const estudianteDoc = snapshot.docs[0];
      const estudianteID = estudianteDoc.id;
      const estudianteData = estudianteDoc.data();

      console.log("EstudianteID encontrado:", estudianteID);

      // 3. Verificar inscripción en el curso
      const cursosEstudiante = estudianteData.courses || [];
      if (!cursosEstudiante.includes(courseID)) {
        console.log(`[!] ADVERTENCIA: Estudiante no inscrito en curso ${courseID}`);
      }

      // 4. Referencia al documento de asistencia
      const asistenciaRef = this.db
        .collection('courses')
        .doc(courseID)
        .collection('assistances')
        .doc(fechaHoy);

      // 5. Obtener documento actual
      const asistenciaDoc = await asistenciaRef.get();

      // 6. Datos de asistencia del estudiante
      const datosAsistencia = {
        [estudianteID]: {
          estadoAsistencia: estadoAsistencia,
          horaRegistro: horaActual
        }
      };

      if (asistenciaDoc.exists) {
        // Verificar si ya existe registro
        const datosExistentes = asistenciaDoc.data();
        
        if (datosExistentes && datosExistentes[estudianteID]) {
          console.log("[!] Ya existe registro previo para este estudiante");
          return res.status(200).json({
            mensaje: "El estudiante ya tiene asistencia registrada hoy",
            registroExistente: datosExistentes[estudianteID],
            estudianteID: estudianteID,
            fecha: fechaHoy
          });
        }

        // Actualizar documento existente
        await asistenciaRef.update(datosAsistencia);
        console.log("[✔] Asistencia ACTUALIZADA");
      } else {
        // Crear nuevo documento
        await asistenciaRef.set(datosAsistencia);
        console.log("[✔] Asistencia CREADA");
      }

      console.log("=== REGISTRO EXITOSO ===\n");

      res.status(200).json({
        mensaje: "Asistencia registrada exitosamente",
        estudianteID: estudianteID,
        estudiante: estudiante,
        fecha: fechaHoy,
        hora: horaActual,
        curso: courseID,
        estado: estadoAsistencia
      });

    } catch (err) {
      console.error("[✖] ERROR en ingresar:", err);
      res.status(500).json({
        error: "Error en ingresar: " + err.message
      });
    }
  }
}

module.exports = new AsistenciaControlador();