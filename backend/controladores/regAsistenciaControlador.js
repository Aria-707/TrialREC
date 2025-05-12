const admin = require("../../netlify/functions/firebaseAdmin");

class AsistenciaControlador {
  constructor() {
    this.db = admin.firestore();
    this.collection = this.db.collection("asistenciaReconocimiento");

    //this.consultar = this.consultar.bind(this);
    this.ingresar = this.ingresar.bind(this);
    //this.actualizar = this.actualizar.bind(this);
    //this.borrar = this.borrar.bind(this);
  }

  async ingresar(req, res) {
    try {
      let body = req.body;

      if (Buffer.isBuffer(body)) {
        body = JSON.parse(body.toString("utf8"));
      }

      const nuevaAsistencia = {
        estudiante: body.estudiante,
        estadoAsistencia: body.estadoAsistencia,
        fechaYhora: admin.firestore.Timestamp.now(),
        asignatura: "Física"
      };

      const ref = await this.collection.add(nuevaAsistencia);
      const nuevoDoc = await ref.get();

      res.status(200).json({ id: ref.id, ...nuevoDoc.data() });
    } catch (err) {
        res.status(500).json({
          error: "Error en ingresar: " + err.message
        });
    }
  }
}

module.exports = new AsistenciaControlador();