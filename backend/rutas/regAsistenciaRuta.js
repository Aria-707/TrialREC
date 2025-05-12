const express = require("express");
const router = express.Router();
const controlador = require("../controladores/regAsistenciaControlador.js");

router.post("/", controlador.ingresar);
module.exports = router;
