/*const express = require("express");
const cors = require("cors");
const serverless = require("serverless-http");

const app = express();
const asistenciaRuta = require("../../backend/rutas/regAsistenciaRuta.js");

app.use(cors());
app.use(express.json());
app.use("/.netlify/functions/regAsistencia", asistenciaRuta);

module.exports.handler = serverless(app);*/

const express = require("express");
const cors = require("cors");
const serverless = require("serverless-http");

const app = express();

app.use(cors({
    origin: "*", // También puedes poner tu dominio específico en lugar de "*"
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
  }));

app.use(express.json());
  
const asistenciaRuta = require("../../backend/rutas/regAsistenciaRuta.js");
app.use("/.netlify/functions/regAsistencia", asistenciaRuta);

module.exports.handler = serverless(app);
