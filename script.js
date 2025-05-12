function guardar(event) {
    event.preventDefault();
  
    const estudiante = document.getElementById("estudiante").value;
    const estadoAsistencia = document.getElementById("estadoAsistencia").value;
    if (!estadoAsistencia) {
      alert("Por favor selecciona un estado de asistencia.");
      return;
    }
    const data = JSON.stringify({
        estudiante,
        estadoAsistencia
    });
  
    fetch("/.netlify/functions/regAsistencia", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
      .then(response => {
        if (!response.ok) {
          return response.text().then(text => {
            throw new Error(text || "Error al guardar");
          });
        }
        return response.text();
      })
      .then(result => {
        alert("Asistencia registrada");
      })
      .catch(error => {
        alert("Error guardando: " + error.message);
      });
      document.getElementById('popup-confirmacion').style.display = 'block';

      // Limpiar formulario (opcional)
      document.getElementById('estudiante').value = '';
      document.getElementById('estadoAsistencia').value = '';
  }

  function cerrarConfirmacion() {
    document.getElementById('popup-confirmacion').style.display = 'none';
  }
  