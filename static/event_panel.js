const mode_switch = document.getElementById("modo-evento");
const mode_name = document.getElementById("modo-nombre");
const mode_start = document.getElementById("modo-inicio");
const mode_end = document.getElementById("modo-fin");
const mode_error = document.getElementById("modo-error");
const mode_step = document.getElementById("modo-step");
const salon_step = document.getElementById("salon-step");
const reminder = document.getElementById("aviso-evento");

function publish_event_mode(enabled) {
  window.eventModeEnabled = enabled === true;
  window.dispatchEvent(new CustomEvent("event-mode", {
    detail: { enabled: window.eventModeEnabled },
  }));
}

function lock_event_fields(locked) {
  mode_name.disabled = locked;
  mode_start.disabled = locked;
  mode_end.disabled = locked;
}

function show_event_salon_tab() {
  salon_step.style.display = "block";
  mode_step.style.display = "none";
  document.getElementById("tab-salon").classList.add("active");
  document.getElementById("tab-modo").classList.remove("active");
}

async function fill_event_form() {
  const response = await fetch("/api/event_mode");
  const data = await response.json();
  mode_switch.checked = data.enabled === true;
  mode_name.value = data.name || "";
  mode_start.value = data.ini_time || "";
  mode_end.value = data.end_time || "";
  lock_event_fields(data.enabled !== true);
  publish_event_mode(data.enabled === true);
}

window.show_event_salon_tab = show_event_salon_tab;

document.getElementById("tab-salon").addEventListener("click", show_event_salon_tab);

document.getElementById("tab-modo").addEventListener("click", () => {
  salon_step.style.display = "none";
  mode_step.style.display = "block";
  document.getElementById("tab-salon").classList.remove("active");
  document.getElementById("tab-modo").classList.add("active");
  mode_error.style.display = "none";
  fill_event_form().catch(() => {
    mode_error.textContent = "No se pudo leer el modo evento.";
    mode_error.style.display = "block";
  });
});

mode_switch.addEventListener("change", () => {
  lock_event_fields(!mode_switch.checked);
  mode_error.style.display = "none";
});

document.getElementById("btn-modo-cancelar").addEventListener("click", () => {
  document.getElementById("config-modal").classList.remove("active");
});

function close_reminder() {
  reminder.style.display = "none";
  document.getElementById("config-modal").classList.remove("active");
}

document.getElementById("aviso-evento-cerrar").addEventListener("click", close_reminder);
document.getElementById("aviso-evento-x").addEventListener("click", close_reminder);

document.getElementById("btn-modo-confirmar").addEventListener("click", async () => {
  const button = document.getElementById("btn-modo-confirmar");
  const turning_on = mode_switch.checked;
  const body = { enabled: turning_on };
  if (turning_on) {
    body.name = mode_name.value.trim();
    body.ini_time = mode_start.value;
    body.end_time = mode_end.value;
    if (!body.name || !body.ini_time || !body.end_time) {
      mode_error.textContent = "El nombre, la hora de inicio y la hora final son obligatorios.";
      mode_error.style.display = "block";
      return;
    }
  }

  button.disabled = true;
  mode_error.style.display = "none";
  try {
    const response = await fetch("/api/event_mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || "No se pudo guardar el modo.");
    }
    publish_event_mode(data.enabled === true);
    if (turning_on) {
      reminder.style.display = "flex";
    } else {
      document.getElementById("config-modal").classList.remove("active");
    }
  } catch (error) {
    mode_error.textContent = error.message;
    mode_error.style.display = "block";
  } finally {
    button.disabled = false;
  }
});

fetch("/api/event_mode")
  .then((response) => response.json())
  .then((data) => publish_event_mode(data.enabled === true))
  .catch(() => publish_event_mode(false));
