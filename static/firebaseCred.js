import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";

export async function initFirebase() {
  try {
    const res = await fetch("/firebase-config");
    const config = await res.json();

    const app = initializeApp(config);
    const auth = getAuth(app);
    const db = getFirestore(app);

    console.log("🔥 Firebase inicializado correctamente (cliente)");
    return { app, auth, db };
  } catch (err) {
    console.error("Error cargando configuración Firebase:", err);
    throw err;
  }
}
