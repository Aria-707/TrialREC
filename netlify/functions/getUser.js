const admin = require('./firebaseAdmin');
 
exports.handler = async (event, context) => {
  try {
    const userId = event.queryStringParameters.id;
    const userDoc = await admin.firestore().collection('asistenciaReconocimiento').doc(userId).get();
 
    if (!userDoc.exists) {
      return {
        statusCode: 404,
        body: JSON.stringify({ error: 'Estudiante no encontrado' }),
      };
    }
 
    return {
      statusCode: 200,
      body: JSON.stringify(userDoc.data()),
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Error al obtener el estudiante.' }),
    };
  }
};