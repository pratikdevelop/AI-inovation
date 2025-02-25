// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyCBEMkGMTXptCb9nNvktcxWo1OpA2tvnCs",
    authDomain: "ai-inovation.firebaseapp.com",
    projectId: "ai-inovation",
    storageBucket: "ai-inovation.firebasestorage.app",
    messagingSenderId: "139929699058",
    appId: "1:139929699058:web:c12d785d8932dfca32c66b",
    measurementId: "G-ZK72ZYMC2E"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
export { db }; // export db