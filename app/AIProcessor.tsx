// AIProcessor.js
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI("AIzaSyBw6iBg7CG4Gnx0Oe0dzsTX24JkSHM9TKQ");

export const analyzeSentiment = async (text: string) => {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });
    const prompt = `Analyze this journal entry and return JSON with:
    - sentiment_score (-1 to 1)
    - primary_emotion
    - suggested_activities
    
    Entry: ${text}`;

    const result = await model.generateContent(prompt);
    return JSON.parse(result.response.text());
  } catch (error) {
    console.error("AI Analysis Error:", error);
    return null;
  }
};


// import React, { useEffect, useState } from "react";
// import { View, Text, Button } from "react-native";
// import * as tf from "@tensorflow/tfjs";
// import { bundleResourceIO } from "@tensorflow/tfjs-react-native";

// // const modelJson = require("./assets/adaptive_learning_model.json"); // If converted to TFJS format
// // const modelWeights = require("./assets/adaptive_learning_model.bin"); // If converted
// const modelTflite = require("../assets/adaptive_learning_model.tflite");
// const Index = () => {
//   const [model, setModel] = useState<any>(null);
//   const [prediction, setPrediction] = useState(null);

//   // Scaler values from Python (replace with your scaler.mean_ and scaler.scale_)
//   const scalerMean: any = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // Example
//   const scalerStd: any = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]; // Example

//   useEffect(() => {
//     const loadModel = async () => {
//       try {
//         await tf.ready(); // Initialize TFJS
//         // Load .tflite model (requires custom setup, see below for pure TFLite)
//         // For TFJS-converted model:
//         // const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
//         setModel(modelTflite);

//         // For now, assume TFJS-compatible model; pure .tflite needs native module
//         console.log("Model loaded (placeholder)");
//       } catch (error) {
//         console.error("Error loading model:", error);
//       }
//     };
//     loadModel();
//   }, []);

//   const normalizeInput = (input: any[]) => {
//     return input.map((val: number, idx: string | number) => (val - scalerMean[idx]) / scalerStd[idx]);
//   };

//   const predictDifficulty = async () => {
//     if (!model) {
//       console.log("Model not loaded yet");
//       return;
//     }

//     // Example user data (11 features)
//     const userData = [0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.85, 0.2, 0.75, 0.9, 0.65];
//     const normalizedData = normalizeInput(userData);

//     // Prepare tensor
//     const inputTensor = tf.tensor2d([normalizedData], [1, 11]);
//     const outputTensor = model.predict(inputTensor);
//     const predictionArray = await outputTensor.data();
//     const difficulty = predictionArray.indexOf(Math.max(...predictionArray));

//     setPrediction(difficulty);
//     console.log("Prediction:", difficulty); // 0=beginner, 1=intermediate, 2=advanced
//   };

//   return (
//     <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
//       <Text>AI Language Learning Companion</Text>
//       <Button title="Predict Difficulty" onPress={predictDifficulty} />
//       {prediction !== null && (
//         <Text>
//           Predicted Difficulty:{" "}
//           {["Beginner", "Intermediate", "Advanced"][prediction]}
//         </Text>
//       )}
//     </View>
//   );
// };

// export default Index;


// import React, { useState } from "react";
// import {
//   View,
//   Text,
//   Button,
//   TextInput,
//   StyleSheet,
//   ScrollView,
//   ActivityIndicator,
// } from "react-native";
// import axios from "axios";

// const Index = () => {
//   const [features, setFeatures] = useState(Array(11).fill(""));
//   const [prediction, setPrediction] = useState(null);
//   const [error, setError] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const predictDifficulty = async () => {
//     setLoading(true);
//     setError(null);
//     try {
//       const userData = {
//         features: features.map((f) => parseFloat(f) || 0),
//       };
//       if (userData.features.some((f) => isNaN(f) || f < 0 || f > 1)) {
//         throw new Error("All inputs must be numbers between 0 and 1");
//       }
//       const response = await axios.post(
//         "http://localhost:8000/predict",
//         userData,
//         {
//           timeout: 5000,
//         }
//       );
//       setPrediction(response.data.difficulty);
//     } catch (err:any:any) {
//       setError(err.message || "Failed to get prediction");
//       setPrediction(null);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const labels = [
//     "Vocabulary",
//     "Grammar",
//     "Listening",
//     "Speaking",
//     "Pronunciation",
//     "Time Spent",
//     "Quiz Rate",
//     "Error Freq",
//     "Confidence",
//     "Completion Rate",
//     "Cultural Quiz",
//   ];

//   return (
//     <ScrollView style={styles.container}>
//       <Text style={styles.title}>AI Language Learning Companion</Text>
//       {labels.map((label, index) => (
//         <View key={index} style={styles.inputContainer}>
//           <Text style={styles.label}>{label} (0-1):</Text>
//           <TextInput
//             style={styles.input}
//             keyboardType="numeric"
//             placeholder="0.0"
//             value={features[index]}
//             onChangeText={(text) => {
//               const newFeatures = [...features];
//               newFeatures[index] = text;
//               setFeatures(newFeatures);
//             }}
//           />
//         </View>
//       ))}
//       <Button
//         title="Predict Difficulty"
//         onPress={predictDifficulty}
//         disabled={loading}
//       />
//       {loading && (
//         <ActivityIndicator size="large" color="#0000ff" style={styles.loader} />
//       )}
//       {prediction !== null && (
//         <Text style={styles.result}>
//           Predicted Difficulty:{" "}
//           {["Beginner", "Intermediate", "Advanced"][prediction]}
//         </Text>
//       )}
//       {error && <Text style={styles.error}>{error}</Text>}
//     </ScrollView>
//   );
// };

// const styles = StyleSheet.create({
//   container: { padding: 20, backgroundColor: "#f5f5f5" },
//   title: {
//     fontSize: 24,
//     textAlign: "center",
//     marginBottom: 20,
//     fontWeight: "bold",
//   },
//   inputContainer: {
//     flexDirection: "row",
//     alignItems: "center",
//     marginBottom: 15,
//   },
//   label: { flex: 1, fontSize: 16 },
//   input: {
//     borderWidth: 1,
//     borderColor: "#ccc",
//     padding: 8,
//     width: 80,
//     borderRadius: 5,
//   },
//   loader: { marginTop: 20 },
//   result: {
//     fontSize: 18,
//     marginTop: 20,
//     color: "#2e7d32",
//     textAlign: "center",
//   },
//   error: { fontSize: 16, marginTop: 20, color: "#d32f2f", textAlign: "center" },
// });

// export default Index;