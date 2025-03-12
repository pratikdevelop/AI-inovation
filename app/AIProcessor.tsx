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
