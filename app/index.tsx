import React, { useState } from "react";
import {
  View,
  Text,
  Button,
  TextInput,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import axios from "axios";

const Index = () => {
  const [features, setFeatures] = useState(Array(11).fill(""));
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const predictDifficulty = async () => {
    setLoading(true);
    setError(null);
    try {
      const userData = {
        features: features.map((f) => parseFloat(f) || 0),
      };
      if (userData.features.some((f) => isNaN(f) || f < 0 || f > 1)) {
        throw new Error("All inputs must be numbers between 0 and 1");
      }
      const response = await axios.post(
        "http://localhost:8000/predict",
        userData,
        {
          timeout: 5000,
        }
      );
      setPrediction(response.data.difficulty);
    } catch (err:any) {
      setError(err.message || "Failed to get prediction");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const labels = [
    "Vocabulary",
    "Grammar",
    "Listening",
    "Speaking",
    "Pronunciation",
    "Time Spent",
    "Quiz Rate",
    "Error Freq",
    "Confidence",
    "Completion Rate",
    "Cultural Quiz",
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>AI Language Learning Companion</Text>
      {labels.map((label, index) => (
        <View key={index} style={styles.inputContainer}>
          <Text style={styles.label}>{label} (0-1):</Text>
          <TextInput
            style={styles.input}
            keyboardType="numeric"
            placeholder="0.0"
            value={features[index]}
            onChangeText={(text) => {
              const newFeatures = [...features];
              newFeatures[index] = text;
              setFeatures(newFeatures);
            }}
          />
        </View>
      ))}
      <Button
        title="Predict Difficulty"
        onPress={predictDifficulty}
        disabled={loading}
      />
      {loading && (
        <ActivityIndicator size="large" color="#0000ff" style={styles.loader} />
      )}
      {prediction !== null && (
        <Text style={styles.result}>
          Predicted Difficulty:{" "}
          {["Beginner", "Intermediate", "Advanced"][prediction]}
        </Text>
      )}
      {error && <Text style={styles.error}>{error}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { padding: 20, backgroundColor: "#f5f5f5" },
  title: {
    fontSize: 24,
    textAlign: "center",
    marginBottom: 20,
    fontWeight: "bold",
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 15,
  },
  label: { flex: 1, fontSize: 16 },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 8,
    width: 80,
    borderRadius: 5,
  },
  loader: { marginTop: 20 },
  result: {
    fontSize: 18,
    marginTop: 20,
    color: "#2e7d32",
    textAlign: "center",
  },
  error: { fontSize: 16, marginTop: 20, color: "#d32f2f", textAlign: "center" },
});

export default Index;
