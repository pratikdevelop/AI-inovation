import React, { useState, useEffect } from 'react';
import { View, Text, Button, TextInput, StyleSheet, ScrollView, ActivityIndicator, PermissionsAndroid, Platform } from 'react-native';
import axios from 'axios';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';

const Index = () => {
  const [features, setFeatures] = useState<any>(Array(11).fill(''));
  const [prediction, setPrediction] = useState<any>(null);
  const [pronunciationScore, setPronunciationScore] = useState<any>(null);
  const [transcript, setTranscript] = useState<any>(null);
  const [error, setError] = useState<any>(null);
  const [loading, setLoading] = useState<any>(false);
  const [recording, setRecording] = useState<any>(false);
  const [audioPath, setAudioPath] = useState<any>(null);

  const audioRecorderPlayer = new AudioRecorderPlayer(); // Instantiate here
  const baseUrl = 'http://localhost:8000'; // Replace with your IP or deployed URL

  useEffect(() => {
    requestMicPermission();
    return () => {
      audioRecorderPlayer.removeRecordBackListener();
    };
  }, []);

  const requestMicPermission = async () => {
    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title: 'Microphone Permission',
          message: 'App needs access to your microphone for pronunciation feedback.',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        }
      );
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
        setError('Microphone permission denied');
      }
    }
  };

  const predictDifficulty = async () => {
    setLoading(true);
    setError(null);
    try {
      const userData = {
        features: features.map((f: string) => parseFloat(f) || 0),
      };
      if (userData.features.some((f: number) => isNaN(f) || f < 0 || f > 1)) {
        throw new Error('All inputs must be numbers between 0 and 1');
      }
      const response = await axios.post(`${baseUrl}/predict`, userData, { timeout: 5000 });
      setPrediction(response.data.difficulty);
    } catch (err:any) {
      setError(err.message || 'Failed to get prediction');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const path = Platform.OS === 'android' ? 'sdcard/recording.wav' : 'recording.wav';
      const result = await audioRecorderPlayer.startRecorder(path);
      setRecording(true);
      setAudioPath(result);
      console.log('Recording started at:', result);
    } catch (err:any) {
      setError('Failed to start recording: ' + err.message);
    }
  };

  const stopRecording = async () => {
    try {
      const result = await audioRecorderPlayer.stopRecorder();
      setRecording(false);
      console.log('Recording stopped at:', result);
      await sendAudioForPronunciation(result);
    } catch (err:any) {
      setError('Failed to stop recording: ' + err.message);
      setRecording(false);
    }
  };

  const sendAudioForPronunciation = async (filePath: string) => {
    setLoading(true);
    setError(null);
    try {
      const formData:any = new FormData();
      formData.append('audio_file', {
        uri: Platform.OS === 'android' ? `file://${filePath}` : filePath,
        type: 'audio/wav',
        name: 'recording.wav',
      });

      const response = await axios.post(`${baseUrl}/pronunciation`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 10000,
      });
      setPronunciationScore(response.data.score);
      setTranscript(response.data.transcript);
    } catch (err:any) {
      setError(err.response?.data?.detail || 'Failed to analyze pronunciation: ' + err.message);
      setPronunciationScore(null);
      setTranscript(null);
    } finally {
      setLoading(false);
    }
  };

  const labels = [
    'Vocabulary', 'Grammar', 'Listening', 'Speaking', 'Pronunciation',
    'Time Spent', 'Quiz Rate', 'Error Freq', 'Confidence', 'Completion Rate', 'Cultural Quiz'
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>AI Language Learning Companion</Text>
      
      {/* Difficulty Prediction Inputs */}
      {labels.map((label, index) => (
        <View key={index} style={styles.inputContainer}>
          <Text style={styles.label}>{label} (0-1):</Text>
          <TextInput
            style={styles.input}
            keyboardType="numeric"
            placeholder="0.0"
            value={features[index]}
            onChangeText={text => {
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
        disabled={loading || recording}
      />
      {prediction !== null && (
        <Text style={styles.result}>
          Predicted Difficulty: {['Beginner', 'Intermediate', 'Advanced'][prediction]}
        </Text>
      )}

      {/* Pronunciation Feedback */}
      <View style={styles.section}>
        <Text style={styles.subtitle}>Pronunciation Feedback</Text>
        <Button 
          title={recording ? "Stop Recording" : "Start Recording"} 
          onPress={recording ? stopRecording : startRecording}
          disabled={loading}
        />
        {pronunciationScore !== null && (
          <Text style={styles.result}>
            Pronunciation Score: {(pronunciationScore * 100).toFixed(1)}%{'\n'}
            Transcript: {transcript}
          </Text>
        )}
      </View>

      {loading && <ActivityIndicator size="large" color="#0000ff" style={styles.loader} />}
      {error && <Text style={styles.error}>{error}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { padding: 20, backgroundColor: '#f5f5f5' },
  title: { fontSize: 24, textAlign: 'center', marginBottom: 20, fontWeight: 'bold' },
  section: { marginTop: 30 },
  subtitle: { fontSize: 20, marginBottom: 10, textAlign: 'center' },
  inputContainer: { flexDirection: 'row', alignItems: 'center', marginBottom: 15 },
  label: { flex: 1, fontSize: 16 },
  input: { borderWidth: 1, borderColor: '#ccc', padding: 8, width: 80, borderRadius: 5 },
  loader: { marginTop: 20 },
  result: { fontSize: 18, marginTop: 20, color: '#2e7d32', textAlign: 'center' },
  error: { fontSize: 16, marginTop: 20, color: '#d32f2f', textAlign: 'center' },
});

export default Index;