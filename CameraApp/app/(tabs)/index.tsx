import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Image,
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
  Dimensions,
  ScrollView,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';

const { width, height } = Dimensions.get('window');

// Replace with your actual server IP
const API_BASE_URL = 'http://192.168.100.38:5000/';

export default function CameraApp() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [photo, setPhoto] = useState<any>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [connectionTested, setConnectionTested] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const cameraRef = useRef<any>(null);

  // Enhanced image compression and optimization
  const optimizeImage = async (imageUri: string, quality: number = 0.7) => {
    try {
      console.log('Optimizing image...');
      const manipResult = await ImageManipulator.manipulateAsync(
        imageUri,
        [
          // Resize to maximum 1200px on longest side
          { resize: { width: 1200 } },
        ],
        {
          compress: quality,
          format: ImageManipulator.SaveFormat.JPEG,
          base64: true,
        }
      );
      
      console.log('Image optimized:', manipResult.width, 'x', manipResult.height);
      return manipResult;
    } catch (error) {
      console.error('Failed to optimize image:', error);
      throw error;
    }
  };

  // Test server connectivity with timeout
  const testConnection = async () => {
    try {
      console.log('Testing connection to:', `${API_BASE_URL}api/health`);
      setProcessingStatus('Testing connection...');
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // Shorter timeout

      const response = await fetch(`${API_BASE_URL}api/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      setProcessingStatus('');

      if (response.ok) {
        const result = await response.json();
        console.log('Health check result:', result);
        Alert.alert(
          'Connection Success! 🟢', 
          `Server is ready!\n` +
          `Status: ${result.status}\n` +
          `Detector: ${result.detector_status}\n` +
          `Detections: ${result.total_detections}`
        );
        setConnectionTested(true);
        return true;
      } else {
        Alert.alert('Connection Failed', `Server error: ${response.status}`);
        return false;
      }
    } catch (error: any) {
      console.error('Connection test failed:', error);
      setProcessingStatus('');
      
      let errorMessage = 'Cannot reach server';
      if (error.name === 'AbortError') {
        errorMessage = 'Connection timeout (8s)';
      } else if (error.message.includes('Network request failed')) {
        errorMessage = 'Network error. Check:\n• Server is running\n• IP address is correct\n• Same WiFi network';
      }
      
      Alert.alert('Connection Failed ❌', errorMessage);
      return false;
    }
  };

  // Enhanced picture taking with compression
  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        setProcessingStatus('Taking picture...');
        
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8, // Slightly reduced quality for faster processing
          skipProcessing: false,
        });
        
        // Optimize the image immediately
        const optimized = await optimizeImage(photo.uri, 0.7);
        
        setPhoto({
          ...optimized,
          originalUri: photo.uri
        });
        
        setProcessingStatus('');
        console.log('Picture taken and optimized');
      } catch (error) {
        console.error('Failed to take picture:', error);
        setProcessingStatus('');
        Alert.alert('Error', 'Failed to take picture');
      }
    }
  };

  // Enhanced image picker with compression
  const pickImage = async () => {
    try {
      setProcessingStatus('Selecting image...');
      
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        const selected = result.assets[0];
        
        // Optimize the selected image
        const optimized = await optimizeImage(selected.uri, 0.7);
        
        setPhoto({
          ...optimized,
          originalUri: selected.uri
        });
        
        console.log('Image picked and optimized');
      }
      
      setProcessingStatus('');
    } catch (error) {
      console.error('Failed to pick image:', error);
      setProcessingStatus('');
      Alert.alert('Error', 'Failed to pick image from gallery');
    }
  };

  // Try quick detection first, then full detection
  const uploadImage = async () => {
    if (!photo || !photo.base64) {
      Alert.alert('Error', 'No image to upload');
      return;
    }

    if (!connectionTested) {
      Alert.alert(
        'Test Connection First',
        'Recommend testing connection first for best results',
        [
          { text: 'Test Connection', onPress: testConnection },
          { text: 'Upload Anyway', onPress: () => performUpload() },
          { text: 'Cancel', style: 'cancel' }
        ]
      );
      return;
    }

    // Try quick detection first
    Alert.alert(
      'Detection Mode',
      'Choose detection speed:',
      [
        { 
          text: 'Quick (20s)', 
          onPress: () => performUpload(true) 
        },
        { 
          text: 'Full Analysis (60s)', 
          onPress: () => performUpload(false) 
        },
        { 
          text: 'Cancel', 
          style: 'cancel' 
        }
      ]
    );
  };

  const performUpload = async (quickMode: boolean = false) => {
    setUploading(true);
    setUploadResult(null);
    setProcessingStatus('Preparing upload...');

    try {
      const endpoint = quickMode ? 'api/quick-detect' : 'api/upload';
      const timeout = quickMode ? 25000 : 70000; // Longer timeout for full analysis
      
      console.log(`Uploading to: ${API_BASE_URL}${endpoint}`);
      console.log('Image size (base64):', photo.base64.length, 'characters');
      
      setProcessingStatus(quickMode ? 'Quick analysis...' : 'Full analysis...');

      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        setProcessingStatus('Upload timed out');
      }, timeout);

      const requestData = {
        image: `data:image/jpeg;base64,${photo.base64}`,
      };

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      setProcessingStatus('Processing results...');

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log('Error response:', errorText);
        
        if (response.status === 408) {
          // Timeout - suggest quick mode
          Alert.alert(
            'Processing Timeout',
            'The image took too long to process. Try:\n• Quick mode for faster results\n• Smaller/clearer image\n• Better lighting',
            [
              { text: 'Try Quick Mode', onPress: () => performUpload(true) },
              { text: 'OK', style: 'cancel' }
            ]
          );
        } else {
          Alert.alert('Upload Error', `Server error: ${response.status}\n${errorText}`);
        }
        return;
      }

      const result = await response.json();
      console.log('Success result:', result);

      if (result.success) {
        setUploadResult(result);
        
        const detection = result.detection;
        let alertMessage = quickMode ? 
          '🚀 Quick Analysis Complete!\n\n' : 
          '🔬 Full Analysis Complete!\n\n';
        
        if (detection && detection.kenyan_model !== 'No vehicle detected') {
          alertMessage += `🚗 Model: ${detection.kenyan_model}\n`;
          alertMessage += `🎯 Confidence: ${(detection.model_confidence * 100).toFixed(1)}%\n`;
          
          if (!quickMode) {
            alertMessage += `🎨 Color: ${detection.color}\n`;
            alertMessage += `📋 Category: ${detection.market_category || 'Unknown'}\n`;
            
            if (detection.license_plate && detection.license_plate !== 'No plate detected') {
              alertMessage += `🔢 License: ${detection.license_plate}`;
            } else {
              alertMessage += `🔢 License: Not detected`;
            }
          }
        } else {
          alertMessage += '❌ No vehicle detected in image';
        }
        
        Alert.alert('Success! 🎉', alertMessage);
      } else {
        Alert.alert('Analysis Failed', result.error || result.message || 'Unknown error');
      }
    } catch (error: any) {
      console.error('❌ Upload error:', error);
      
      let errorMessage = 'Upload failed';
      if (error.name === 'AbortError') {
        errorMessage = `Upload timed out (${quickMode ? '25' : '70'}s).\n\nTry:\n• Quick mode\n• Smaller image\n• Better connection`;
      } else if (error.message.includes('Network request failed')) {
        errorMessage = 'Network error. Check connection and try again.';
      } else {
        errorMessage = `Error: ${error.message}`;
      }
      
      Alert.alert('Upload Error ❌', errorMessage);
    } finally {
      setUploading(false);
      setProcessingStatus('');
    }
  };

  const retakePhoto = () => {
    setPhoto(null);
    setUploadResult(null);
    setProcessingStatus('');
    console.log('Photo cleared for retake');
  };

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  // Show processing status
  const renderProcessingStatus = () => {
    if (processingStatus) {
      return (
        <View style={styles.processingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.processingText}>{processingStatus}</Text>
        </View>
      );
    }
    return null;
  };

  if (!permission) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.title}>🚗 Vehicle Detection App</Text>
        <Text style={styles.errorText}>Camera access required for vehicle detection</Text>
        
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>📷 Grant Camera Permission</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={pickImage}>
          <Text style={styles.buttonText}>🖼️ Pick from Gallery</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.button, styles.testButton]} onPress={testConnection}>
          <Text style={styles.buttonText}>🔍 Test Server Connection</Text>
        </TouchableOpacity>
        
        {renderProcessingStatus()}
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {!photo ? (
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
          />
          
          <View style={styles.cameraOverlay}>
            <View style={styles.topControls}>
              <TouchableOpacity
                style={styles.testConnectionButton}
                onPress={testConnection}
              >
                <Text style={styles.smallButtonText}>
                  {connectionTested ? '🟢 Connected' : '🔍 Test'}
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.flipButton}
                onPress={toggleCameraFacing}
              >
                <Text style={styles.flipText}>🔄 Flip</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.centerOverlay}>
              <View style={styles.targetFrame}>
                <Text style={styles.instructionText}>
                  📷 Point at vehicle{'\n'}(Good lighting helps)
                </Text>
              </View>
            </View>

            <View style={styles.bottomControls}>
              <TouchableOpacity style={styles.galleryButton} onPress={pickImage}>
                <Text style={styles.controlText}>🖼️ Gallery</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.captureButton} 
                onPress={takePicture}
                disabled={!!processingStatus}
              >
                <View style={[
                  styles.captureButtonInner,
                  processingStatus && styles.disabledCapture
                ]} />
              </TouchableOpacity>
              
              <View style={styles.placeholder} />
            </View>
            
            {renderProcessingStatus()}
          </View>
        </View>
      ) : (
        <ScrollView style={styles.previewContainer}>
          <Image source={{ uri: photo.uri }} style={styles.preview} />
          
          <View style={styles.previewControls}>
            <TouchableOpacity style={styles.retakeButton} onPress={retakePhoto}>
              <Text style={styles.buttonText}>🔄 Retake</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.uploadButton, (uploading || processingStatus) && styles.disabledButton]} 
              onPress={uploadImage}
              disabled={uploading || !!processingStatus}
            >
              {uploading ? (
                <View style={styles.uploadingContainer}>
                  <ActivityIndicator color="white" size="small" />
                  <Text style={styles.uploadingText}>Analyzing...</Text>
                </View>
              ) : (
                <Text style={styles.buttonText}>🚗 Detect Vehicle</Text>
              )}
            </TouchableOpacity>
          </View>

          {renderProcessingStatus()}

          {uploadResult && (
            <View style={styles.resultContainer}>
              <Text style={styles.resultTitle}>
                {uploadResult.success ? '✅ Analysis Complete!' : '❌ Analysis Failed'}
              </Text>
              
              {uploadResult.quick_mode && (
                <Text style={styles.quickModeLabel}>🚀 Quick Mode Results</Text>
              )}
              
              {uploadResult.success && uploadResult.detection && (
                <View style={styles.detectionResults}>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>🆔 ID:</Text>
                    <Text style={styles.resultValue}>{uploadResult.image_id}</Text>
                  </View>
                  
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>🚗 Model:</Text>
                    <Text style={styles.resultValue}>
                      {uploadResult.detection.kenyan_model || 'Unknown'}
                    </Text>
                  </View>
                  
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>🎯 Confidence:</Text>
                    <Text style={styles.resultValue}>
                      {uploadResult.detection.model_confidence ? 
                        `${(uploadResult.detection.model_confidence * 100).toFixed(1)}%` : 
                        uploadResult.detection.confidence ?
                        `${(uploadResult.detection.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </Text>
                  </View>
                  
                  {!uploadResult.quick_mode && (
                    <>
                      <View style={styles.resultRow}>
                        <Text style={styles.resultLabel}>🎨 Color:</Text>
                        <Text style={styles.resultValue}>
                          {uploadResult.detection.color || 'Unknown'}
                        </Text>
                      </View>
                      
                      <View style={styles.resultRow}>
                        <Text style={styles.resultLabel}>🔢 License:</Text>
                        <Text style={styles.resultValue}>
                          {uploadResult.detection.license_plate || 'Not detected'}
                        </Text>
                      </View>
                      
                      <View style={styles.resultRow}>
                        <Text style={styles.resultLabel}>📋 Category:</Text>
                        <Text style={styles.resultValue}>
                          {uploadResult.detection.market_category || 'Unknown'}
                        </Text>
                      </View>
                    </>
                  )}
                </View>
              )}
              
              {uploadResult.message && (
                <Text style={styles.resultMessage}>{uploadResult.message}</Text>
              )}
            </View>
          )}
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  // ... keeping all your existing styles ...
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#666',
  },
  
  // Processing status styles
  processingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    backgroundColor: 'rgba(0,0,0,0.8)',
    borderRadius: 10,
    margin: 20,
  },
  processingText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  
  // Camera styles
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'transparent',
    flexDirection: 'column',
    justifyContent: 'space-between',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 40,
  },
  testConnectionButton: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 15,
  },
  smallButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  flipButton: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  flipText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  centerOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  targetFrame: {
    width: width * 0.8,
    height: width * 0.6,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.7)',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.1)',
  },
  instructionText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 50,
    paddingHorizontal: 20,
  },
  galleryButton: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 25,
  },
  controlText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'white',
  },
  disabledCapture: {
    backgroundColor: '#ccc',
  },
  placeholder: {
    width: 60,
  },
  
  // Preview styles
  previewContainer: {
    flex: 1,
    backgroundColor: 'black',
  },
  preview: {
    width: width,
    height: height * 0.6,
    resizeMode: 'contain',
  },
  previewControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
    backgroundColor: 'rgba(0,0,0,0.9)',
  },
  
  // Button styles
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    marginVertical: 10,
    minWidth: 200,
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: '#FF9500',
  },
  testButton: {
    backgroundColor: '#34C759',
  },
  retakeButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  uploadButton: {
    backgroundColor: '#34C759',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    minWidth: 140,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#999',
  },
  uploadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  uploadingText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  
  // Result styles
  resultContainer: {
    backgroundColor: 'rgba(0,0,0,0.9)',
    padding: 20,
    margin: 20,
    borderRadius: 15,
  },
  resultTitle: {
    color: '#34C759',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  quickModeLabel: {
    color: '#FF9500',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10,
  },
  detectionResults: {
    marginTop: 10,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  resultLabel: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
    flex: 1,
  },
  resultValue: {
    color: '#FFFFFF',
    fontSize: 16,
    flex: 2,
    textAlign: 'right',
  },
  resultMessage: {
    color: '#FFFFFF',
    fontSize: 14,
    marginTop: 15,
    textAlign: 'center',
    fontStyle: 'italic',
  },
});