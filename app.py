from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import json
from datetime import datetime
import uuid
import numpy as np
import cv2
import tempfile
import logging

# Import your detection system
from detection import KenyanVehicleDetector

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Kenyan vehicle detector (singleton pattern)
detector = None

def get_detector():
    global detector
    if detector is None:
        try:
            logger.info("Initializing Kenyan Vehicle Detector...")
            detector = KenyanVehicleDetector()
            logger.info("Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            detector = "failed"
    return detector if detector != "failed" else None

# In-memory storage for results
detection_results = []

# Enhanced HTML Dashboard with vehicle detection results
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kenyan Vehicle Detection Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
        .header h1 { color: #2c3e50; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
        .header p { text-align: center; color: #7f8c8d; font-size: 1.1em; }
        
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: rgba(255,255,255,0.9); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-number { font-size: 2.5em; font-weight: bold; color: #3498db; margin-bottom: 5px; }
        .stat-label { color: #7f8c8d; font-size: 1.1em; }
        
        .detection-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 25px; }
        .detection-card { background: rgba(255,255,255,0.95); border-radius: 15px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease; }
        .detection-card:hover { transform: translateY(-5px); }
        
        .detection-image { position: relative; }
        .detection-image img { width: 100%; height: 250px; object-fit: cover; }
        .confidence-badge { position: absolute; top: 10px; right: 10px; background: rgba(46, 204, 113, 0.9); color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; font-weight: bold; }
        
        .detection-info { padding: 20px; }
        .vehicle-type { font-size: 1.3em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .kenyan-model { font-size: 1.1em; color: #3498db; margin-bottom: 8px; }
        .market-category { background: #f39c12; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; display: inline-block; margin-bottom: 10px; }
        
        .detection-details { margin-top: 15px; }
        .detail-row { display: flex; justify-content: space-between; margin-bottom: 8px; padding: 8px; background: #f8f9fa; border-radius: 5px; }
        .detail-label { font-weight: bold; color: #5d6d7e; }
        .detail-value { color: #2c3e50; }
        
        .license-plate { background: #2ecc71; color: white; padding: 6px 12px; border-radius: 8px; font-family: monospace; font-weight: bold; text-align: center; margin: 10px 0; }
        .plate-failed { background: #e74c3c; }
        
        .timestamp { color: #95a5a6; font-size: 0.9em; text-align: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #ecf0f1; }
        
        .refresh-btn { background: linear-gradient(45deg, #3498db, #2980b9); color: white; border: none; padding: 12px 25px; border-radius: 25px; cursor: pointer; font-size: 1.1em; transition: all 0.3s ease; }
        .refresh-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); }
        
        .no-detections { text-align: center; padding: 60px 20px; background: rgba(255,255,255,0.9); border-radius: 15px; backdrop-filter: blur(10px); }
        .no-detections h3 { color: #7f8c8d; margin-bottom: 15px; font-size: 1.5em; }
        .no-detections p { color: #95a5a6; font-size: 1.1em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 Kenyan Vehicle Detection System</h1>
            <p>Advanced AI-powered vehicle recognition with license plate detection</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Dashboard</button>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{{ total_detections }}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ successful_plates }}</div>
                <div class="stat-label">License Plates Found</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ popular_vehicles }}</div>
                <div class="stat-label">Popular Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ avg_confidence }}%</div>
                <div class="stat-label">Average Confidence</div>
            </div>
        </div>
        
        <div class="detection-grid">
            {% for result in detection_results %}
            <div class="detection-card">
                <div class="detection-image">
                    <img src="/api/image/{{ result.image_id }}" alt="Detection result">
                    <div class="confidence-badge">{{ "%.1f"|format(result.detection.detection_confidence * 100) }}%</div>
                </div>
                
                <div class="detection-info">
                    <div class="vehicle-type">{{ result.detection.vehicle_type.title() }}</div>
                    <div class="kenyan-model">{{ result.detection.kenyan_model_prediction }}</div>
                    <div class="market-category">{{ result.detection.market_category }}</div>
                    
                    {% if result.detection.license_plate not in ['No plate detected', 'Detection failed'] %}
                    <div class="license-plate">{{ result.detection.license_plate }}</div>
                    {% else %}
                    <div class="license-plate plate-failed">Plate Not Detected</div>
                    {% endif %}
                    
                    <div class="detection-details">
                        <div class="detail-row">
                            <span class="detail-label">Color:</span>
                            <span class="detail-value">{{ result.detection.color }}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Model Confidence:</span>
                            <span class="detail-value">{{ "%.1f"|format(result.detection.confidence_score * 100) }}%</span>
                        </div>
                        {% if result.detection.heavy_load %}
                        <div class="detail-row">
                            <span class="detail-label">Load Status:</span>
                            <span class="detail-value">Heavy Load ({{ "%.1f"|format(result.detection.load_confidence * 100) }}%)</span>
                        </div>
                        {% endif %}
                    </div>
                    
                    <div class="timestamp">
                        📅 {{ result.detection.timestamp }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if not detection_results %}
        <div class="no-detections">
            <h3>No Vehicle Detections Yet</h3>
            <p>Upload vehicle images from your mobile app to see AI-powered analysis results here!</p>
        </div>
        {% endif %}
    </div>
    
    <script>
        // Auto-refresh every 60 seconds
        setTimeout(() => location.reload(), 60000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Enhanced dashboard with detection statistics"""
    try:
        total = len(detection_results)
        successful_plates = sum(1 for r in detection_results 
                               if r['detection'].license_plate not in ['No plate detected', 'Detection failed'])
        
        popular_count = sum(1 for r in detection_results 
                           if 'Popular' in r['detection'].market_category or 
                              'Leader' in r['detection'].market_category)
        
        avg_conf = 0
        if detection_results:
            avg_conf = sum(r['detection'].detection_confidence for r in detection_results) / len(detection_results)
            avg_conf = int(avg_conf * 100)
        
        return render_template_string(DASHBOARD_HTML,
                                    detection_results=detection_results[-20:],  # Show last 20
                                    total_detections=total,
                                    successful_plates=successful_plates,
                                    popular_vehicles=popular_count,
                                    avg_confidence=avg_conf)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard error: {e}", 500

@app.route('/api/upload', methods=['POST'])
def upload_and_detect():
    """Enhanced upload endpoint with vehicle detection"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get detector instance
        vehicle_detector = get_detector()
        if not vehicle_detector:
            return jsonify({'error': 'Vehicle detection system not available'}), 500
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
                
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Save temporary file for detection
        image_id = str(uuid.uuid4())[:8]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file.name, 'JPEG', quality=95)
            temp_path = temp_file.name
        
        try:
            # Run Kenyan vehicle detection
            logger.info("Running vehicle detection...")
            detections, processed_image = vehicle_detector.detect_kenyan_vehicles(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if detections and len(detections) > 0:
                # Use the best detection
                detection = detections[0]
                
                # Store the result
                result_data = {
                    'image_id': image_id,
                    'image_data': image_data,
                    'detection': detection,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_time': 'N/A'
                }
                
                detection_results.append(result_data)
                
                # Prepare response
                response_data = {
                    'success': True,
                    'image_id': image_id,
                    'message': 'Vehicle detected and analyzed successfully!',
                    'detection': {
                        'vehicle_type': detection.vehicle_type,
                        'kenyan_model': detection.kenyan_model_prediction,
                        'model_confidence': round(detection.confidence_score, 3),
                        'color': detection.color,
                        'license_plate': detection.license_plate,
                        'plate_confidence': round(detection.plate_confidence, 3),
                        'detection_confidence': round(detection.detection_confidence, 3),
                        'market_category': detection.market_category,
                        'heavy_load': detection.heavy_load
                    }
                }
                
                logger.info(f"Detection successful: {detection.kenyan_model_prediction}")
                return jsonify(response_data)
                
            else:
                # No vehicle detected
                basic_info = {
                    'image_id': image_id,
                    'image_data': image_data,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                
                # Store as basic upload without detection
                detection_results.append({
                    **basic_info,
                    'detection': type('obj', (object,), {
                        'vehicle_type': 'Unknown',
                        'kenyan_model_prediction': 'No vehicle detected',
                        'confidence_score': 0.0,
                        'color': 'Unknown',
                        'license_plate': 'No plate detected',
                        'plate_confidence': 0.0,
                        'detection_confidence': 0.0,
                        'market_category': 'Unknown',
                        'heavy_load': False,
                        'timestamp': datetime.now().isoformat()
                    })()
                })
                
                return jsonify({
                    'success': True,
                    'image_id': image_id,
                    'message': 'Image uploaded but no vehicle detected',
                    'detection': {
                        'vehicle_type': 'Unknown',
                        'kenyan_model': 'No vehicle detected',
                        'detection_confidence': 0.0
                    }
                })
        
        except Exception as detection_error:
            logger.error(f"Detection failed: {detection_error}")
            
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Store as failed detection
            detection_results.append({
                'image_id': image_id,
                'image_data': image_data,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'detection': type('obj', (object,), {
                    'vehicle_type': 'Error',
                    'kenyan_model_prediction': 'Detection failed',
                    'confidence_score': 0.0,
                    'color': 'Unknown',
                    'license_plate': 'Detection failed',
                    'plate_confidence': 0.0,
                    'detection_confidence': 0.0,
                    'market_category': 'Error',
                    'heavy_load': False,
                    'timestamp': datetime.now().isoformat()
                })()
            })
            
            return jsonify({
                'success': False,
                'image_id': image_id,
                'message': f'Vehicle detection failed: {str(detection_error)}',
                'error': str(detection_error)
            })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/image/<image_id>')
def get_image(image_id):
    """Serve individual images"""
    try:
        for result in detection_results:
            if result['image_id'] == image_id:
                image_data = base64.b64decode(result['image_data'])
                return send_file(
                    io.BytesIO(image_data),
                    mimetype='image/jpeg',
                    as_attachment=False
                )
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections')
def list_detections():
    """API endpoint to get all detection results"""
    try:
        results = []
        for result in detection_results:
            detection_data = {
                'image_id': result['image_id'],
                'timestamp': result['timestamp'],
                'vehicle_type': result['detection'].vehicle_type,
                'kenyan_model': result['detection'].kenyan_model_prediction,
                'confidence': result['detection'].detection_confidence,
                'color': result['detection'].color,
                'license_plate': result['detection'].license_plate,
                'market_category': result['detection'].market_category
            }
            results.append(detection_data)
        
        return jsonify({
            'detections': results,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Enhanced statistics with detection data"""
    try:
        total = len(detection_results)
        successful_detections = sum(1 for r in detection_results 
                                  if r['detection'].vehicle_type not in ['Unknown', 'Error'])
        
        successful_plates = sum(1 for r in detection_results 
                               if r['detection'].license_plate not in ['No plate detected', 'Detection failed'])
        
        avg_confidence = 0
        if detection_results:
            confidences = [r['detection'].detection_confidence for r in detection_results 
                          if r['detection'].detection_confidence > 0]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        return jsonify({
            'total_uploads': total,
            'successful_detections': successful_detections,
            'successful_plates': successful_plates,
            'detection_rate': successful_detections / total if total > 0 else 0,
            'plate_detection_rate': successful_plates / total if total > 0 else 0,
            'average_confidence': avg_confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check with detector status"""
    detector_status = "available" if get_detector() else "unavailable"
    
    return jsonify({
        'status': 'healthy',
        'detector_status': detector_status,
        'timestamp': datetime.now().isoformat(),
        'total_detections': len(detection_results)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

"""
DEPLOYMENT NOTES:
1. Make sure detection.py is in the same directory as app.py
2. The system will automatically download YOLO weights on first run
3. OCR models will be downloaded automatically
4. Consider using lighter models for Render's 512MB RAM limit
5. Cold starts may take 30+ seconds due to model loading
"""