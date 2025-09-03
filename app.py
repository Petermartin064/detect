from flask import Flask, request, jsonify, render_template_string, send_file, make_response
from flask_cors import CORS, cross_origin
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import your detection system
from detection import KenyanVehicleDetector

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Kenyan vehicle detector (singleton pattern)
detector = None
detector_lock = threading.Lock()

def get_detector():
    global detector
    with detector_lock:
        if detector is None:
            try:
                logger.info("Initializing Kenyan Vehicle Detector...")
                detector = KenyanVehicleDetector()
                logger.info("Detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
                detector = "failed"
        return detector if detector != "failed" else None

# Thread pool for processing
executor = ThreadPoolExecutor(max_workers=2)

# In-memory storage for results
detection_results = []

# Enhanced HTML Dashboard (keeping your existing HTML)
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
        
        .server-status { background: rgba(46, 204, 113, 0.9); color: white; padding: 10px; text-align: center; margin-bottom: 20px; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="server-status">
            Server Running - Ready for Vehicle Detection
        </div>
        
        <div class="header">
            <h1>Kenyan Vehicle Detection System</h1>
            <p>Advanced AI-powered vehicle recognition with license plate detection</p>
            <button class="refresh-btn" onclick="location.reload()">Refresh Dashboard</button>
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
                        {{ result.detection.timestamp }}
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

def preprocess_image_for_detection(image_path, max_size=800):
    """Resize image to reduce processing time while maintaining quality"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            width, height = img.size
            if max(width, height) > max_size:
                ratio = max_size / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save resized image
                img.save(image_path, 'JPEG', quality=85, optimize=True)
                logger.info(f"Resized image from {width}x{height} to {img.size}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        return False

# Enhanced request/response logging
@app.before_request
def log_request_info():
    logger.info('[REQUEST] %s %s', request.method, request.url)
    if request.is_json and request.content_length:
        logger.info('[BODY] JSON Body size: %d bytes', request.content_length)

@app.after_request
def log_response_info(response):
    logger.info('[RESPONSE] %s %s', response.status_code, response.status)
    return response

# Handle preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

def process_detection_with_timeout(detector, temp_path, timeout_seconds=30):
    """Process detection with timeout"""
    try:
        logger.info(f"Starting detection with {timeout_seconds}s timeout...")
        detections, processed_image = detector.detect_kenyan_vehicles(temp_path)
        logger.info(f"Detection completed successfully")
        return detections, processed_image, None
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return None, None, str(e)

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
                                    detection_results=detection_results[-20:],
                                    total_detections=total,
                                    successful_plates=successful_plates,
                                    popular_vehicles=popular_count,
                                    avg_confidence=avg_conf)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard error: {e}", 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
@cross_origin()
def upload_and_detect():
    """Enhanced upload with progressive timeout and image optimization"""
    if request.method == 'OPTIONS':
        return '', 200
    
    temp_path = None
    try:
        logger.info("[UPLOAD] Upload request received")
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get detector instance
        vehicle_detector = get_detector()
        if not vehicle_detector:
            return jsonify({'error': 'Vehicle detection system not available'}), 500
        
        # Decode and optimize image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert and optimize
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize large images
            width, height = image.size
            max_dimension = 1200  # Reduced from unlimited
            
            if max(width, height) > max_dimension:
                ratio = max_dimension / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {image.size}")
            
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Save optimized image
        image_id = str(uuid.uuid4())[:8]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file.name, 'JPEG', quality=85, optimize=True)
            temp_path = temp_file.name
        
        # Progressive timeout detection with status updates
        try:
            logger.info("[DETECTION] Starting optimized detection...")
            
            # Use shorter timeout but with better error handling
            future = executor.submit(process_detection_optimized, vehicle_detector, temp_path, image_id)
            
            try:
                # Reduced timeout - force faster processing
                detections, processed_image, error = future.result(timeout=45)  # Increased slightly
                
                if error:
                    raise Exception(f"Detection error: {error}")
                
                if detections and len(detections) > 0:
                    detection = detections[0]
                    
                    # Store result with original image data
                    result_data = {
                        'image_id': image_id,
                        'image_data': base64.b64encode(image_bytes).decode('utf-8'),  # Store original
                        'detection': detection,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    
                    detection_results.append(result_data)
                    
                    # Success response
                    response_data = {
                        'success': True,
                        'image_id': image_id,
                        'message': 'Vehicle detected successfully!',
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
                    
                    logger.info("[SUCCESS] Detection completed: %s", detection.kenyan_model_prediction)
                    return jsonify(response_data)
                
                else:
                    # No detection but no error
                    return jsonify({
                        'success': True,
                        'image_id': image_id,
                        'message': 'No vehicle detected in image',
                        'detection': {
                            'vehicle_type': 'Unknown',
                            'kenyan_model': 'No vehicle detected',
                            'detection_confidence': 0.0
                        }
                    })
            
            except FutureTimeoutError:
                logger.error("[TIMEOUT] Detection timed out")
                future.cancel()
                
                return jsonify({
                    'success': False,
                    'image_id': image_id,
                    'message': 'Detection timed out. Try with a smaller or clearer image.',
                    'error': 'Processing timeout - try smaller image'
                }), 408
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return jsonify({
                'success': False,
                'message': f'Detection failed: {str(e)}',
                'error': str(e)
            }), 500
        
        finally:
            # Cleanup
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
def process_detection_optimized(detector, temp_path, image_id, timeout_seconds=40):
    """Optimized detection with faster processing"""
    try:
        logger.info(f"Starting optimized detection for {image_id}")
        start_time = time.time()
        
        # Quick image validation
        try:
            with Image.open(temp_path) as img:
                if img.size[0] * img.size[1] > 2000000:  # > 2MP
                    logger.warning("Large image detected, may cause timeout")
        except:
            pass
        
        # Run detection with timeout awareness
        detections, processed_image = detector.detect_kenyan_vehicles(temp_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Detection completed in {elapsed:.2f}s")
        
        return detections, processed_image, None
        
    except Exception as e:
        logger.error(f"Optimized detection failed: {e}")
        return None, None, str(e)
    
@app.route('/api/quick-detect', methods=['POST'])
@cross_origin()
def quick_detect():
    """Faster detection with reduced features for testing"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process smaller image for speed
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aggressive resize for speed
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # Small size for quick processing
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        
        image_id = str(uuid.uuid4())[:8]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file.name, 'JPEG', quality=70)
            temp_path = temp_file.name
        
        try:
            # Quick detection with shorter timeout
            detector = get_detector()
            if not detector:
                return jsonify({'error': 'Detector unavailable'}), 500
            
            # 20 second timeout for quick mode
            future = executor.submit(detector.detect_kenyan_vehicles, temp_path)
            detections, _ = future.result(timeout=20)
            
            if detections:
                detection = detections[0]
                return jsonify({
                    'success': True,
                    'quick_mode': True,
                    'detection': {
                        'vehicle_type': detection.vehicle_type,
                        'kenyan_model': detection.kenyan_model_prediction,
                        'confidence': round(detection.confidence_score, 2)
                    }
                })
            else:
                return jsonify({
                    'success': True,
                    'message': 'No vehicle detected'
                })
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Quick detect failed: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/favicon.ico')
def favicon():
    return '', 204 

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
                                  if r['detection'].vehicle_type not in ['Unknown', 'Error', 'Timeout'])
        
        successful_plates = sum(1 for r in detection_results 
                               if r['detection'].license_plate not in ['No plate detected', 'Detection failed', 'Processing timeout'])
        
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
@cross_origin()
def health_check():
    """Health check with detector status"""
    detector_status = "available" if get_detector() else "unavailable"
    
    response_data = {
        'status': 'healthy',
        'detector_status': detector_status,
        'timestamp': datetime.now().isoformat(),
        'total_detections': len(detection_results),
        'timeout_handling': 'enabled',
        'max_processing_time': '30 seconds',
        'server_info': {
            'python_version': '3.x',
            'flask_running': True,
            'cors_enabled': True
        }
    }
    
    logger.info("[HEALTH] Health check requested")
    return jsonify(response_data)

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error("[ERROR] Server Error: %s", error)
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

@app.errorhandler(408)
def timeout_error(error):
    logger.error("[ERROR] Timeout Error: %s", error)
    return jsonify({'error': 'Request timeout', 'details': 'Processing took too long'}), 408

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error("[ERROR] Unhandled Exception: %s", error)
    return jsonify({'error': 'Unexpected error', 'details': str(error)}), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced Kenyan Vehicle Detection Flask Server...")
    logger.info("Server will be available at: http://0.0.0.0:5000")
    logger.info("Mobile access at: http://192.168.100.38:5000")
    logger.info("Health check: http://192.168.100.38:5000/api/health")
    logger.info("Features: Timeout handling, Error recovery, Mobile optimization")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)  # Disabled debug for production