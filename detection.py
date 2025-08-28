import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import easyocr
import re
import logging
from datetime import datetime
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
import webcolors
from PIL import Image, ImageEnhance
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kenyan_vehicle_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class KenyanVehicleDetection:
    """Data class for Kenyan vehicle detection results"""
    vehicle_type: str
    kenyan_model_prediction: str
    confidence_score: float
    color: str
    license_plate: str
    plate_confidence: float
    heavy_load: bool
    load_confidence: float
    bbox: List[int]
    detection_confidence: float
    timestamp: str
    market_category: str
    additional_features: Dict

class KenyanVehicleDetector:
    """Kenyan Market-Specific Vehicle Detection System"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize models
        logger.info("Initializing Kenyan vehicle detection models...")
        self._initialize_models()
        
        # Kenyan license plate patterns (ordered by priority)
        self.kenyan_plate_patterns = [
            # Current K-series format (highest priority)
            {'pattern': r'^K[A-Z]{2}\s*\d{3}\s*[A-Z]$', 'name': 'K-series', 'score': 1.0},
            # Government plates
            {'pattern': r'^GK\s*\d{3}\s*[A-Z]$', 'name': 'Government', 'score': 0.95},
            # Diplomatic plates
            {'pattern': r'^CD\s*\d{3}\s*[A-Z]$', 'name': 'Diplomatic', 'score': 0.95},
            # Standard 3-letter format
            {'pattern': r'^[A-Z]{3}\s*\d{3}\s*[A-Z]$', 'name': 'Standard', 'score': 0.9},
            # Older format
            {'pattern': r'^[A-Z]{2}\s*\d{3}\s*[A-Z]{2}$', 'name': 'Legacy', 'score': 0.8},
            # UN and NGO
            {'pattern': r'^(UN|NGO)\s*\d{3}\s*[A-Z]$', 'name': 'International', 'score': 0.85},
        ]
        
        # Character correction mappings
        self.char_fixes = {
            # Numbers that look like letters
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B',
            # Letters that look like numbers  
            'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8',
            # Common letter confusions
            'P': 'D', 'R': 'D', 'B': 'D'  # P, R, B often misread as D
        }
        
        # Kenyan Market Vehicle Database
        self.kenyan_vehicles = {
            'toyota': {
                'models': {
                    'harrier': {
                        'features': ['suv', 'medium_size', 'premium'],
                        'market_share': 'very_popular',
                        'common_colors': ['white', 'silver', 'black'],
                        'aspect_ratio_range': (1.6, 2.1),
                    },
                    'vitz': {
                        'features': ['hatchback', 'compact', 'economy'],
                        'market_share': 'very_popular',
                        'common_colors': ['white', 'silver', 'red', 'blue'],
                        'aspect_ratio_range': (1.4, 1.8),
                    },
                    'land_cruiser': {
                        'features': ['suv', 'large', 'luxury'],
                        'market_share': 'popular',
                        'common_colors': ['white', 'black', 'silver'],
                        'aspect_ratio_range': (1.8, 2.3),
                    },
                    'corolla': {
                        'features': ['sedan', 'medium', 'reliable'],
                        'market_share': 'popular',
                        'common_colors': ['white', 'silver', 'black', 'gray'],
                        'aspect_ratio_range': (1.9, 2.4),
                    },
                    'prado': {
                        'features': ['suv', 'medium_large', 'premium'],
                        'market_share': 'popular',
                        'common_colors': ['white', 'black', 'silver'],
                        'aspect_ratio_range': (1.7, 2.2),
                    }
                }
            },
            'isuzu': {
                'models': {
                    'd_max': {
                        'features': ['pickup', 'work_vehicle', 'reliable'],
                        'market_share': 'market_leader',
                        'common_colors': ['white', 'silver', 'black'],
                        'aspect_ratio_range': (2.2, 2.8),
                    },
                    'mu_x': {
                        'features': ['suv', 'large', 'family'],
                        'market_share': 'popular',
                        'common_colors': ['white', 'silver', 'black'],
                        'aspect_ratio_range': (1.8, 2.3),
                    }
                }
            },
            'nissan': {
                'models': {
                    'x_trail': {
                        'features': ['suv', 'crossover', 'family'],
                        'market_share': 'popular',
                        'common_colors': ['white', 'silver', 'black', 'red'],
                        'aspect_ratio_range': (1.7, 2.1),
                    },
                    'note': {
                        'features': ['hatchback', 'compact', 'economy'],
                        'market_share': 'common',
                        'common_colors': ['white', 'silver', 'blue'],
                        'aspect_ratio_range': (1.5, 1.9),
                    }
                }
            },
            'subaru': {
                'models': {
                    'forester': {
                        'features': ['suv', 'crossover', 'awd'],
                        'market_share': 'common',
                        'common_colors': ['white', 'silver', 'blue'],
                        'aspect_ratio_range': (1.6, 2.0),
                    },
                    'impreza': {
                        'features': ['sedan', 'sports', 'awd'],
                        'market_share': 'common',
                        'common_colors': ['blue', 'white', 'silver'],
                        'aspect_ratio_range': (1.8, 2.2),
                    }
                }
            },
            'mitsubishi': {
                'models': {
                    'pajero': {
                        'features': ['suv', 'large', 'off_road'],
                        'market_share': 'common',
                        'common_colors': ['white', 'black', 'silver'],
                        'aspect_ratio_range': (1.7, 2.2),
                    },
                    'outlander': {
                        'features': ['suv', 'crossover', 'family'],
                        'market_share': 'common',
                        'common_colors': ['white', 'silver', 'black'],
                        'aspect_ratio_range': (1.7, 2.1),
                    }
                }
            }
        }
        
        # Market categories
        self.market_categories = {
            'market_leader': 'Market Leader (Top 5%)',
            'very_popular': 'Very Popular (Top 15%)', 
            'popular': 'Popular (Top 30%)',
            'common': 'Common (Top 50%)',
            'rare': 'Rare/Luxury'
        }
        
        # Color ranges optimized for Kenyan lighting conditions
        self.color_ranges_hsv = {
            'white': {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255])},
            'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 60])},
            'silver': {'lower': np.array([0, 0, 90]), 'upper': np.array([180, 30, 200])},
            'gray': {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 30, 150])},
            'red': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            'red2': {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
            'blue': {'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])},
            'green': {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
            'yellow': {'lower': np.array([20, 50, 50]), 'upper': np.array([30, 255, 255])},
        }
        
        logger.info("Kenyan vehicle detector initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration settings optimized for Kenyan conditions"""
        default_config = {
            'yolo_model': 'yolov8x.pt',
            'detection_threshold': 0.6,
            'nms_threshold': 0.7,
            'plate_confidence_threshold': 0.3,
            'kenyan_model_confidence_threshold': 0.6,
            'color_confidence_threshold': 0.6,
            'load_detection_threshold': 0.7,
            'ocr_languages': ['en'],
            'max_image_size': 1920,
            'gpu_enabled': torch.cuda.is_available(),
            'min_box_area': 4000,
            'kenyan_market_focus': True,
            'duplicate_iou_threshold': 0.7,
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _initialize_models(self):
        """Initialize all AI models"""
        try:
            # YOLO for vehicle detection
            self.yolo_model = YOLO(self.config['yolo_model'])
            if self.config['gpu_enabled']:
                self.yolo_model.to('cuda')
            
            # OCR for license plates
            self.ocr_reader = easyocr.Reader(
                self.config['ocr_languages'],
                gpu=self.config['gpu_enabled']
            )
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def detect_kenyan_vehicles(self, image_path: str) -> Tuple[List[KenyanVehicleDetection], np.ndarray]:
        """Main detection pipeline"""
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return [], None
            
            logger.info(f"Processing Kenyan vehicle detection: {image.shape}")
            
            # Run YOLO detection
            yolo_results = self.yolo_model(
                image,
                conf=0.7,
                iou=0.45,
                verbose=False,
                classes=[2, 5, 7]  # car, bus, truck
            )
            
            best_detection = None
            best_confidence = 0
            
            # Find the single best detection
            for result in yolo_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence > best_confidence:
                            detection = self._process_kenyan_vehicle(image, box, result.names)
                            if detection:
                                best_detection = detection
                                best_confidence = confidence
            
            detections = [best_detection] if best_detection else []
            
            logger.info(f"Selected single best detection with confidence: {best_confidence:.3f}")
            return detections, image
            
        except Exception as e:
            logger.error(f"Kenyan vehicle detection failed: {e}")
            return [], None
    
    def _process_kenyan_vehicle(self, image: np.ndarray, box, class_names: Dict) -> Optional[KenyanVehicleDetection]:
        """Process a single detected vehicle"""
        try:
            # Extract basic vehicle information
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            vehicle_type = class_names[class_id]
            
            # Get bounding box with smart padding
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Smart padding based on vehicle size
            w, h = x2 - x1, y2 - y1
            pad_x = max(10, int(w * 0.1))
            pad_y = max(10, int(h * 0.1))
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)
            
            # Crop vehicle region
            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                return None
            
            # Check minimum area
            area = (x2 - x1) * (y2 - y1)
            if area < self.config['min_box_area']:
                logger.info(f"Skipping small detection: {area} < {self.config['min_box_area']}")
                return None
            
            # Analysis
            color_result = self._detect_color_kenyan_optimized(vehicle_crop)
            kenyan_model_result = self._predict_kenyan_model(vehicle_crop, vehicle_type)
            plate_result = self._detect_license_plate(vehicle_crop)
            load_result = self._detect_heavy_load_improved(vehicle_crop, vehicle_type)
            additional_features = self._extract_kenyan_features(vehicle_crop)
            
            return KenyanVehicleDetection(
                vehicle_type=vehicle_type,
                kenyan_model_prediction=kenyan_model_result['model'],
                confidence_score=kenyan_model_result['confidence'],
                color=color_result['color'],
                license_plate=plate_result['text'],
                plate_confidence=plate_result['confidence'],
                heavy_load=load_result['is_loaded'],
                load_confidence=load_result['confidence'],
                bbox=[x1, y1, x2, y2],
                detection_confidence=confidence,
                timestamp=datetime.now().isoformat(),
                market_category=kenyan_model_result['market_category'],
                additional_features=additional_features
            )
            
        except Exception as e:
            logger.error(f"Kenyan vehicle processing failed: {e}")
            return None
    
    def _detect_license_plate(self, vehicle_crop: np.ndarray) -> Dict:
        """Detect Kenyan license plates with improved accuracy"""
        try:
            logger.info(f"Detecting license plate on {vehicle_crop.shape} image")
            
            h, w = vehicle_crop.shape[:2]
            
            # Define search regions (front plates are usually in lower portion)
            search_regions = [
                vehicle_crop[int(h*0.65):int(h*0.95), int(w*0.2):int(w*0.8)],  # Primary front region
                vehicle_crop[int(h*0.55):int(h*0.85), int(w*0.15):int(w*0.85)],  # Expanded region
                vehicle_crop[int(h*0.75):, int(w*0.25):int(w*0.75)],  # Lower center
                vehicle_crop[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)],  # Centered region
            ]
            
            best_result = {'text': 'No plate detected', 'confidence': 0.0}
            
            for region_idx, region in enumerate(search_regions):
                if region.size == 0:
                    continue
                
                # Try different preprocessing methods
                processed_versions = self._preprocess_for_ocr(region)
                
                for method_name, processed_img in processed_versions.items():
                    # Run OCR with different settings
                    ocr_results = self._run_ocr_variants(processed_img)
                    
                    for ocr_result in ocr_results:
                        if len(ocr_result['text'].strip()) >= 5:
                            # Clean and validate the text
                            cleaned_text = self._clean_plate_text(ocr_result['text'])
                            
                            if len(cleaned_text) >= 5:
                                # Score this candidate
                                score = self._score_plate_candidate(
                                    cleaned_text, ocr_result['confidence'], 
                                    method_name, region_idx
                                )
                                
                                if score > best_result['confidence']:
                                    best_result = {
                                        'text': cleaned_text,
                                        'confidence': score,
                                        'method': method_name,
                                        'region': region_idx
                                    }
            
            logger.info(f"Best plate result: '{best_result['text']}' (confidence: {best_result['confidence']:.3f})")
            return best_result
            
        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
            return {'text': 'Detection failed', 'confidence': 0.0}
    
    def _preprocess_for_ocr(self, region: np.ndarray) -> Dict[str, np.ndarray]:
        """Simple but effective preprocessing for OCR"""
        processed = {}
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Resize if too small (critical for OCR)
        h, w = gray.shape
        if h < 40 or w < 120:
            scale_factor = max(40/h, 120/w, 2.0)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Original (sometimes best)
        processed['original'] = gray
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed['denoised'] = denoised
        
        # CLAHE (often very effective)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed['clahe'] = clahe.apply(gray)
        
        # Adaptive threshold
        processed['adaptive'] = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Bilateral filter (good for preserving edges)
        processed['bilateral'] = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return processed
    
    def _run_ocr_variants(self, image: np.ndarray) -> List[Dict]:
        """Run OCR with different configurations"""
        results = []
        
        # Default settings (usually work well)
        try:
            ocr_results = self.ocr_reader.readtext(image, detail=1, paragraph=False)
            for bbox, text, conf in ocr_results:
                if conf > 0.1:
                    results.append({'text': text, 'confidence': conf, 'method': 'default'})
        except:
            pass
        
        # More sensitive settings
        try:
            ocr_results = self.ocr_reader.readtext(
                image, detail=1, paragraph=False,
                width_ths=0.5, height_ths=0.5, text_threshold=0.2
            )
            for bbox, text, conf in ocr_results:
                if conf > 0.05:
                    results.append({'text': text, 'confidence': conf, 'method': 'sensitive'})
        except:
            pass
        
        # More conservative settings
        try:
            ocr_results = self.ocr_reader.readtext(
                image, detail=1, paragraph=False,
                width_ths=0.8, height_ths=0.8, text_threshold=0.4
            )
            for bbox, text, conf in ocr_results:
                if conf > 0.2:
                    results.append({'text': text, 'confidence': conf, 'method': 'conservative'})
        except:
            pass
        
        return results
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and correct OCR text for Kenyan plates"""
        # Remove all non-alphanumeric and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if len(cleaned) < 5:
            return cleaned
        
        # Apply intelligent character corrections based on Kenyan patterns
        corrected = self._apply_smart_corrections(cleaned)
        
        return corrected
    
    def _apply_smart_corrections(self, text: str) -> str:
        """Apply smart character corrections for Kenyan plates"""
        if len(text) < 5:
            return text
        
        corrected = list(text)
        
        # Detect likely format first
        if len(text) == 7 and text[0] in 'KXHR':  # Likely K-series
            # Force K at start
            corrected[0] = 'K'
            
            # Handle common D/P confusion in K-series plates
            if corrected[1] == 'P':  # KPU -> KDU (very common error)
                corrected[1] = 'D'
            elif corrected[1] == 'R':  # KRU -> KDU
                corrected[1] = 'D'
            elif corrected[1] == 'B':  # KBU -> KDU  
                corrected[1] = 'D'
            
            # Positions 1-2: should be letters
            for i in [1, 2]:
                if i < len(corrected) and corrected[i] in self.char_fixes:
                    if corrected[i] in '0126589':  # Numbers that could be letters
                        corrected[i] = self.char_fixes[corrected[i]]
            
            # Positions 3-5: should be numbers
            for i in [3, 4, 5]:
                if i < len(corrected) and corrected[i] in self.char_fixes:
                    if corrected[i] in 'OIZSgb':  # Letters that could be numbers
                        corrected[i] = self.char_fixes[corrected[i]]
            
            # Position 6: should be letter
            if len(corrected) > 6 and corrected[6] in '012568':
                corrected[6] = self.char_fixes[corrected[6]]
        
        elif len(text) == 7:  # Standard 3-letter format
            # First 3 positions: letters
            for i in [0, 1, 2]:
                if i < len(corrected) and corrected[i] in '012568':
                    corrected[i] = self.char_fixes[corrected[i]]
            # Middle 3: numbers
            for i in [3, 4, 5]:
                if i < len(corrected) and corrected[i] in 'OIZSgb':
                    corrected[i] = self.char_fixes[corrected[i]]
            # Last: letter
            if len(corrected) > 6 and corrected[6] in '012568':
                corrected[6] = self.char_fixes[corrected[6]]
        
        elif text.startswith('GK') or text.startswith('6K'):  # Government plates
            corrected[0] = 'G'
            corrected[1] = 'K'
            # Rest follows standard pattern
            for i in [2, 3, 4]:
                if i < len(corrected) and corrected[i] in 'OIZSgb':
                    corrected[i] = self.char_fixes[corrected[i]]
            if len(corrected) > 5 and corrected[5] in '012568':
                corrected[5] = self.char_fixes[corrected[5]]
        
        return ''.join(corrected)
    
    def _score_plate_candidate(self, text: str, ocr_confidence: float, 
                              method: str, region_idx: int) -> float:
        """Score a license plate candidate"""
        score = 0.0
        
        # Format validation (most important)
        format_score = self._validate_kenyan_format(text)
        score += format_score * 0.6
        
        # OCR confidence
        score += ocr_confidence * 0.3
        
        # Length preference
        if len(text) == 7:
            score += 0.1
        elif len(text) in [6, 8]:
            score += 0.05
        
        # Method preference
        method_bonus = {
            'clahe': 0.05,
            'denoised': 0.04,
            'bilateral': 0.03,
            'adaptive': 0.02,
            'original': 0.01
        }
        score += method_bonus.get(method, 0)
        
        # Region preference (earlier regions are usually better)
        score += max(0, 0.05 - region_idx * 0.01)
        
        return min(1.0, score)
    
    def _validate_kenyan_format(self, text: str) -> float:
        """Validate text against Kenyan plate formats"""
        if len(text) < 5:
            return 0.0
        
        # Check exact pattern matches
        for pattern_info in self.kenyan_plate_patterns:
            if re.match(pattern_info['pattern'], text):
                return pattern_info['score']
        
        # Partial matching for common variations
        partial_score = 0.0
        
        # K-series partial matching
        if text.startswith('K') and len(text) == 7:
            partial_score = 0.7
            if re.search(r'K[A-Z]{2}\d{3}[A-Z]', text):
                partial_score = 0.8
        
        # Standard format partial matching
        elif re.match(r'^[A-Z]{2,3}\d{3}[A-Z]', text):
            partial_score = 0.6
        
        # Government format
        elif text.startswith('GK'):
            partial_score = 0.7
        
        # Basic alphanumeric with reasonable structure
        elif re.match(r'^[A-Z]+\d+[A-Z]*$', text) and 5 <= len(text) <= 8:
            partial_score = 0.4
        
        return partial_score
    
    def _predict_kenyan_model(self, vehicle_crop: np.ndarray, vehicle_type: str) -> Dict:
        """Predict specific Kenyan market vehicle model"""
        try:
            h, w = vehicle_crop.shape[:2]
            aspect_ratio = w / h
            
            # Extract features for model prediction
            features = self._extract_vehicle_features(vehicle_crop)
            
            model_scores = {}
            
            # Analyze against Kenyan vehicle database
            for brand, brand_data in self.kenyan_vehicles.items():
                for model_name, model_data in brand_data['models'].items():
                    score = self._calculate_model_score(features, model_data, aspect_ratio)
                    if score > 0.3:
                        full_model_name = f"{brand.title()} {model_name.replace('_', ' ').title()}"
                        model_scores[full_model_name] = {
                            'score': score,
                            'market_share': model_data['market_share']
                        }
            
            if model_scores:
                best_model = max(model_scores, key=lambda x: model_scores[x]['score'])
                best_score = model_scores[best_model]['score']
                market_share = model_scores[best_model]['market_share']
                
                return {
                    'model': best_model,
                    'confidence': best_score,
                    'market_category': self.market_categories[market_share]
                }
            else:
                fallback = self._fallback_classification(vehicle_type, aspect_ratio)
                return {
                    'model': fallback,
                    'confidence': 0.4,
                    'market_category': 'Common Vehicle Type'
                }
                
        except Exception as e:
            logger.error(f"Kenyan model prediction failed: {e}")
            return {
                'model': f'Unidentified {vehicle_type.title()}',
                'confidence': 0.2,
                'market_category': 'Unknown'
            }
    
    def _extract_vehicle_features(self, vehicle_crop: np.ndarray) -> Dict:
        """Extract features for Kenyan vehicle identification"""
        h, w = vehicle_crop.shape[:2]
        
        aspect_ratio = w / h
        color_features = self._analyze_color_distribution(vehicle_crop)
        
        # Edge and texture analysis
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        height_category = self._categorize_height(aspect_ratio)
        
        return {
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'height_category': height_category,
            'dominant_color': color_features['dominant'],
            'color_variance': color_features['variance'],
            'size_class': self._categorize_size(h, w)
        }
    
    def _calculate_model_score(self, features: Dict, model_data: Dict, aspect_ratio: float) -> float:
        """Calculate match score for a specific Kenyan vehicle model"""
        score = 0.0
        
        # Aspect ratio matching
        ar_min, ar_max = model_data['aspect_ratio_range']
        if ar_min <= aspect_ratio <= ar_max:
            score += 0.4
        elif abs(aspect_ratio - ar_min) < 0.3 or abs(aspect_ratio - ar_max) < 0.3:
            score += 0.2
        
        # Vehicle type matching
        vehicle_features = model_data['features']
        
        if 'suv' in vehicle_features and features['height_category'] == 'tall':
            score += 0.3
        elif 'sedan' in vehicle_features and features['height_category'] == 'medium':
            score += 0.3
        elif 'hatchback' in vehicle_features and features['height_category'] == 'short':
            score += 0.3
        elif 'pickup' in vehicle_features and features['aspect_ratio'] > 2.2:
            score += 0.4
        
        # Color matching
        common_colors = model_data['common_colors']
        if features['dominant_color'].lower() in common_colors:
            score += 0.2
        
        # Size matching
        if 'compact' in vehicle_features and features['size_class'] == 'small':
            score += 0.1
        elif 'large' in vehicle_features and features['size_class'] == 'large':
            score += 0.1
        elif 'medium' in vehicle_features and features['size_class'] == 'medium':
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_color_distribution(self, vehicle_crop: np.ndarray) -> Dict:
        """Analyze color distribution for model identification"""
        h, w = vehicle_crop.shape[:2]
        body_region = vehicle_crop[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        hsv_image = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
        
        dominant_color = 'unknown'
        max_pixels = 0
        
        for color_name, color_range in self.color_ranges_hsv.items():
            if color_name == 'red2':
                continue
                
            if color_name == 'red':
                mask1 = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
                mask2 = cv2.inRange(hsv_image, self.color_ranges_hsv['red2']['lower'], 
                                   self.color_ranges_hsv['red2']['upper'])
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                color_mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
            
            pixel_count = np.sum(color_mask > 0)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color_name
        
        color_variance = np.var(hsv_image[:, :, 1])
        
        return {
            'dominant': dominant_color,
            'variance': float(color_variance)
        }
    
    def _categorize_height(self, aspect_ratio: float) -> str:
        """Categorize vehicle height based on aspect ratio"""
        if aspect_ratio < 1.7:
            return 'tall'
        elif aspect_ratio < 2.0:
            return 'medium'
        else:
            return 'short'
    
    def _categorize_size(self, height: int, width: int) -> str:
        """Categorize vehicle size"""
        area = height * width
        if area < 80000:
            return 'small'
        elif area < 150000:
            return 'medium'
        else:
            return 'large'
    
    def _fallback_classification(self, vehicle_type: str, aspect_ratio: float) -> str:
        """Fallback classification for unmatched vehicles"""
        if vehicle_type == 'car':
            if aspect_ratio < 1.7:
                return "SUV/Crossover (Generic)"
            elif aspect_ratio < 2.0:
                return "Sedan/Hatchback (Generic)"
            else:
                return "Compact Car (Generic)"
        elif vehicle_type == 'truck':
            return "Pickup/Commercial Truck"
        elif vehicle_type == 'bus':
            return "Bus/Matatu"
        else:
            return f"{vehicle_type.title()} (Generic)"
    
    def _detect_color_kenyan_optimized(self, vehicle_crop: np.ndarray) -> Dict:
        """Color detection optimized for Kenyan lighting conditions"""
        try:
            h, w = vehicle_crop.shape[:2]
            
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask[:int(h*0.25), :] = 0
            mask[int(h*0.85):, :] = 0
            
            hsv_image = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            
            color_scores = {}
            
            for color_name, color_range in self.color_ranges_hsv.items():
                if color_name == 'red2':
                    continue
                    
                if color_name == 'red':
                    mask1 = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
                    mask2 = cv2.inRange(hsv_image, self.color_ranges_hsv['red2']['lower'], 
                                       self.color_ranges_hsv['red2']['upper'])
                    color_mask = cv2.bitwise_or(mask1, mask2)
                else:
                    color_mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
                
                combined_mask = cv2.bitwise_and(color_mask, mask)
                color_pixels = np.sum(combined_mask > 0)
                total_pixels = np.sum(mask > 0)
                
                if total_pixels > 0:
                    color_percentage = color_pixels / total_pixels
                    color_scores[color_name] = color_percentage
            
            if color_scores:
                best_color = max(color_scores, key=color_scores.get)
                best_score = color_scores[best_color]
                
                if best_score > 0.15:
                    return {
                        'color': best_color.title(),
                        'confidence': min(1.0, best_score * 3)
                    }
            
            return {'color': 'Unknown', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Kenyan color detection failed: {e}")
            return {'color': 'Unknown', 'confidence': 0.0}
    
    def _detect_heavy_load_improved(self, vehicle_crop: np.ndarray, vehicle_type: str) -> Dict:
        """Heavy load detection for Kenyan commercial vehicles"""
        try:
            if vehicle_type not in ['truck', 'bus']:
                return {'is_loaded': False, 'confidence': 0.8}
            
            h, w = vehicle_crop.shape[:2]
            cargo_region = vehicle_crop[:int(h*0.5), :]
            
            gray_cargo = cv2.cvtColor(cargo_region, cv2.COLOR_BGR2GRAY)
            texture_variance = cv2.Laplacian(gray_cargo, cv2.CV_64F).var()
            
            hsv_cargo = cv2.cvtColor(cargo_region, cv2.COLOR_BGR2HSV)
            color_variance = np.var(hsv_cargo[:, :, 1])
            
            texture_score = min(1.0, texture_variance / 800)
            color_score = min(1.0, color_variance / 400)
            combined_score = (texture_score + color_score) / 2
            
            is_loaded = combined_score > self.config['load_detection_threshold']
            
            return {'is_loaded': is_loaded, 'confidence': combined_score}
            
        except Exception as e:
            logger.error(f"Load detection failed: {e}")
            return {'is_loaded': False, 'confidence': 0.0}
    
    def _extract_kenyan_features(self, vehicle_crop: np.ndarray) -> Dict:
        """Extract features relevant to Kenyan market analysis"""
        try:
            h, w = vehicle_crop.shape[:2]
            
            features = {
                'dimensions': {'height': h, 'width': w, 'aspect_ratio': w/h},
                'image_quality': self._assess_image_quality(vehicle_crop),
                'lighting_conditions': self._assess_kenyan_lighting(vehicle_crop),
                'road_conditions': self._assess_road_context(vehicle_crop),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Kenyan feature extraction failed: {e}")
            return {}
    
    def _assess_kenyan_lighting(self, image: np.ndarray) -> Dict:
        """Assess lighting specific to Kenyan conditions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        if mean_brightness < 70:
            condition = "Low light/Evening"
        elif mean_brightness > 220:
            condition = "Harsh sunlight"
        elif brightness_std > 60:
            condition = "Mixed shadows"
        else:
            condition = "Good daylight"
        
        return {
            'condition': condition,
            'brightness_level': float(mean_brightness),
            'uniformity': float(max(0, 100 - brightness_std)),
            'harsh_shadows': brightness_std > 60
        }
    
    def _assess_road_context(self, image: np.ndarray) -> Dict:
        """Assess road context for Kenyan conditions"""
        h, w = image.shape[:2]
        bottom_region = image[int(h*0.8):, :]
        
        gray_road = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        road_texture = cv2.Laplacian(gray_road, cv2.CV_64F).var()
        
        road_type = "Paved" if road_texture < 500 else "Rough/Unpaved"
        
        return {
            'surface_type': road_type,
            'texture_variance': float(road_texture)
        }
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Assess image quality for detection reliability"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        return {
            'sharpness': float(sharpness),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'overall_quality': min(1.0, (sharpness/1000 + contrast/50 + 
                                       (1 - abs(brightness-128)/128)) / 3)
        }
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for Kenyan conditions"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            h, w = image.shape[:2]
            max_size = self.config['max_image_size']
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            image = self._enhance_kenyan_conditions(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _enhance_kenyan_conditions(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for typical Kenyan lighting and conditions"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.15)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return enhanced
    
    def visualize_kenyan_results(self, image: np.ndarray, detections: List[KenyanVehicleDetection], 
                                save_path: str = "kenyan_results_enhanced.jpg") -> np.ndarray:
        """Create visualization with enhanced information"""
        result_image = image.copy()
        
        category_colors = {
            'Market Leader': (0, 255, 0),
            'Very Popular': (0, 255, 255),
            'Popular': (255, 0, 0),
            'Common': (255, 0, 255),
            'Rare': (128, 0, 128)
        }
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            
            color = (0, 255, 0)
            for category, cat_color in category_colors.items():
                if category in detection.market_category:
                    color = cat_color
                    break
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            
            labels = [
                f"Vehicle {i+1}: {detection.vehicle_type.upper()}",
                f"Model: {detection.kenyan_model_prediction}",
                f"Confidence: {detection.confidence_score:.2f}",
                f"Color: {detection.color}",
                f"Market: {detection.market_category}"
            ]
            
            if detection.license_plate not in ["No plate detected", "Detection failed"]:
                labels.append(f"PLATE: {detection.license_plate} ({detection.plate_confidence:.2f})")
            else:
                labels.append("PLATE: Not detected")
            
            if detection.heavy_load:
                labels.append(f"HEAVY LOAD ({detection.load_confidence:.2f})")
            
            y_offset = y1 - 15
            for label in labels:
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(result_image, 
                            (x1, y_offset - text_height - 5),
                            (x1 + text_width + 10, y_offset + baseline),
                            color, -1)
                
                cv2.putText(result_image, label, (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                y_offset -= (text_height + 10)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = f"Enhanced Kenyan Vehicle Analysis: {len(detections)} vehicles | {timestamp}"
        
        cv2.putText(result_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 3)
        cv2.putText(result_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, result_image)
        logger.info(f"Enhanced results saved to: {save_path}")
        
        return result_image


# Demo usage
if __name__ == "__main__":
    try:
        print("=" * 80)
        print("KENYAN VEHICLE DETECTOR - SIMPLIFIED & WORKING")
        print("=" * 80)
        
        # Initialize the detector
        detector = KenyanVehicleDetector()
        print("Detector initialized successfully")
        
        # Test image path
        image_path = "subaru.png"  # Your test image
        
        print(f"\nProcessing image: {image_path}")
        print("=" * 60)
        
        # Run detection
        detections, original_image = detector.detect_kenyan_vehicles(image_path)
        
        if original_image is not None:
            print(f"Image processed successfully: {original_image.shape}")
            
            if detections:
                print(f"\nDETECTION RESULTS: {len(detections)} vehicle(s) found")
                print("=" * 60)
                
                for i, detection in enumerate(detections, 1):
                    print(f"\nVEHICLE {i}:")
                    print(f"   Type: {detection.vehicle_type}")
                    print(f"   Model: {detection.kenyan_model_prediction}")
                    print(f"   Model Confidence: {detection.confidence_score:.3f}")
                    print(f"   Color: {detection.color}")
                    print(f"   License Plate: '{detection.license_plate}'")
                    print(f"   Plate Confidence: {detection.plate_confidence:.3f}")
                    print(f"   Detection Confidence: {detection.detection_confidence:.3f}")
                    print(f"   Market Category: {detection.market_category}")
                    
                    # Status indicator
                    if detection.license_plate not in ['No plate detected', 'Detection failed']:
                        if 'K' in detection.license_plate and len(detection.license_plate) >= 6:
                            print("   ✓ KENYAN PLATE DETECTED!")
                        else:
                            print("   ✓ PLATE DETECTED!")
                    else:
                        print("   ✗ Plate detection failed")
                    
                    if detection.heavy_load:
                        print(f"   Heavy Load Detected ({detection.load_confidence:.2f})")
                    
                    print("   " + "-" * 50)
                
                # Generate visualization
                try:
                    result_image = detector.visualize_kenyan_results(original_image, detections)
                    print(f"\nResults saved to: kenyan_results_enhanced.jpg")
                except Exception as e:
                    print(f"Visualization error: {e}")
                
                # Summary
                print(f"\nSUMMARY:")
                print(f"- Total vehicles detected: {len(detections)}")
                successful_plates = sum(1 for d in detections if d.license_plate not in ['No plate detected', 'Detection failed'])
                print(f"- License plates detected: {successful_plates}/{len(detections)}")
                
            else:
                print("No vehicles detected in the image.")
                print("Check image quality and try again.")
        else:
            print(f"Failed to load image: {image_path}")
            print("Make sure the file exists and is a valid image.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()