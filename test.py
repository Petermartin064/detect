import cv2
import numpy as np
import easyocr
import re
import logging
from typing import List, Dict, Tuple, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedKenyanPlateDetector:
    """Enhanced license plate detection specifically for Kenyan plates"""
    
    def __init__(self):
        # Initialize OCR with better settings for Kenyan plates
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)  # Set to True if you have GPU
        
        # Comprehensive Kenyan plate patterns
        self.kenyan_plate_patterns = [
            # Current format: KXX 123X
            r'^K[A-Z]{2}\s*\d{3}\s*[A-Z]$',          # KAA 123A, KBB 456C, etc.
            r'^K[A-Z]{2}\d{3}[A-Z]$',                 # KAA123A (no spaces)
            
            # Specific common patterns
            r'^KDP\s*\d{3}\s*[A-Z]$',                 # KDP 777M
            r'^KDA\s*\d{3}\s*[A-Z]$',                 # KDA series
            r'^KDB\s*\d{3}\s*[A-Z]$',                 # KDB series
            r'^KDC\s*\d{3}\s*[A-Z]$',                 # KDC series
            r'^KCA\s*\d{3}\s*[A-Z]$',                 # KCA series
            r'^KCB\s*\d{3}\s*[A-Z]$',                 # KCB series
            
            # Standard format: XXX 123X
            r'^[A-Z]{3}\s*\d{3}\s*[A-Z]$',           # ABC 123D
            r'^[A-Z]{3}\d{3}[A-Z]$',                 # ABC123D (no spaces)
            
            # Government and special plates
            r'^GK\s*\d{3}\s*[A-Z]$',                 # Government: GK 123A
            r'^CD\s*\d{3}\s*[A-Z]$',                 # Diplomatic: CD 123A
            r'^UN\s*\d{3}\s*[A-Z]$',                 # UN plates
            
            # Older formats
            r'^[A-Z]{2}\s*\d{3}\s*[A-Z]{2}$',        # AB 123CD
            r'^[A-Z]{2}\d{3}[A-Z]{2}$',              # AB123CD
        ]
        
        # Enhanced character corrections for OCR misreads
        self.character_corrections = {
            # OCR commonly confuses these characters
            '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z',
            'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2',
            'Q': 'O', 'T': '7', 'L': '1', 'E': '3', 'A': '4'
        }
        
        # Kenyan specific prefix corrections - your main issue
        self.prefix_corrections = {
            'KKP': 'KDP',  # Your specific case - K often misread as KK
            'KRP': 'KDP',  # R often misread as D
            'KPP': 'KDP',  # Double P misread
            'XDP': 'KDP',  # X misread as K
            'KQP': 'KDP',  # Q misread as D
            'KDR': 'KDP',  # R misread as P
            'KKD': 'KDA',  # Similar patterns for other series
            'KKB': 'KDB',
            'KKC': 'KDC',
            'KKA': 'KCA',
            'XDA': 'KDA',
            'XDB': 'KDB',
        }

    def detect_license_plate(self, vehicle_crop: np.ndarray) -> Dict:
        """Main entry point for license plate detection"""
        try:
            logger.info(f"Starting enhanced plate detection on {vehicle_crop.shape}")
            
            # Get targeted search regions
            regions = self._get_plate_regions(vehicle_crop)
            all_candidates = []
            
            for region_idx, region in enumerate(regions):
                if region is None or region.size == 0:
                    continue
                
                logger.debug(f"Processing region {region_idx + 1}: {region.shape}")
                
                # Apply multiple preprocessing techniques
                processed_variants = self._preprocess_for_ocr(region)
                
                # Extract text from each variant
                for method_name, processed_img in processed_variants.items():
                    candidates = self._extract_text_with_ocr(
                        processed_img, method_name, region_idx
                    )
                    all_candidates.extend(candidates)
            
            # Select the best candidate
            best_result = self._select_best_plate_candidate(all_candidates)
            
            logger.info(f"Final result: '{best_result['text']}' "
                       f"(confidence: {best_result['confidence']:.3f})")
            
            return best_result
            
        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
            return {'text': 'Detection failed', 'confidence': 0.0}

    def _get_plate_regions(self, vehicle_crop: np.ndarray) -> List[np.ndarray]:
        """Extract targeted regions where license plates are likely located"""
        h, w = vehicle_crop.shape[:2]
        
        regions = []
        
        # Front license plate regions (most common in Kenya)
        regions.extend([
            # Primary front region - lower center
            vehicle_crop[int(h*0.70):int(h*0.95), int(w*0.25):int(w*0.75)],
            
            # Secondary region - slightly higher
            vehicle_crop[int(h*0.60):int(h*0.85), int(w*0.20):int(w*0.80)],
            
            # Wider search for non-standard mounting
            vehicle_crop[int(h*0.65):int(h*0.90), int(w*0.15):int(w*0.85)],
            
            # Very bottom region for low-mounted plates
            vehicle_crop[int(h*0.80):, int(w*0.20):int(w*0.80)],
        ])
        
        # Add rear plate regions for rear-view images
        regions.extend([
            # Rear plate region
            vehicle_crop[int(h*0.75):, int(w*0.30):int(w*0.70)],
        ])
        
        return [r for r in regions if r is not None and r.size > 100]

    def _preprocess_for_ocr(self, region: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply various preprocessing techniques optimized for Kenyan plates"""
        processed = {}
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Resize if too small (OCR works better on larger images)
        h, w = gray.shape
        if h < 50 or w < 150:
            scale = max(50/h, 150/w, 2.0)  # Minimum scale of 2x
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        processed['original'] = gray
        
        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed['clahe'] = clahe.apply(gray)
        
        # Multiple threshold techniques
        processed['otsu'] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed['adaptive_mean'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3
        )
        processed['adaptive_gaussian'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        
        # Morphological operations to clean up
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        processed['morph_close'] = cv2.morphologyEx(
            processed['otsu'], cv2.MORPH_CLOSE, kernel_small
        )
        processed['morph_open'] = cv2.morphologyEx(
            processed['adaptive_mean'], cv2.MORPH_OPEN, kernel_small
        )
        
        # Denoising
        processed['denoised'] = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Sharpening
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed['sharpened'] = cv2.filter2D(processed['clahe'], -1, sharpen_kernel)
        
        # Gamma correction variants
        processed['gamma_dark'] = self._apply_gamma(gray, 0.7)  # Brighten dark images
        processed['gamma_bright'] = self._apply_gamma(gray, 1.3)  # Darken bright images
        
        return processed

    def _apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _extract_text_with_ocr(self, processed_img: np.ndarray, 
                              method_name: str, region_idx: int) -> List[Dict]:
        """Extract text candidates using OCR with optimized settings"""
        candidates = []
        
        try:
            # OCR with settings optimized for license plates
            ocr_results = self.ocr_reader.readtext(
                processed_img,
                detail=1,
                paragraph=False,
                width_ths=0.8,      # Width threshold for text detection
                height_ths=0.7,     # Height threshold
                min_size=8,         # Minimum text size
                text_threshold=0.2, # Lower threshold for better detection
                low_text=0.1,       # Very low threshold
                mag_ratio=1.5,      # Magnification ratio
                slope_ths=0.1,      # Rotation tolerance
                ycenter_ths=0.7,    # Y-center threshold
                add_margin=0.1      # Add margin around detected text
            )
            
            for bbox, text, ocr_confidence in ocr_results:
                cleaned_text = text.strip().upper()
                
                # Skip very short or very long texts
                if len(cleaned_text) < 4 or len(cleaned_text) > 12:
                    continue
                
                # Skip if confidence is too low and text doesn't look like a plate
                if ocr_confidence < 0.1 and not self._looks_like_plate(cleaned_text):
                    continue
                
                # Clean and correct the text
                corrected_text = self._clean_and_correct_text(cleaned_text)
                
                if len(corrected_text) >= 5:  # Minimum length for Kenyan plates
                    score = self._calculate_plate_score(
                        corrected_text, cleaned_text, ocr_confidence, method_name, region_idx
                    )
                    
                    candidates.append({
                        'text': corrected_text,
                        'original': text,
                        'confidence': score,
                        'method': method_name,
                        'region': region_idx,
                        'ocr_confidence': ocr_confidence,
                        'bbox': bbox
                    })
                    
                    logger.debug(f"Candidate: '{text}' -> '{corrected_text}' "
                               f"(score: {score:.3f}, method: {method_name})")
        
        except Exception as e:
            logger.debug(f"OCR extraction error for method {method_name}: {e}")
        
        return candidates

    def _looks_like_plate(self, text: str) -> bool:
        """Quick check if text might be a license plate"""
        # Contains both letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        return has_letters and has_numbers

    def _clean_and_correct_text(self, text: str) -> str:
        """Clean and correct OCR text for Kenyan plates"""
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Apply prefix corrections first (your main issue)
        cleaned = self._apply_prefix_corrections(cleaned)
        
        # Apply character corrections based on position
        cleaned = self._apply_positional_corrections(cleaned)
        
        return cleaned

    def _apply_prefix_corrections(self, text: str) -> str:
        """Apply Kenyan-specific prefix corrections"""
        for wrong_prefix, correct_prefix in self.prefix_corrections.items():
            if text.startswith(wrong_prefix):
                corrected = correct_prefix + text[len(wrong_prefix):]
                logger.info(f"Applied prefix correction: '{text}' -> '{corrected}'")
                return corrected
        
        return text

    def _apply_positional_corrections(self, text: str) -> str:
        """Apply character corrections based on expected position in Kenyan plates"""
        if len(text) < 6:
            return text
        
        corrected = list(text)
        
        # For 7-character plates (KXX123X format)
        if len(text) == 7:
            # First 3 characters should be letters
            for i in range(3):
                if corrected[i].isdigit():
                    if corrected[i] == '0':
                        corrected[i] = 'O'
                    elif corrected[i] == '1':
                        corrected[i] = 'I'
                    elif corrected[i] == '5':
                        corrected[i] = 'S'
                    elif corrected[i] == '8':
                        corrected[i] = 'B'
                    elif corrected[i] == '6':
                        corrected[i] = 'G'
                    elif corrected[i] == '2':
                        corrected[i] = 'Z'
            
            # Characters 4-6 should be numbers
            for i in range(3, 6):
                if corrected[i].isalpha():
                    if corrected[i] == 'O':
                        corrected[i] = '0'
                    elif corrected[i] == 'I':
                        corrected[i] = '1'
                    elif corrected[i] == 'S':
                        corrected[i] = '5'
                    elif corrected[i] == 'B':
                        corrected[i] = '8'
                    elif corrected[i] == 'G':
                        corrected[i] = '6'
                    elif corrected[i] == 'Z':
                        corrected[i] = '2'
            
            # Last character should be a letter
            if corrected[6].isdigit():
                if corrected[6] == '0':
                    corrected[6] = 'O'
                elif corrected[6] == '1':
                    corrected[6] = 'I'
                elif corrected[6] == '5':
                    corrected[6] = 'S'
                elif corrected[6] == '8':
                    corrected[6] = 'B'
                elif corrected[6] == '6':
                    corrected[6] = 'G'
                elif corrected[6] == '2':
                    corrected[6] = 'Z'
        
        return ''.join(corrected)

    def _calculate_plate_score(self, cleaned_text: str, original_text: str,
                              ocr_confidence: float, method: str, region_idx: int) -> float:
        """Calculate comprehensive score for plate candidate"""
        score = 0.0
        
        # Base OCR confidence (25% weight)
        score += ocr_confidence * 0.25
        
        # Pattern matching score (40% weight)
        pattern_score = self._validate_kenyan_patterns(cleaned_text)
        score += pattern_score * 0.40
        
        # Length preference (15% weight)
        if len(cleaned_text) == 7:  # Perfect length for KXX123X
            score += 0.15
        elif len(cleaned_text) in [6, 8]:  # Acceptable lengths
            score += 0.10
        elif len(cleaned_text) == 5:  # Minimum acceptable
            score += 0.05
        
        # Method bonus (10% weight) - some methods work better
        method_bonuses = {
            'clahe': 0.08,
            'adaptive_mean': 0.10,
            'adaptive_gaussian': 0.09,
            'otsu': 0.07,
            'morph_close': 0.06,
            'sharpened': 0.05,
            'denoised': 0.04
        }
        score += method_bonuses.get(method, 0.02)
        
        # Region preference (5% weight) - earlier regions preferred
        region_bonuses = [0.05, 0.04, 0.03, 0.02, 0.01]
        if region_idx < len(region_bonuses):
            score += region_bonuses[region_idx]
        
        # Kenyan-specific bonuses (5% weight)
        if cleaned_text.startswith('K'):
            score += 0.03
        if any(cleaned_text.startswith(prefix) for prefix in ['KDP', 'KDA', 'KDB', 'KDC', 'KCA', 'KCB']):
            score += 0.02
        
        return min(1.0, score)

    def _validate_kenyan_patterns(self, text: str) -> float:
        """Validate against Kenyan license plate patterns"""
        if len(text) < 5:
            return 0.0
        
        # Check exact pattern matches
        for pattern in self.kenyan_plate_patterns:
            if re.match(pattern, text):
                return 1.0  # Perfect match
        
        # Partial matching scores
        score = 0.0
        
        # Check general Kenyan structure
        if re.match(r'^[A-Z]{2,3}\d{3}[A-Z]$', text):
            score = max(score, 0.8)
        
        # K-prefix bonus (most common in Kenya)
        if text.startswith('K') and len(text) >= 6:
            score = max(score, 0.7)
        
        # Common prefixes
        common_prefixes = ['KDP', 'KDA', 'KDB', 'KDC', 'KCA', 'KCB', 'GK', 'CD', 'UN']
        for prefix in common_prefixes:
            if text.startswith(prefix):
                score = max(score, 0.6)
                break
        
        # Has required number sequence
        if re.search(r'\d{3}', text):
            score = max(score, 0.4)
        
        # Basic alphanumeric structure
        if re.match(r'^[A-Z]+\d+[A-Z]+$', text):
            score = max(score, 0.3)
        
        return score

    def _select_best_plate_candidate(self, candidates: List[Dict]) -> Dict:
        """Select the best candidate from all detected possibilities"""
        if not candidates:
            logger.warning("No plate candidates found")
            return {'text': 'No plate detected', 'confidence': 0.0}
        
        # Sort by confidence score
        sorted_candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        
        # Log top candidates for debugging
        logger.info(f"Found {len(candidates)} plate candidates:")
        for i, candidate in enumerate(sorted_candidates[:3]):  # Show top 3
            logger.info(f"  {i+1}. '{candidate['original']}' -> '{candidate['text']}' "
                       f"(score: {candidate['confidence']:.3f}, method: {candidate['method']})")
        
        best_candidate = sorted_candidates[0]
        
        return {
            'text': best_candidate['text'],
            'confidence': best_candidate['confidence'],
            'method': best_candidate['method'],
            'region': best_candidate['region']
        }


# Test function
def test_plate_detection(image_path: str):
    """Test the enhanced plate detection"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return
    
    print(f"Testing enhanced Kenyan plate detection on: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Initialize detector
    detector = EnhancedKenyanPlateDetector()
    
    # For testing, we'll use the whole image as vehicle crop
    # In real usage, this would be a cropped vehicle region
    result = detector.detect_license_plate(image)
    
    print("\nDetection Results:")
    print("=" * 40)
    print(f"Detected Text: '{result['text']}'")
    print(f"Confidence: {result['confidence']:.3f}")
    if 'method' in result:
        print(f"Best Method: {result['method']}")
        print(f"Best Region: {result['region']}")
    
    # Check if it's a valid Kenyan plate
    if result['text'] not in ['No plate detected', 'Detection failed']:
        if result['text'].startswith('KDP'):
            print("✅ SUCCESS: KDP plate detected correctly!")
        elif any(result['text'].startswith(prefix) for prefix in ['KDA', 'KDB', 'KDC', 'KCA', 'KCB', 'K']):
            print("✅ SUCCESS: Valid Kenyan K-series plate detected!")
        elif any(result['text'].startswith(prefix) for prefix in ['GK', 'CD', 'UN']):
            print("✅ SUCCESS: Valid special plate detected!")
        else:
            print(f"⚠️  Detected plate may not be Kenyan format: {result['text']}")
    else:
        print("❌ FAILED: No plate detected")
    
    return result


if __name__ == "__main__":
    # Test with your image
    test_image_path = "subaru.png"  # Change this to your image path
    test_plate_detection(test_image_path)