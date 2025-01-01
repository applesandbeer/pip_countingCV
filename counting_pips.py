import cv2
import numpy as np
from pathlib import Path

def detect_pips(image_path):
    """
    Detect pips by finding white circles within black domino in binary image.
    """
    # Read image (assuming it's already binary)
    binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create debug directory
    debug_dir = Path(image_path).parent / 'debug' / Path(image_path).stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input binary
    cv2.imwrite(str(debug_dir / '1_input_binary.jpg'), binary)
    
    # Find all contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create debug image for contours
    contour_debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    valid_pips = []
    min_radius = 5
    max_radius = 50
    
    for contour in contours:
        # Get enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Check if size is reasonable
        if radius < min_radius or radius > max_radius:
            continue
        
        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Only keep very circular contours (close to 1.0)
        if circularity > 0.8:
            valid_pips.append((center, radius))
            # Draw on debug image in different colors
            cv2.circle(contour_debug, center, radius, (0, 255, 0), 2)
            cv2.circle(contour_debug, center, 2, (0, 0, 255), -1)
    
    # Save debug visualization
    cv2.imwrite(str(debug_dir / '2_detected_pips.jpg'), contour_debug)
    
    return len(valid_pips), contour_debug

def process_folder(folder_path):
    """Process all images in a folder."""
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for image_path in folder.glob('*'):
        if image_path.suffix.lower() in image_extensions:
            try:
                pip_count, annotated = detect_pips(str(image_path))
                print(f"Found {pip_count} pips in {image_path.name}")
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")

if __name__ == "__main__":
    process_folder("pics")