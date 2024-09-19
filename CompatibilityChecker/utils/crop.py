import cv2
from typing import List, Dict

def crop_image(detection_results: List[Dict[str, float]], image_path: str) -> cv2.Mat:
    """
    Crop an image based on bounding box coordinates from detection results.

    Args:
        detection_results (List[Dict[str, float]]): Detection results containing
            bounding box coordinates.
        image_path (str): Path to the input image.

    Returns:
        cv2.Mat: The cropped image.
    """
    img = cv2.imread(image_path)
    
    # Extract bounding box coordinates
    bbox = detection_results[0]['bbox']
    l, t, r, b = [round(x) for x in bbox]

    # Crop the image
    cropped = img[t:b, l:r]
    cv2.imwrite('characterfiles/texture.png', cropped)
    
    return cropped
