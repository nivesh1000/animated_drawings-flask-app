from CompatibilityChecker.humanoid_detector import detection
from CompatibilityChecker.pose_estimator import pose_estimator

def models(image_path: str) -> int:
    """
    Process an image to detect humanoid figures and estimate pose.

    Args:
        image_path (str): Path to the input image.

    Returns:
        int: Returns 1 if a humanoid figure is detected and 0 otherwise.
    """
    with open(image_path, 'rb') as image_file:
        image = image_file.read()

    results = detection(image)
    
    if not results:
        print("--No humanoid figure detected.")
        return 0
    else:
        print(f"--Humanoid figure detected with score {results[0]['score']}")
        pose_estimator(results, image_path)
        return 1
