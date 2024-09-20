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
    try:
        # Attempt to open and read the image file
        with open(image_path, 'rb') as image_file:
            image = image_file.read()
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return 0
    except IOError:
        print(f"Error: Could not read file '{image_path}'.")
        return 0

    try:
        # Attempt to detect humanoid figures in the image
        results = detection(image)
    except Exception as e:
        print(f"Error during humanoid detection: {e}")
        return 0

    if not results:
        print("--No humanoid figure detected.")
        return 0
    else:
        try:
            # Attempt to estimate pose if a humanoid is detected
            print(f"--Humanoid figure detected with score {results[0]['score']}")
            pose_estimator(results, image_path)
        except Exception as e:
            print(f"Error during pose estimation: {e}")
            return 0

        return 1
