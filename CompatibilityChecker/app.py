from humanoid_detector import detection
from pose_estimator import pose_estimator

def main() -> None:
    """
    Main function to detect humanoid figures in an image and estimate
    their pose. The image is read from a file, and results are processed
    using the `detection` and `pose_estimator` functions.
    """
    image_path = 'CompatibilityChecker/drawn_humanoid_detector/image1.jpg'
    
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    results = detection(image_bytes)
    
    if len(results) == 0:
        print("--No humanoid figure detected.")
    else:
        print(f"--Humanoid figure detected with score {results[0]['score']}")
        
        pose_estimator(results, image_path)

if __name__ == "__main__":
    main()
