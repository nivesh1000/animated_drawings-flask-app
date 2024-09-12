from humanoid_detector import detection
from pose_estimator import pose_estimator



def main():
    image_path='CompatibilityChecker/drawn_humanoid_detector/image1.jpg'
    #reading image this time but in flask app the image will be uploaded by the user.
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    results=detection(image_bytes)
    if len(results[0])==0:
        print("--No humanoid figure detected.")
    else:
        print(f"--Humanoid figure detected with score {results[0][0]['score']}")
        pose_estimator(results[0], image_path)
if __name__ == "__main__":
    main()    