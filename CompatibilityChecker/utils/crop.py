import cv2

def crop_image(detection_results,image_path):
    img = cv2.imread(image_path)

    bbox = detection_results[0]['bbox']
    l, t, r, b = [round(x) for x in bbox]

    # crop the image
    cropped = img[t:b, l:r]
    cv2.imwrite('CompatibilityChecker/drawn_humanoid_pose_estimator/output_files/texture.png', cropped)
    return cropped