from CompatibilityChecker.utils.crop import crop_image
from CompatibilityChecker.utils.segment import segment
from CompatibilityChecker.drawn_humanoid_pose_estimator.mmpose_handler import MMPoseHandler
import cv2
import numpy as np
import yaml
import os

class Context:
    """Context class to store system properties and model configuration."""
    
    def __init__(self) -> None:
        self.system_properties = {
            'model_dir': 'CompatibilityChecker/drawn_humanoid_pose_estimator/',
            'gpu_id': 0
        }
        self.manifest = {
            'model': {
                'serializedFile': 'best_AP_epoch_72.pth'
            }
        }

def pose_estimator(datafile: str, image_path: str) -> None:
    """
    Estimate pose from an image and save the configuration and overlay.

    Args:
        datafile (str): Path to the data file.
        image_path (str): Path to the input image.

    Returns:
        None
    """
    try:
        # Initialize the handler
        context = Context()
        handler = MMPoseHandler()
        handler.initialize(context)
    except Exception as e:
        print(f"Error initializing pose handler: {e}")
        return

    try:
        # Crop and segment the image
        cropped = crop_image(datafile, image_path)
        segment(cropped)
    except Exception as e:
        print(f"Error during image cropping or segmentation: {e}")
        return

    try:
        # Convert the cropped image to bytes
        data = [{'data': cv2.imencode('.png', cropped)[1].tobytes()}]

        # Preprocess the input data
        preprocessed_data = handler.preprocess(data)

        # Run inference
        inference_results = handler.inference(preprocessed_data)

        # Postprocess the results
        pose_results = handler.postprocess(inference_results)
    except Exception as e:
        print(f"Error during pose estimation or inference: {e}")
        return

    try:
        # Get x, y coordinates of detection joint keypoints
        kpts = np.array(pose_results[0]['keypoints'])[:, :2]

        # Build character skeleton
        skeleton = [
            {'loc': [round(x) for x in (kpts[11] + kpts[12]) / 2], 'name': 'root', 'parent': None},
            {'loc': [round(x) for x in (kpts[11] + kpts[12]) / 2], 'name': 'hip', 'parent': 'root'},
            {'loc': [round(x) for x in (kpts[5] + kpts[6]) / 2], 'name': 'torso', 'parent': 'hip'},
            {'loc': [round(x) for x in kpts[0]], 'name': 'neck', 'parent': 'torso'},
            {'loc': [round(x) for x in kpts[6]], 'name': 'right_shoulder', 'parent': 'torso'},
            {'loc': [round(x) for x in kpts[8]], 'name': 'right_elbow', 'parent': 'right_shoulder'},
            {'loc': [round(x) for x in kpts[10]], 'name': 'right_hand', 'parent': 'right_elbow'},
            {'loc': [round(x) for x in kpts[5]], 'name': 'left_shoulder', 'parent': 'torso'},
            {'loc': [round(x) for x in kpts[7]], 'name': 'left_elbow', 'parent': 'left_shoulder'},
            {'loc': [round(x) for x in kpts[9]], 'name': 'left_hand', 'parent': 'left_elbow'},
            {'loc': [round(x) for x in kpts[12]], 'name': 'right_hip', 'parent': 'root'},
            {'loc': [round(x) for x in kpts[14]], 'name': 'right_knee', 'parent': 'right_hip'},
            {'loc': [round(x) for x in kpts[16]], 'name': 'right_foot', 'parent': 'right_knee'},
            {'loc': [round(x) for x in kpts[11]], 'name': 'left_hip', 'parent': 'root'},
            {'loc': [round(x) for x in kpts[13]], 'name': 'left_knee', 'parent': 'left_hip'},
            {'loc': [round(x) for x in kpts[15]], 'name': 'left_foot', 'parent': 'left_knee'}
        ]

        # Create character configuration
        char_cfg = {
            'skeleton': skeleton,
            'height': cropped.shape[0],
            'width': cropped.shape[1]
        }

        # Write the new character configuration file
        os.makedirs('characterfiles', exist_ok=True)
        with open('characterfiles/char_cfg.yaml', 'w') as f:
            yaml.dump(char_cfg, f)
        print("--Configuration file created.")
    except Exception as e:
        print(f"Error creating character configuration: {e}")
        return

    try:
        # Draw joints on the image and create an overlay
        joint_overlay = cropped.copy()
        for joint in skeleton:
            x, y = joint['loc']
            name = joint['name']
            cv2.circle(joint_overlay, (int(x), int(y)), 5, (0, 0, 0), 5)
            cv2.putText(joint_overlay, name, (int(x), int(y + 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)

        # Save the joint overlay image
        cv2.imwrite('characterfiles/joint_overlay.png', joint_overlay)
        print("--Joint_overlay.png created successfully.")
    except Exception as e:
        print(f"Error generating or saving joint overlay: {e}")
