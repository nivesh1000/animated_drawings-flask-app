from animated_drawings.render import start
from pathlib import Path
import yaml

def annotations_to_animation():
    """
    Given a path to a directory with character annotations, a motion configuration file, and a retarget configuration file,
    creates an animation and saves it to {annotation_dir}/video.png
    """

    # package character_cfg_fn, motion_cfg_fn, and retarget_cfg_fn
    animated_drawing_dict = {
        'character_cfg': "CompatibilityChecker/drawn_humanoid_pose_estimator/output_files/char_cfg.yaml",
        'motion_cfg': "RenderAnimation/ConfigFiles/config/motion/dab.yaml",
        'retarget_cfg': "RenderAnimation/ConfigFiles/config/retarget/fair1_spf.yaml"
    }

    # create mvc config
    mvc_cfg = {
        'scene': {'ANIMATED_CHARACTERS': [animated_drawing_dict]},  # add the character to the scene
        'controller': {
            'MODE': 'video_render',  # 'video_render' or 'interactive'
            'OUTPUT_VIDEO_PATH': "CompatibilityChecker/drawn_humanoid_pose_estimator/output_files/video.gif"} # set the output location
    }

    # write the new mvc config file out
    output_mvc_cfn_fn = "CompatibilityChecker/drawn_humanoid_pose_estimator/output_files/mvc_cfg.yaml"
    with open(output_mvc_cfn_fn, 'w') as f:
        yaml.dump(dict(mvc_cfg), f)

    # render the video
    start(output_mvc_cfn_fn)

annotations_to_animation()    