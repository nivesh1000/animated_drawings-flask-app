import animated_drawings.render
from pathlib import Path
import yaml
from pkg_resources import resource_filename
import sys
import time


def annotations_to_animation(users_choice: str) -> None:
    """
    Create an animation based on user choice and save it as 'video.gif' in the 'characterfiles' directory.

    Args:
        users_choice (str): User's choice to determine which animation to create. Valid options are '1', '2', '3', '4', '5'.

    Returns:
        None
    """
    print(users_choice)

    # Dictionary mapping user choices to animation configurations
    animation_configs = {
        '1': {
            'character_cfg': "characterfiles/char_cfg.yaml",
            'motion_cfg': 'RenderAnimation/examples/config/motion/jesse_dance.yaml',
            'retarget_cfg': 'RenderAnimation/examples/config/retarget/mixamo_fff.yaml'
        },
        '2': {
            'character_cfg': "characterfiles/char_cfg.yaml",
            'motion_cfg': 'RenderAnimation/examples/config/motion/jumping.yaml',
            'retarget_cfg': 'RenderAnimation/examples/config/retarget/fair1_spf.yaml'
        },
        '3': {
            'character_cfg': "characterfiles/char_cfg.yaml",
            'motion_cfg': 'RenderAnimation/examples/config/motion/zombie.yaml',
            'retarget_cfg': 'RenderAnimation/examples/config/retarget/fair1_spf.yaml'
        },
        '4': {
            'character_cfg': "characterfiles/char_cfg.yaml",
            'motion_cfg': 'RenderAnimation/examples/config/motion/wave_hello.yaml',
            'retarget_cfg': 'RenderAnimation/examples/config/retarget/fair1_spf.yaml'
        },
        '5': {
            'character_cfg': "characterfiles/char_cfg.yaml",
            'motion_cfg': 'RenderAnimation/examples/config/motion/jumping_jacks.yaml',
            'retarget_cfg': 'RenderAnimation/examples/config/retarget/cmu1_pfp.yaml'
        }
    }
    
    animated_drawing_dict = animation_configs[users_choice]

    # Create MVC config
    mvc_cfg = {
        'scene': {'ANIMATED_CHARACTERS': [animated_drawing_dict]},  # Add the character to the scene
        'controller': {
            'MODE': 'video_render',  # 'video_render' or 'interactive'
            'OUTPUT_VIDEO_PATH': "characterfiles/video.gif"  # Set the output location
        }
    }


    # Write the new MVC config file
    output_mvc_cfn_fn = "characterfiles/output_mvc.yaml"
    with open(output_mvc_cfn_fn, 'w') as f:
        yaml.dump(mvc_cfg, f)

    # Render the video
    animated_drawings.render.start(output_mvc_cfn_fn)
