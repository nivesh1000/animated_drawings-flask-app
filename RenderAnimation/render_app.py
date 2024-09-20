import animated_drawings.render
import yaml
from typing import Optional

def annotations_to_animation(users_choice: str) -> None:
    """
    Create an animation based on user choice and save it as 'video.gif'
    in the 'characterfiles' directory.

    Args:
        users_choice (str): User's choice to determine which animation 
        to create. Valid options are '1', '2', '3', '4', '5'.

    Returns:
        None
    """
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

    # Get the animation configuration based on the user's choice
    animated_drawing_dict: Optional[dict] = animation_configs.get(users_choice)
    
    if animated_drawing_dict is None:
        print(f"animated_drawing_dict not initialized properly.")
        return

    # Create MVC config
    mvc_cfg = {
        'scene': {'ANIMATED_CHARACTERS': [animated_drawing_dict]},
        'controller': {
            'MODE': 'video_render',
            'OUTPUT_VIDEO_PATH': "characterfiles/video.gif"  # Set the output location
        }
    }

    output_mvc_cfg_fn = "characterfiles/output_mvc.yaml"

    try:
        # Writing the YAML configuration file
        with open(output_mvc_cfg_fn, 'w') as f:
            yaml.dump(mvc_cfg, f)
    except (OSError, IOError) as e:
        print(f"Error writing config file: {e}")
        return

    try:
        # Render the video
        animated_drawings.render.start(output_mvc_cfg_fn)
    except AttributeError:
        print("Error: 'animated_drawings.render' not initialized properly.")
