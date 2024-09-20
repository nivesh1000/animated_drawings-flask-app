from CompatibilityChecker.drawn_humanoid_detector.mmdet_handler import MMdetHandler

class Context:
    """Context class to store system properties and model configuration
       to initialize the MMdetHandler object.
    """
    
    def __init__(self) -> None:
        self.system_properties = {
            'model_dir': 'CompatibilityChecker/drawn_humanoid_detector/',
            'gpu_id': 0
        }
        self.manifest = {
            'model': {
                'serializedFile': 'latest.pth'
            }
        }

def detection(image_bytes: bytes) -> dict:
    """
    Perform detection on the provided image bytes using the MMdetHandler.

    Args:
        image_bytes (bytes): The image data to be processed.

    Returns:
        dict: The postprocessed results from the inference or an empty dict
        if an error occurs.
    """
    try:
        # Initialize Context object to initialize model handler
        context = Context()
        handler = MMdetHandler()

        try:
            # Attempt to initialize the handler
            handler.initialize(context)
        except Exception as e:
            print(f"Error initializing handler: {e}")
            return {}

        data = [{'data': image_bytes}]

        try:
            # Attempt to preprocess, run inference, and postprocess
            preprocessed_data = handler.preprocess(data)
            inference_results = handler.inference(preprocessed_data)
            postprocessed_results = handler.postprocess(inference_results)
        except Exception as e:
            print(f"Error during detection process: {e}")
            return {}

        return postprocessed_results

    except Exception as e:
        print(f"General error in detection function: {e}")
        return {}
