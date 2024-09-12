from drawn_humanoid_detector.mmdet_handler import MMdetHandler

class Context:
    def __init__(self):
        self.system_properties = {
            'model_dir': 'CompatibilityChecker/drawn_humanoid_detector/' ,
            'gpu_id': 0
        }
        self.manifest = {
            'model': {
                'serializedFile': 'latest.pth'
            }
        }
def detection(image_bytes):
    #Initialize Context object to initialize model handler
    context = Context()
    handler = MMdetHandler()
    handler.initialize(context)
    

    data = [{'data': image_bytes}]

    preprocessed_data = handler.preprocess(data)

    inference_results = handler.inference(preprocessed_data)

    postprocessed_results = handler.postprocess(inference_results)

    return postprocessed_results
  
