# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdet.apis import inference_detector, init_detector


class MMdetHandler(BaseHandler):
    threshold = 0.5

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_detector(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        image = []
        imagebytes = data[0].get('data')
        imagemmcv = mmcv.imfrombytes(imagebytes)
        image.append(imagemmcv)
        return image

    def inference(self, data):
        results = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        output = []
        bbox_result, _= data[0]
        bbox_coords=bbox_result[0][0][:-1]
        score=bbox_result[0][0][-1]
        if score >= self.threshold:
            output.append({
                'bbox': bbox_coords,
                'score': score
            })
        return output
