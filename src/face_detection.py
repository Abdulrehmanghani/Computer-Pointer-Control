'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore
from model import Model

class FaceDetection(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold = 0.6, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        ''' 
        super(). __init__( model_name, device='CPU', threshold = 0.6, extensions=None)


    def predict(self, image,disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_inputs(image)
        self.net.start_async( inputs =input_img_dict, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            
            result = self.net.requests[0].outputs[self.output_name]
            #result = output[self.output_name]
            cords , face_image = self.preprocess_output(result, image, disp)
            
        return face_image, cords

    def preprocess_output(self, outputs, image, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        cords = []
        for box in outputs[0][0]: # Output shape is [1, 1, N, 7]
            conf = box[2]
            if conf >= self.threshold:
                x1 = int(box[3] * image.shape[1])
                y1 = int(box[4] * image.shape[0])
                x2 = int(box[5] * image.shape[1])
                y2 = int(box[6] * image.shape[0])
                face = image[y1:y2, x1:x2]
                cords.append((x1,y1,x2,y2))
                if disp:
                    cv2.rectangle(image, (x1-20, y1), (x2+20, y2), (255, 0, 0),thickness = 5)
                cords = np.asarray(cords[0], dtype=np.int32)
        return cords ,face
