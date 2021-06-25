'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore
from model import Model

class FacialLandMark(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU',threshold = 0.6, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        super(). __init__(model_name, device='CPU', threshold = 0.6, extensions=None)


    def predict(self, image ,face, face_cords, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_inputs(face)
        self.net.start_async( inputs =input_img_dict, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            
            result = self.net.requests[0].outputs[self.output_name]
            left_eye, right_eye, eyes_center = self.preprocess_output(result,face_cords,image, disp)
        return left_eye, right_eye, eyes_center


    def preprocess_output(self, outputs, face_cords, image, disp):
        '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    ''' 
        
        landmarks = outputs.reshape(1, 10)[0]
        height = face_cords[3] - face_cords[1]
        width = face_cords[2] - face_cords[0]
        
        x_l = int(landmarks[0] * width) 
        y_l = int(landmarks[1]  *  height)
        
        xmin_l = face_cords[0] + x_l - 30
        ymin_l = face_cords[1] + y_l - 30
        xmax_l = face_cords[0] + x_l + 30
        ymax_l = face_cords[1] + y_l + 30
        
        x_r = int(landmarks[2]  *  width)
        y_r = int(landmarks[3]  *  height)
        
        xmin_r = face_cords[0] + x_r - 30
        ymin_r = face_cords[1] + y_r - 30
        xmax_r = face_cords[0] + x_r + 30
        ymax_r = face_cords[1] + y_r + 30
        if disp:
            cv2.rectangle(image, (xmin_l, ymin_l), (xmax_l, ymax_l), (255,0,0),thickness = 3)        
            cv2.rectangle(image, (xmin_r, ymin_r), (xmax_r, ymax_r), (255,0,0),thickness = 3)
        left_eye_center =[face_cords[0] + x_l, face_cords[1] + y_l]
        right_eye_center = [face_cords[0] + x_r , face_cords[1] + y_r]      
        eyes_center = [left_eye_center, right_eye_center ]
        
        left_eye = image[ymin_l:ymax_l, xmin_l:xmax_l]
        
        right_eye = image[ymin_r:ymax_r, xmin_r:xmax_r]

        return left_eye, right_eye, eyes_center

        
