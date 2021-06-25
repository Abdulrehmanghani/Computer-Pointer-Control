'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
from math import cos, sin, pi
import cv2
from openvino.inference_engine import IENetwork, IECore
from model import Model

class HeadPoseEstimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU',threshold = 0.6, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        super(). __init__( model_name, device='CPU', threshold = 0.6, extensions=None)


    def predict(self, image,face, face_cords, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_inputs(face)
        self.net.start_async( inputs =input_img_dict, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            #print(self.output_name)
            result=[]
            for o in self.model.outputs:
                result.append(self.net.requests[0].outputs[o])
            
            #result = output[self.output_name]
            headpose_angles = self.preprocess_output(result,image,face_cords, disp)
        
        return headpose_angles

    
    def draw_outputs(self, image, head_angle ,face_cords): 
        '''
        Draw model output on the image.
        '''
        
        cos_r = cos(head_angle[2] * pi / 180)
        sin_r = sin(head_angle[2] * pi / 180)
        sin_y = sin(head_angle[0] * pi / 180)
        cos_y = cos(head_angle[0] * pi / 180)
        sin_p = sin(head_angle[1] * pi / 180)
        cos_p = cos(head_angle[1] * pi / 180)
        
        x = int((face_cords[0] + face_cords[2]) / 2)
        y = int((face_cords[1] + face_cords[3]) / 2)
        
        cv2.line(image, (x,y), (x+int(70*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(70*cos_p*sin_r)), (255, 0, 0), 2)
        cv2.line(image, (x, y), (x+int(70*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(70*cos_p*cos_r)), (0, 0, 255), 2)
        cv2.line(image, (x, y), (x + int(70*sin_y*cos_p), y + int(70*sin_p)), (0, 255, 0), 2)
       
        return image

    def preprocess_output(self, outputs,image, face_cords, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        head_angles = []
        
        for i in range(len(outputs)):
            head_angles.append(outputs[i])
        if disp:
            out_image = self.draw_outputs(image,  head_angles, face_cords)
        return head_angles
