'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore
from model import Model

class GazeEstimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold = 0.6, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        super(). __init__( model_name, device='CPU', threshold = 0.6, extensions=None)

        self.input_name=[i for i in self.model.inputs.keys()]
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=[o for o in self.model.outputs.keys()]   

  
    def predict(self, image, left_eye_image, right_eye_image, head_pose_angles, eyes_center, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_left_eye = self.preprocess_input( left_eye_image)
        input_right_eye = self.preprocess_input( right_eye_image)
        head_pose_angles = np.asarray(head_pose_angles)
        head_pose_angles = head_pose_angles.transpose((2, 0, 1))
        head_pose_angles = head_pose_angles[0]
        head_pose_angles = head_pose_angles.transpose((1, 0))
        self.net.start_async( inputs ={'left_eye_image': input_left_eye,'right_eye_image': input_right_eye,'head_pose_angles': head_pose_angles}, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            
            outputs = self.net.requests[0].outputs[self.output_name[0]]
            
            gaze_vector = self.preprocess_output(image, outputs, eyes_center, disp)
        #gaze_vector = preprocess_outputs(result)
        
        return gaze_vector


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_img = cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape(1, *input_img.shape)
        return input_img

    def preprocess_output(self, image, outputs,eyes_center, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector =  outputs[0]
        
        if disp:
        
            left_eye_center_x = int(eyes_center[0][0])
            left_eye_center_y = int(eyes_center[0][1])
            
            right_eye_center_x = int(eyes_center[1][0])
            right_eye_center_y = int(eyes_center[1][1])
            
            cv2.arrowedLine(image, (left_eye_center_x, left_eye_center_y), (left_eye_center_x + int(gaze_vector[0] * 100), left_eye_center_y + int(-gaze_vector[1] * 100)), (0,0,255), 3)
            cv2.arrowedLine(image, (right_eye_center_x, right_eye_center_y), (right_eye_center_x + int(gaze_vector[0] * 100), right_eye_center_y + int(-gaze_vector[1] * 100)), (0,0,255), 3)

        return gaze_vector 
