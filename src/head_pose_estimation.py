'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
from math import cos, sin, pi
import cv2
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU',threshold = 0.6, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_bin = model_name+'.bin'
        self.model_xml = model_name+'.xml'
        self.device = device
        self.threshold = threshold
        self.infer_network = IECore()
        try:
            self.model = IENetwork(model= self.model_xml, weights= self.model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = self.infer_network.load_network( self.model, self.device, num_requests=0)

    def predict(self, image,face, face_cords, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_input(face)
        self.net.start_async( inputs =input_img_dict, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            #print(self.output_name)
            result=[]
            for o in self.model.outputs:
                result.append(self.net.requests[0].outputs[o])
            
            #result = output[self.output_name]
            headpose_angles = self.preprocess_output(result,image,face_cords, disp)
        
        return headpose_angles

    def check_model(self):
        supported_layers = self.infer_network.query_network(network=self.net, device_name="CPU")
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        input_img = cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape(1, *input_img.shape)
        
        input_dict = {self.input_name: input_img}
        
        return input_dict
    
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
