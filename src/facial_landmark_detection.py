'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class FacialLandMark:
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

    def predict(self, image ,face, face_cords, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_input(face)
        self.net.start_async( inputs =input_img_dict, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            
            result = self.net.requests[0].outputs[self.output_name]
            left_eye, right_eye, eyes_center = self.preprocess_output(result,face_cords,image, disp)
        return left_eye, right_eye, eyes_center

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
        input_img = cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape(1, *input_img.shape)
        
        input_dict = {self.input_name: input_img}
        
        return input_dict

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
            cv2.rectangle(image, (xmin_l, ymin_l), (xmax_l, ymax_l), (255,0,0), 2)        
            cv2.rectangle(image, (xmin_r, ymin_r), (xmax_r, ymax_r), (255,0,0), 2)
        left_eye_center =[face_cords[0] + x_l, face_cords[1] + y_l]
        right_eye_center = [face_cords[0] + x_r , face_cords[1] + y_r]      
        eyes_center = [left_eye_center, right_eye_center ]
        
        left_eye = image[ymin_l:ymax_l, xmin_l:xmax_l]
        
        right_eye = image[ymin_r:ymax_r, xmin_r:xmax_r]

        return left_eye, right_eye, eyes_center

        
