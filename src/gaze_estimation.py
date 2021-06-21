'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold = 0.6, extensions=None):
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

        self.input_name= next(iter(self.model.inputs))
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

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_input(head_pose_angle, left_eye_image, right_eye_image)
        self.net.start_async( inputs =input_img_dict, request_id=0)
        if self.net.requests[0].wait(-1) == 0:
            result = self.output[self.output_name]
            print(result)
        #gaze_vector = preprocess_outputs(result)
        
        #return gaze_vector

    def check_model(self):
        supported_layers = self.infer_network.query_network(network=self.net, device_name="CPU")
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, head_pose_angle, left_eye_image, right_eye_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        print(type(left_eye_image),left_eye_image.shape)
        input_left_eye = cv2.resize(left_eye_image, (self.input_shape[3],self.input_shape[2]))
        input_right_eye = cv2.resize(right_eye_image, (self.input_shape[3],self.input_shape[2]))
        input_left_eye = input_left_eye.transpose((2, 0, 1))
        input_right_eye = input_right_eye.transpose((2, 0, 1))
        input_left_eye = input_left_eye.reshape(1, *input_left_eye.shape)
        input_right_eye = input_right_eye.reshape(1, *input_right_eye.shape)
        input_dict = {self.input_name: head_pose_angle,self.input_name: input_left_eye,self.input_name: input_right_eye}
        
        return input_dict

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #self.net.requests[0].outputs[o]
