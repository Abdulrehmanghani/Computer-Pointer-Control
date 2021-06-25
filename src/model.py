import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model:
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

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = self.infer_network.load_network( self.model, self.device, num_requests=0)

    
    def check_model(self):
        supported_layers = self.infer_network.query_network(network=self.net, device_name="CPU")
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_inputs(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        ''' 
        input_img = cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape(1, *input_img.shape)
        input_dict = {self.input_name: input_img}
        
        return input_dict