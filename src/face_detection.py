'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class FaceDetection:
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
