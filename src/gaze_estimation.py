'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model=core.read_network(model=model_structure, weights=model_weights)
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
        self.net = self.core.load_network(network= self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img_dict = self.preprocess_input(self, head_pose_angle, left_eye_image, right_eye_image)
        output = net.infer(input_img_dict)
        result = self.output[self.output_name]
        cords = preprocess_outputs(result,image)
        
        return face_image, cords

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, head_pose_angle, left_eye_image, right_eye_image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        input_left_eye=cv2.resize(left_eye_image, (self.input_shape[3],self.input_shape[2]))
        input_right_eye=cv2.resize(right_eye_image, (self.input_shape[3],self.input_shape[2]))
        input_left_eye = input_left_eye.transpose((2, 0, 1))
        input_right_eye = input_right_eye.transpose((2, 0, 1))
        input_left_eye = input_left_eye.reshape(1, *input_left_eye.shape)
        input_right_eye = input_right_eye.reshape(1, *input_right_eye.shape)
        input_dict{self.input_name: head_pose_angle,self.input_name: input_left_eye,self.input_name: input_right_eye}
        
        return input_dict

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
