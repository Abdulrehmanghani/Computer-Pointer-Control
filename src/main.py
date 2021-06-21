import sys
import cv2
import time
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder

from face_detection import FaceDetection
from facial_landmark_detection import FacialLandMark
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fd_model", required=True, type=str,
                        help="Path to an xml file of the Face Detection model.")
   
    parser.add_argument("-hp", "--hp_model", required=True, type=str,
                        help="Path to an xml file of the Head Pose Estimation model.")
                        
    parser.add_argument("-fl", "--fl_model", required=True, type=str,
                        help="Path to an xml file of the Facial Landmarks Detection model.")
                        
    parser.add_argument("-ge", "--ge_model", required=True, type=str,
                        help="Path to an xml file of the Gaze Estimation model.")
                       
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="CAM or path to image or video file.")
    
    # parser.add_argument("-dis", "--display", required=False, default=True, type=str,
    #                     help="Flag to display the outputs of the intermediate models")

    parser.add_argument("-d", "--device", required=False, default="CPU", type=str,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD.")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    
    # parser.add_argument("-p", "--mouse_precision", required=False, default='high', type=str,
    #                     help="Set the precision for mouse movement: high, low, medium.")
                        
    # parser.add_argument("-sp", "--mouse_speed", required=False, default='fast', type=str,
    #                     help="Set the speed for mouse movement: fast, slow, medium.")
                        
    return parser

def handle_input_type(input_stream):
    '''
     Handle image, video or webcam
    '''
    
    # Check if the input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        input_type = 'image'
        
    # Check if the input is a webcam
    elif input_stream == 'CAM':
        input_type = 'cam'
        
    # Check if the input is a video    
    elif input_stream.endswith('.mp4'):
        input_type = 'video'
    else: 
        log.error('Please enter a valid input! .jpg, .png, .bmp, .mp4, CAM')
        sys.exit()    
    return input_type

def infer_on_stream(args):
    """
    Initialize the inference networks, stream video to network,
    and output stats, video and control the mouse pointer.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    # Initialise the classes
    face_detection = FaceDetection(args.fd_model, args.device)
    landmark_detection = FacialLandMark(args.fl_model, args.device)
    headpose_estimation = HeadPoseEstimation(args.hp_model, args.device)
    gaze_estimation = GazeEstimation(args.ge_model, args.device)
    
    start_load = time.time()
    
    # Load the models 
    face_detection.load_model()
    landmark_detection.load_model()
    headpose_estimation.load_model()
    gaze_estimation.load_model()

    end_load = time.time() -  start_load 
    
    # Handle the input stream
    input_type = handle_input_type(args.input)
    
    # Initialise the InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    # Load the video capture
    feed.load_data()
    
    start_inf = time.time()
    
    # Read from the video capture 
    for flag, frame in feed.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Run inference on the models     
        face, face_coords = face_detection.predict(frame)
        left_eye, right_eye, eyes_center = landmark_detection.predict(face, face_coords)
        headpose_angles = headpose_estimation.predict(face, face_coords)
        gaze_estimation.predict(left_eye, right_eye, headpose_angles)
        cv2.imwrite("face.jpg",face)
        ## If no face detected move back to the top of the loop
        if len(face_coords) == 0:
            log.error("No face detected.")
            continue
            
        if key_pressed == 27:
            break
       
       # Display the resulting frame
        cv2.imshow('Visualization', cv2.resize(face,(600,400)))
     
    end_inf = time.time() - start_inf
    
    print("Total loading time: {}\nTotal inference time: {} ".format(end_load, end_inf))
    
    # Release the capture
    feed.close()
    # Destroy any OpenCV windows
    cv2.destroyAllWindows
 
def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    #Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()