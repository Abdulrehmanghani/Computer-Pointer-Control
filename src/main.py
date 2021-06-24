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

from mouse_controller import MouseController

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
    
    parser.add_argument("-disp", "--display", required=False, default=True, type=str,
                        help="Flag to display the outputs of the intermediate models")

    parser.add_argument("-d", "--device", required=False, default="CPU", type=str,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD.")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
                       
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
    try:
        face_detection = FaceDetection(args.fd_model, args.device)
        landmark_detection = FacialLandMark(args.fl_model, args.device)
        headpose_estimation = HeadPoseEstimation(args.hp_model, args.device)
        gaze_estimation = GazeEstimation(args.ge_model, args.device)
    except:
        log.error('Please enter a valid model file address')    
        sys.exit()

    
    mouse_controller = MouseController('medium', 'fast')
    log_object = log.getLogger()

    start_load = time.time()
    
    # Load the models 
    face_detection.load_model()
    landmark_detection.load_model()
    headpose_estimation.load_model()
    gaze_estimation.load_model()
    log_object.error("Models loaded: time: {:.3f} ms".format((time.time() - start_load) * 1000))
    end_load = time.time() -  start_load 
    
    # Handle the input stream
    input_type = handle_input_type(args.input)
    
    # Initialise the InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    # Load the video capture
    feed.load_data()
    frame_count = 0
    start_inf = time.time()
    
    # Read from the video capture 
    for flag, frame in feed.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        frame_count += 1
        try:
            # Run inference on the models     
            face, face_cords = face_detection.predict(frame, args.display)
            ## If no face detected move back to the top of the loop
            if len(face_cords) == 0:
                log_object.error("No face found")
                continue
            if key_pressed == 27:
                break
            left_eye, right_eye, eyes_center = landmark_detection.predict(frame, face, face_cords, args.display)
            headpose_angles = headpose_estimation.predict(frame, face, face_cords, args.display)
            gaze_vector = gaze_estimation.predict(frame, left_eye, right_eye, headpose_angles, eyes_center, args.display)
            mouse_controller.move(gaze_vector[0], gaze_vector[1])
        except Exception as e:
            print(str(e) + " for frame " + str(frame_count))
            continue
        # Display the resulting frame
        if args.display:
            cv2.imshow('Visualization', cv2.resize(frame,(600,400)))
     
    end_inf = time.time() - start_inf
    log_object.error("Total loading time: {}\nTotal inference time: {}\nFPS: {}".format(end_load, end_inf,frame_count/end_inf))
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