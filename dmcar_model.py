# Auto Streering for lane follower using NVIDIA CNN model
# $ python dmcar_model.py -b 4
# Date: Sep 1, 2021
# Jeongkyu Lee

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from picar import back_wheels, front_wheels
import picar
from Line import Line
from lane_detection import color_frame_pipeline
from lane_detection import stabilize_steering_angle
from lane_detection import compute_steering_angle 
from lane_detection import compute_steering_angle_model 
from lane_detection import PID
import time
import datetime
import math
import os
from keras.models import load_model
import tensorflow as tf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the output video clip header, e.g., -v out_video")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
ap.add_argument("-f", "--file",
        help="path for the training file header, e.g., -f out_file")
args = vars(ap.parse_args())

# define the paths to the Stop/Non-Stop NVIDIA CNN learning model
MODEL_PATH = "./models/lane.model"

# to hide warning message for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# PiCar setup
picar.setup()
db_file = "/home/pi/dmcar-student/picar/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)

bw.ready()
fw.ready()

# Time init and frame sequence
start_time = 0.0

def main():
    #load the trained NVIDIA CNN model for lane detection
    model = load_model(MODEL_PATH)

    # Grab the reference to the webcam
    #vs = VideoStream(src=0).start()
    vs = cv2.VideoCapture(-1)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # detect lane based on the last # of frames
    frame_buffer = deque(maxlen=args["buffer"])

    # initialize video writer
    writer = None

    # allow the camera or video file to warm up
    time.sleep(1.0)

    SPEED = 50
    ANGLE = 90			# steering wheel angle: 90 -> straight 
    MAX_ANGLE = 20		# Maximum angle to turn right at one time
    MIN_ANGLE = -MAX_ANGLE	# Maximum angle to turn left at one time
    isMoving = False		# True: car is moving
    posError = []		# difference between middle and car position
    bw.speed = 0		# car speed
    fw.turn(90)			# steering wheel angle
    curr_steering_angle = 90	# default angle
    i = 0			# frome sequence

    # initialize the total number of frames that *consecutively* contain
    # stop sign along with threshold required to trigger the sign alarm
    TOTAL_CONSEC = 0
    TOTAL_THRESH = 2		# fast speed-> low, slow speed -> high
    STOP_SEC = 0

    # keep looping
    while True:
        # grab the current frame
        ret, frame = vs.read()
        if frame is None:
            break

        # resize the frame (width=320 or 480)
        frame = imutils.resize(frame, width=320)
        (h, w) = frame.shape[:2]

        org_frame = frame 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_steering_angle = compute_steering_angle_model(frame, model)
        ANGLE = curr_steering_angle 

        #print("Angle -> ", ANGLE)
        cv2.imshow('blend', org_frame)

        if isMoving:
            fw.turn(ANGLE)

        # Video Writing
        if writer is None:
            if args.get("video", False):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                path = args["video"] + "_" + datestr + ".avi"
                writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h), True)

        # if a video path is provided, write a video clip
        if args.get("video", False):
            writer.write(org_frame)

        # if a file path is provided, write a training image
        if args.get("file", False):
            cv2.imwrite("./model_lane_follow/train_data/%s_%03d_%03d.png" % (args["file"], i, ANGLE), org_frame)
            i += 1

        keyin = cv2.waitKey(1) & 0xFF
        keycmd = chr(keyin)

        # if the 'q' key is pressed, end program
        # if the 'w' key is pressed, moving forward
        # if the 'x' key is pressed, moving backword
        # if the 'a' key is pressed, turn left
        # if the 'd' key is pressed, turn right
        # if the 's' key is pressed, straight
        # if the 'z' key is pressed, stop a car
        if keycmd == 'q':
    	    break
        elif keycmd == 'w':
    	    isMoving = True
    	    bw.speed = SPEED
    	    bw.forward()
        elif keycmd == 'x':
    	    bw.speed = SPEED
    	    bw.backward()
        elif keycmd == 'a':
    	    ANGLE -= 5
    	    if ANGLE <= 45:
    	        ANGLE = 45
    	    #fw.turn_left()
    	    fw.turn(ANGLE)
        elif keycmd == 'd':
    	    ANGLE += 5
    	    if ANGLE >= 135:
    	        ANGLE = 135
    	    #fw.turn_right()
    	    fw.turn(ANGLE)
        elif keycmd == 's':
    	    ANGLE = 90
    	    #fw.turn_straight()
    	    fw.turn(ANGLE)
        elif keycmd == 'z':
    	    isMoving = False
    	    bw.stop()
    
    # if we are not using a video file, stop the camera video stream
    if writer is not None:
        writer.release()
    vs.release()

    # initialize picar
    bw.speed = 0
    fw.turn(90)
	    
    # close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
