# Auto Streering with Stop-NoStop model using pre-trained MobileNet V2
# Generate Model using Colab (Classification)
# $ python dmcar_coral.py -b 4 
# Date: Sep 1, 2021
# Jeongkyu Lee

# import the necessary packages
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
from PIL import Image
from picar import back_wheels, front_wheels
import picar
from lane_detection import color_frame_pipeline, stabilize_steering_angle, compute_steering_angle
from lane_detection import show_image, steer_car
import time
import datetime
import queue
import threading
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the output video clip header, e.g., -v out_video")
ap.add_argument("-b", "--buffer", type=int, default=5,
	help="max buffer size")
ap.add_argument("-f", "--file",
        help="path for the training file header, e.g., -f out_file")
args = vars(ap.parse_args())

# define the paths to the Stop/Non-Stop TPU lite deep learning model
MODEL_PATH = "./models/stop_not_stop.tflite"
LABEL_PATH = "./models/stop_not_stop.txt"

# to hide warning message for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Start Queues
show_queue = queue.Queue()
steer_queue = queue.Queue()

# PiCar setup
picar.setup()
db_file = "/home/pi/dmcar-student/picar/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)

# Time init and frame sequence
start_time = 0.0

def main():
    # initialize the labels dictionary
    print("[INFO] parsing class labels...")
    labels = read_label_file(LABEL_PATH)

    # load the Google Coral classification model
    print("[INFO] loading Coral model...")
    interpreter1 = make_interpreter(MODEL_PATH)
    interpreter2 = make_interpreter(MODEL_PATH)
    interpreter1.allocate_tensors()
    interpreter2.allocate_tensors()
    _, height, width, _ = interpreter1.get_input_details()[0]['shape']

    # Grab the reference to the webcam
    # vs = VideoStream(src=-1).start()
    vs = cv2.VideoCapture(-1)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # detect lane based on the last # of frames
    frame_buffer = deque(maxlen=args["buffer"])

    # initialize video writer
    writer = None

    # allow the camera or video file to warm up
    time.sleep(1.0)

    bw.ready()
    fw.ready()

    # Setup Threading
    threading.Thread(target=show_image, args=(show_queue,), daemon=True).start()
    threading.Thread(target=steer_car, args=(steer_queue, frame_buffer, fw, args), daemon=True).start()

    SPEED = 40                  # car speed
    ANGLE = 90			# steering wheel angle: 90 -> straight
    isMoving = False		# True: car is moving
    isStart = False             # True: car is started
    bw.speed = 0		# car speed
    fw.turn(ANGLE)		# steering wheel angle
    curr_steering_angle = 90    # default angle
    start_time = time.time()    # Starting time for FPS
    i = 0                       # Image sequence for FPS

    # initialize the total number of frames that *consecutively* contain
    # stop sign along with threshold required to trigger the sign alarm
    TOTAL_CONSEC = 0
    TOTAL_THRESH = 1            # fast speed-> low, slow speed -> high
    STOP_SEC = 0
    AFTER_STOP_SEC = 0          # skip frames after detection
    STOP = False                # If STOP is detected
    LAST_STOP = False

    # keep looping
    while True:
        stop1 = stop2 = notStop1 = notStop2 = 0.0
        # grab the current frame
        ret, frame = vs.read()
        if frame is None:
            break

        # resize the frame (width=320 or width=480)
        frame = imutils.resize(frame, width=320)
        (h, w) = frame.shape[:2]

        # crop for model, i.e., traffic sign location
        # can be adjusted based on camera angle
        image1 = frame[int(h*0.2):int(h*0.35), int(w*0.75):int(w*0.875)]
        image1 = cv2.resize(image1, (width, height))
        image2 = frame[int(h*0.2):int(h*0.35), int(w*0.875):w]
        image2 = cv2.resize(image2, (width, height))

        frame_buffer.append(frame)
        blend_frame, lane_lines = color_frame_pipeline(frames=frame_buffer, \
                            solid_lines=True, \
                            temporal_smoothing=True)

        # Compute and stablize steering angle and draw it on the frame
        blend_frame, steering_angle, no_lines = compute_steering_angle(blend_frame, lane_lines)
        curr_steering_angle = stabilize_steering_angle(curr_steering_angle, steering_angle, no_lines)
        ANGLE = curr_steering_angle
        #print("Angle -> ", ANGLE)

        # classify the input image and initialize the label and
        # probability of the prediction
        common.set_input(interpreter1, image1)
        common.set_input(interpreter2, image2)

        interpreter1.invoke()
        interpreter2.invoke()

        results1 = classify.get_classes(interpreter1, top_k=1)
        results2 = classify.get_classes(interpreter2, top_k=1)

        if len(results1) > 0:
            (classID1, score1) = results1[0]
            if classID1 == 1:
                stop1 = score1
            else:
                notStop1 = score1

        if len(results2) > 0:
            (classID2, score2) = results2[0]
            if classID2 == 1:
                stop2 = score2
            else:
                notStop2 = score2
        #print(classID1, score1, classID2, score2)

        label = "No Stop"
        if ((stop1 > notStop1) or (stop2 > notStop2)):
            label = "Stop"
        else:
            label = "No Stop"

        proba1 = stop1
        proba2 = stop2

        if LAST_STOP:
            AFTER_STOP_SEC += 1
            print("No stop detection ...", AFTER_STOP_SEC)
            if AFTER_STOP_SEC >= 20:
                LAST_STOP = False
                AFTER_STOP_SEC = 0

        # check to see if stop sign was detected using our convolutional
        # neural network
        if ((stop1 > notStop1) or (stop2 > notStop2)) and not LAST_STOP and isStart:
            # update the label and prediction probability
            label = "Stop"
            # increment the total number of consecutive frames that
            # contain stop
            if isMoving:
                TOTAL_CONSEC += 1

            # check to see if we should raise the stop sign alarm
            if isMoving and not STOP and TOTAL_CONSEC >= TOTAL_THRESH:
                # indicate that stop has been found
                STOP = True
                bw.stop()
                isMoving = False
                STOP_SEC += 1
                print("Stop Sign..." + str(STOP_SEC))
            elif STOP and STOP_SEC <= 10:
                bw.stop()
                isMoving = False
                STOP_SEC += 1
                print("Stop is going on..." + str(STOP_SEC))
            elif STOP and STOP_SEC > 10:
                STOP = False
                LAST_STOP = True
                bw.speed = SPEED
                bw.forward()
                isMoving = True
                STOP_SEC = 0
                TOTAL_CONSEC = 0
                print("Stop is done...Going", STOP_SEC)

        # otherwise, reset the total number of consecutive frames and the
        # stop sign alarm
        else:
            TOTAL_CONSEC = 0
            STOP = False
            bw.forward()
            isMoving = True
            STOP_SEC = 0

        # build the label and draw it on the frame
        label = "{}: {:.2f}% {:.2f}%".format(label, proba1 * 100, proba2 * 100)
        #blend_frame = cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)
        blend_frame = cv2.putText(blend_frame, label, (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        blend_frame = cv2.rectangle(blend_frame, (int(w*0.75), int(h*0.2)),
                                    (int(w*0.875), int(h*0.35)), (0,0,255), 2)
        blend_frame = cv2.rectangle(blend_frame, (int(w*0.875), int(h*0.2)),
                                    (w, int(h*0.35)), (0,0,255), 2)
        show_queue.put(blend_frame)

        if isMoving:
            steer_queue.put(ANGLE)

        # Video Writing
        if writer is None:
            if args.get("video", False):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                path = args["video"] + "_" + datestr + ".avi"
                writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h), True)

        # if a video path is provided, write a video clip
        if args.get("video", False):
            writer.write(blend_frame)

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
            # Calculate and display FPS
            end_time = time.time()
            print( i / (end_time - start_time))
            break
        elif keycmd == 'w':
            isMoving = True
            isStart = True
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
