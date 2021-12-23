# Auto Streering with traffic sign model using EfficientDet object detection
# Generate Model using Colab
# $ python dmcar_coco.py -b 4
# 
# Use "coco_model.tflite" and "coco_labels.txt" for COCO existing model
# Date: Sep 1, 2021
# Jeongkyu Lee

# import the necessary packages
from pycoral.adapters import common
from pycoral.adapters import detect
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

# define the paths to the Stop/Non-Stop Keras deep learning model
#MODEL_PATH = "./models/traffic_sign.tflite"
#MODEL_PATH = "./models/traffic_sign_edgetpu.tflite"
#LABEL_PATH = "./models/traffic_sign.txt"
MODEL_PATH = "./models/coco_model.tflite"    # General CoCo Model
LABEL_PATH = "./models/coco_labels.txt"      # CoCo Model Label

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
#i = 0

def main():
    # initialize the labels dictionary
    print("[INFO] parsing class labels...")
    labels = read_label_file(LABEL_PATH)

    # load the COCO object detection model
    print("[INFO] loading Coral model...")
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

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
    ANGLE = 90                  # steering wheel angle: 90 -> straight
    isMoving = False	        # True: car is moving
    isStart = False             # True: car is started
    bw.speed = 0	        # car speed
    fw.turn(ANGLE)	        # steering wheel angle
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

        # crop for COCO model (i.e., whole frame)
        image = frame[0:h, 0:w]
        image = Image.fromarray(image)

        _, scale = common.set_resized_input(interpreter, image.size, 
                        lambda size: image.resize(size, Image.ANTIALIAS))

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
        label = "No Stop"
        proba = 0.0

        interpreter.invoke()
        results = detect.get_objects(interpreter, 0.3, scale)

        for r in results:
            # extract the bounding box and box and predicted class label
            box = r.bbox
            (startX, startY, endX, endY) = box
            label = labels[r.id]

            # draw the bounding box and label on the image
            cv2.rectangle(blend_frame, (int(startX), int(startY)),
                    (int(endX), int(endY)), (0, 255, 0), 2)
            y = int(startY) - 15 if int(startY) - 15 > 15 else int(startY) + 15
            text = "{}: {:.2f}%".format(str(r.id)+"."+ label, r.score * 100)
            cv2.putText(blend_frame, text, (int(startX), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Replace r.id number based on label file
            if r.id == 5:
            #if r.id == 12:
                proba = r.score
                stop1 = proba
                print(text)

        if LAST_STOP:
            AFTER_STOP_SEC += 1
            print("No stop detection ...", AFTER_STOP_SEC)
            if AFTER_STOP_SEC >= 20:
                LAST_STOP = False
                AFTER_STOP_SEC = 0

        # check to see if stop sign was detected using model
        if stop1 > 0.0 and not LAST_STOP and isStart:
            # update the label and prediction probability
            # label = "Stop"
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

        show_queue.put(blend_frame, frame)

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
