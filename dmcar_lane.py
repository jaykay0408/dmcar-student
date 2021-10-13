# Auto Streering for straight lane and curve without Stop-Sign detector
# Date: Sep 1, 2021
# Jeongkyu Lee

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import argparse
import cv2
import imutils
from picar import back_wheels, front_wheels
import picar
from lane_detection import color_frame_pipeline, stabilize_steering_angle, compute_steering_angle
from lane_detection import show_image, steer_car
import time
import datetime
import queue
import threading

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the output video clip header, e.g., -v out_video")
ap.add_argument("-b", "--buffer", type=int, default=5,
	help="max buffer size")
ap.add_argument("-f", "--file",
        help="path for the training file header, e.g., -f out_file")
args = vars(ap.parse_args())

# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 2		# fast speed-> low, slow speed -> high
STOP_SEC = 0

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
    # Grab the reference to the webcam
    #vs = VideoStream(src=-1).start()
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

    SPEED = 50                  # car speed
    ANGLE = 90	                # steering wheel angle: 90 -> straight
    isMoving = False            # True: car is moving
    bw.speed = 0                # car speed
    fw.turn(ANGLE)              # steering wheel angle
    curr_steering_angle = 90    # default angle
    start_time = time.time()    # Starting time for FPS
    i = 0                       # Image sequence for FPS

    # keep looping
    while True:
        # grab the current frame
        ret, frame = vs.read()
        if frame is None:
            break

        # resize the frame
        frame = imutils.resize(frame, width=320)
        (h, w) = frame.shape[:2]

        frame_buffer.append(frame)
        blend_frame, lane_lines = color_frame_pipeline(frames=frame_buffer, \
                            solid_lines=True, \
                            temporal_smoothing=True)

        # Compute and stablize steering angle and draw it on the frame
        blend_frame, steering_angle, no_lines = compute_steering_angle(blend_frame, lane_lines)
        curr_steering_angle = stabilize_steering_angle(curr_steering_angle, steering_angle, no_lines)
        ANGLE = curr_steering_angle
        #print("Angle -> ", ANGLE)

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
