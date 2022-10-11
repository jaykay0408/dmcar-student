# USAGE
# python stop_detector.py 

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os

# define the paths to the Not STOP-NoT-STOP deep learning model
MODEL_PATH = "./models/stop_not_stop.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# initialize the total number of frames that *consecutively* contain
# stop sign along with threshold required to trigger the sign alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20

# initialize is the sign alarm has been triggered
STOP = False

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
vs = cv2.VideoCapture(-1)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 320 pixels
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=320)

	# prepare the image to be classified by our deep learning network
	image = frame[60:120, 240:320]
	image = cv2.resize(image , (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image and initialize the label and
	# probability of the prediction
	(notStop, stop) = model.predict(image)[0]
	label = "Not Stop"
	proba = notStop

	# check to see if stop sign was detected using our convolutional
	# neural network
	if stop > notStop:
		# update the label and prediction probability
		label = "Stop"
		proba = stop

		# increment the total number of consecutive frames that
		# contain stop
		TOTAL_CONSEC += 1

		# check to see if we should raise the stop sign alarm
		if not STOP and TOTAL_CONSEC >= TOTAL_THRESH:
			# indicate that stop has been found
			STOP = True
			print("Stop Sign...")

	# otherwise, reset the total number of consecutive frames and the
	# stop sign alarm
	else:
		TOTAL_CONSEC = 0
		STOP = False

	# build the label and draw it on the frame
	label = "{}: {:.2f}%".format(label, proba * 100)
	frame = cv2.putText(frame, label, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	frame = cv2.rectangle(frame, (240, 60),(320,120), (0,0,255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.release()

