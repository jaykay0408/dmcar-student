# dmcar-student
- dmcar.py              : main file to control autonomous car with Stop detector
                        : $ python dmcar.py -b 4
                        : MODEL: stop_not_stop.model

- dmcar_lane.py         : main file to control autonomous car (lane only)
                        : $ python dmcar_lane.py -b 4

- dmcar_model.py        : lane follower using NVIDIA CNN model
                        : $ python dmcar_model.py -b 4
                        : MODEL: lane_navigation.model

- dmcar_coral.py        : Stop NoStop model using pre-trained MobileNet V2
                        : $ python dmcar_coral.py -b 4
                        : MODEL: stop_not_stop.tflite
                        : LABEL: stop_not_stop.txt

- dmcar_coco.py         : Traffic Sign model using EfficientDet object detection
                        : $ python dmcar_coco.py -b 4
                        : MODEL: traffic_sign.tflite
                        : LABEL: traffic_sign.txt

- lane_detection.py     : functions to detect lanes and helper functions

- Line.py               : Line class

- stop_detector.py      : test program for stop/non-stop CNN model
                        : $ python stop_detector.py
- test-control.py       : test program for controlling servos

- test-servo.py         : test program for servo

- picar                 : directory for servoes (2 back wheels and 1 front
                        : wheels) in a car
                        : mostly doesn't have to change

- model_stop_not_stop   : directory for building stop_not_stop.model for
                        : dmcar.py

- model_traffic_sign    : directory for building dataset of traffic sign
                        : using Colab
                        : dmcar_coco.py

- model_lane_follow     : directory for building dataset of lane follower
                        : dmcar_model.py

- Stop No Stop model    : Google Colab to create Stop NoStop model
                        : Classification model
https://colab.research.google.com/drive/1s-1x8KnNcI5fphLC_yqo7BxsoSWow-1i?usp=sharing

- Traffic Signs model   : Google Colab to create Traffic Sign model
                        : Object Detection model
https://colab.research.google.com/drive/1TXbcYvZ4TAkbzwqfQ_4KVvp4z51HztDO?usp=sharing
