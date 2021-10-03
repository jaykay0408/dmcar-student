'''
**********************************************************************
* Filename    : test-control.py
* Description : test control for servo
* Update      : Lee    2019-02-09    New release
**********************************************************************
'''
from picar import back_wheels, front_wheels
import picar
import time

picar.setup()
db_file = "/home/pi/dmcar-student/picar/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)

bw.ready()
fw.ready()

SPEED = 0
bw.speed = SPEED

while True:
    key = input("> ")
    SPEED = 50

    if key == 'q':
        break
    elif key == 'w':
        bw.speed = SPEED
        bw.forward()
    elif key == 'x':
        bw.speed = SPEED
        bw.backward()
    elif key == 'a':
        fw.turn_left()
    elif key == 'd':
        fw.turn_right()
    elif key == 's':
        fw.turn_straight()
    elif key == 'z':
        bw.stop()

