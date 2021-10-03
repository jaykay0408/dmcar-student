'''
**********************************************************************
* Filename    : test-servo.py
* Description : test for server
* Update      : Lee    2019-02-08    New release
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
 
SPEED = 50

# ============== Back wheels =============
# 'bwready':
#bw.ready()

for i in range(10, 101,10):
    bw.speed = i
    bw.forward()
    time.sleep(2)
	
# 'forward':
bw.speed = SPEED
bw.forward()
time.sleep(1)
		
# 'backward':
bw.speed = SPEED
bw.backward()
time.sleep(1)

# 'stop':
bw.stop()

# ============== Front wheels =============
# Turn Left
fw.turn_left()
time.sleep(1)

# Straight
fw.turn_straight()
time.sleep(1)

# Turn Right
fw.turn_right()
time.sleep(1)

# Straight
fw.turn_straight()

# Angle 45 degree to 135 degree
for i in range(45, 135, 5):
    print(i)
    fw.turn(i)
    time.sleep(1)

fw.turn(90)
