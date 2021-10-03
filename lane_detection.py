# functions to detect lanes and helper functions
# Date: Sep 1, 2021
# Jeongkyu Lee

import numpy as np
import cv2
from Line import Line
import math
import time
from keras.models import load_model

def roi_for_edge(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen
    # 0.5 ~ 1.0 for default camera
    polygon = np.array([[
        (0, height * 0.5),
        (width, height * 0.5),
        (width, height * 1.0),
        (0, height * 1.0),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Returns resulting blend image computed as follows:

    initial_img * α + img * β + λ
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, α, img, β, λ)


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """
    # Left and Right 1/3 regision
    boundary = 0.4
    left_region_boundary = img_shape[1] * (1 - boundary)
    right_region_boundary = img_shape[1] * boundary

    # separate candidate lines according to their slope
    #for l in line_candidates:
    #    print(l.slope, l.x1, l.x2)
    #    print("")

    pos_lines = [l for l in line_candidates if (l.slope > 0 and l.x2 > right_region_boundary) or (l.slope <= 0 and l.x1 > left_region_boundary)]
    neg_lines = [l for l in line_candidates if (l.slope < 0 and l.x1 < left_region_boundary) or (l.slope >= 0 and l.x2 < right_region_boundary)]
    #print(len(neg_lines), len(pos_lines))

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])

    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)
    #print(left_lane.x1, left_lane.y1, left_lane.x2, left_lane.y2, left_lane.slope)

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)
    #print(right_lane.x1, right_lane.y1, right_lane.x2, right_lane.y2, right_lane.slope)
    #print(" ")

    return left_lane, right_lane


def get_lane_lines(color_image, solid_lines=True):
    """
    This function take as input a color road frame and tries to infer the lane lines in the image.
    :param color_image: input frame
    :param solid_lines: if True, only selected lane lines are returned. If False, all candidate lines are returned.
    :return: list of (candidate) lane lines.
    """
    # resize to 960 x 540
    #color_image = cv2.resize(color_image, (960, 540))

    # convert to grayscale
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # perform gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

    # perform edge detection
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

    mask_edge = roi_for_edge(img_edge)

    # perform hough transform
    detected_lines = hough_lines_detection(img=mask_edge,
                                           rho=1,
                                           theta=np.pi / 180,
                                           threshold=1,
                                           min_line_len=20,
                                           max_line_gap=6)
    if detected_lines is None:
        print("No detected lines")
        detected_lines = [Line(0, 0, 0, 0)]
    else:
        # convert (x1, y1, x2, y2) tuples into Lines
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]
    #print("# of detected lanes ", len(detected_lines))

    # if 'solid_lines' infer the two lane lines
    if solid_lines:
        candidate_lines = []
        for line in detected_lines:
                # consider only lines with slope between 30 and 60 degrees
                if 0.3 <= np.abs(line.slope) <= 15:
                    candidate_lines.append(line)
        # interpolate lines candidates to find both lanes
        lane_lines = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    else:
        # if not solid_lines, just return the hough transform output
        lane_lines = detected_lines
    #for l in lane_lines:
    #    print("Lane ",l.x1, l.y1, l.x2, l.y2, l.slope)
    #print(" ")

    return lane_lines


def smoothen_over_time(lane_lines):
    """
    Smooth the lane line inference over a window of frames and returns the average lines.
    """

    avg_lt_x1 = 0.0
    avg_lt_y1 = 9.0
    avg_lt_x2 = 0.0
    avg_lt_y2 = 0.0
    cnt_lt = 0

    avg_rt_x1 = 0.0
    avg_rt_y1 = 9.0
    avg_rt_x2 = 0.0
    avg_rt_y2 = 0.0
    cnt_rt = 0

    for t in range(0, len(lane_lines)):
        if lane_lines[t][0].y1 != 0.0 and lane_lines[t][0].x2 != 0.0:
            avg_lt_x1 += lane_lines[t][0].x1
            avg_lt_y1 += lane_lines[t][0].y1
            avg_lt_x2 += lane_lines[t][0].x2
            avg_lt_y2 += lane_lines[t][0].y2
            cnt_lt += 1

        if lane_lines[t][1].y1 != 0.0 and lane_lines[t][1].x2 != 0.0:
            avg_rt_x1 += lane_lines[t][1].x1
            avg_rt_y1 += lane_lines[t][1].y1
            avg_rt_x2 += lane_lines[t][1].x2
            avg_rt_y2 += lane_lines[t][1].y2
            cnt_rt += 1

    if cnt_lt != 0:
        avg_lt_line = Line(avg_lt_x1/cnt_lt, 
                           avg_lt_y1/cnt_lt,
                           avg_lt_x2/cnt_lt, 
                           avg_lt_y2/cnt_lt)
    else:
        avg_lt_line = Line(0.0, 0.0, 0.0, 0.0)

    if cnt_rt != 0:
        avg_rt_line = Line(avg_rt_x1/cnt_rt, 
                           avg_rt_y1/cnt_rt,
                           avg_rt_x2/cnt_rt, 
                           avg_rt_y2/cnt_rt)
    else:
        avg_rt_line = Line(0.0, 0.0, 0.0, 0.0)

    return avg_lt_line, avg_rt_line


def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    """
    Entry point for lane detection pipeline. Takes as input a list of frames (RGB) and returns an image (RGB)
    with overlaid the inferred road lanes. Eventually, len(frames)==1 in the case of a single image.
    """
    is_videoclip = len(frames) > 0

    img_h, img_w = frames[0].shape[0], frames[0].shape[1]

    lane_lines = []
    for t in range(0, len(frames)):
        inferred_lanes = get_lane_lines(color_image=frames[t], solid_lines=solid_lines)
        lane_lines.append(inferred_lanes)

    if temporal_smoothing and solid_lines:
        lane_lines = smoothen_over_time(lane_lines)
    else:
        lane_lines = lane_lines[0]

    # prepare empty mask on which lines are drawn
    line_img = np.zeros(shape=(img_h, img_w))

    # recover opposite angle
    if lane_lines[0].slope > 0:
        x1 = int((img_h-1-lane_lines[0].bias)/lane_lines[0].slope)
        lane_lines[0].set_coords(x1, img_h - 1, lane_lines[0].x2, lane_lines[0].y2)
    if lane_lines[1].slope < 0:
        x1 = int(-(lane_lines[1].bias/lane_lines[1].slope))
        lane_lines[1].set_coords(x1, 0, lane_lines[1].x2, lane_lines[1].y2)

    # draw lanes found
    for lane in lane_lines:
        lane.draw(line_img)

    img_masked = roi_for_edge(line_img)
    # make blend on color image
    img_color = frames[-1] if is_videoclip else frames[0]
    img_blend = weighted_img(img_masked, img_color, α=0.8, β=1., λ=0.)

    return img_blend, lane_lines


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=4, max_angle_deviation_one_lane=1):
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    #print("Proposed angle: ", new_steering_angle, "stabilized angle: ", stabilized_steering_angle)
    return stabilized_steering_angle

def compute_steering_angle(frame, lane_lines):
    left_YN = 0 != lane_lines[0].slope
    right_YN = lane_lines[1].slope < 15 and 0 != lane_lines[1].slope
    #print(lane_lines[1].slope)

    h, w, _ = frame.shape

    if (not left_YN) and (not right_YN):
        mid_position_lane = int(w/2)
        print("no lane")
        no_lines = 0
        steering_angle = -90
        return frame, steering_angle, no_lines


    if left_YN and (not right_YN):
        #mid_position_lane = w - 1
        print("left only")
        y2L = h - 1
        y2R = h - 1
        x2L = int(((y2L/2) - lane_lines[0].bias) / (lane_lines[0].slope + np.finfo(float).eps))
        x2R = w - 1
        x1L = int((y2L - lane_lines[0].bias) / (lane_lines[0].slope + np.finfo(float).eps))

        #mid_position_lane = int((x2L + x2R) / 2)
        mid_position_lane = x2L + int(w/2 - x1L)
        #print(mid_position_lane)
        x_offset = mid_position_lane - int(w/2)
        y_offset = int(h/2)
        cv2.line(frame, (int(w/2), h-1), (mid_position_lane,
                       int(h/2)), (0,255,0), 5)
        angle_to_mid_radian = math.atan(x_offset / y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
        steering_angle = angle_to_mid_deg + 90
        no_lines = 1
        return frame, steering_angle, no_lines

    if (not left_YN) and right_YN:
        #mid_position_lane = 0
        print("right only")
        #print(mid_position_lane)
        y2L = h - 1
        y2R = h - 1
        x2L = 0
        x2R = int(((y2R/2) - lane_lines[1].bias) / (lane_lines[1].slope + np.finfo(float).eps))
        x1R = int((y2R - lane_lines[1].bias) / (lane_lines[1].slope + np.finfo(float).eps))

        #mid_position_lane = int((x2L + x2R) / 2)
        mid_position_lane = x2R - int(x1R - w/2)
        #print(x2R, x1R, mid_position_lane)
        x_offset = mid_position_lane - int(w/2) 
        y_offset = int(h/2)

        cv2.line(frame, (int(w/2), h-1), (mid_position_lane,
               int(h/2)), (0,255,0), 5)

        angle_to_mid_radian = math.atan(x_offset / y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
        steering_angle = angle_to_mid_deg + 90
        no_lines = 1
        return frame, steering_angle, no_lines

    #for mid-position of lanes
    y2L = h - 1
    y2R = h - 1
    x2L = int(((y2L/2) - lane_lines[0].bias) / (lane_lines[0].slope + np.finfo(float).eps))
    x2R = int(((y2R/2) - lane_lines[1].bias) / (lane_lines[1].slope + np.finfo(float).eps))

    mid_position_lane = int((x2L + x2R) / 2)
    #print(mid_position_lane, mid_position_lane_low)
    x_offset = mid_position_lane - int(w/2)
    y_offset = int(h/2)

    cv2.line(frame, (int(w/2), h-1), (mid_position_lane,
                        int(h/2)), (0,255,0), 5)
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90
    no_lines = 2

    return frame, steering_angle, no_lines


def compute_steering_angle_model(frame, model):
    """ Find the steering angle directly based on video frame
        We assume that camera is calibrated to point to dead center
    """
    preprocessed = img_preprocess(frame)
    X = np.asarray([preprocessed])
    steering_angle = model.predict(X)[0]

    print('new steering angle: %s' % steering_angle)
    return int(steering_angle + 0.5) # round the nearest integer


def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relevant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255
    return image


# show image function
def show_image(q):
    while True:
        frame = q.get()
        # Display frame
        cv2.imshow("blend", frame)
        q.task_done()


# steering car function
def steer_car(q, frames, fw, args):
    img_seq = 0             # Writing image sequence
    while True:
        # Steering a car using ANGLE
        ANGLE = q.get()
        #print("Steering Angle ->", ANGLE)
        fw.turn(ANGLE)
        q.task_done()

        # if a file path is provided, write a training image
        frame = frames[-1]
        if args.get("file", False):
            cv2.imwrite("./model_lane_follow/train_data/%s_%03d_%03d.png" % (args["file"], img_seq, ANGLE), frame)
            img_seq += 1


"""
    # keep only region of interest by masking
    vertices = np.array([[(0,  img_h * 0.5),
                          (img_w, img_h * 0.5),
                          (img_w, img_h * 1.0),
                          (0, img_h * 1.0)]],
                        dtype=np.int32)
    #vertices = np.array([[(0, int(img_h*1.0)),
    #                      (int(img_w*0.2), int(img_h*0.6)),
    #                      (int(img_w*0.8), int(img_h*0.6)),
    #                      (img_w, int(img_h*1.0) )]],
    #                    dtype=np.int32)
    #vertices = np.array([[
    #                     (1, img_h - 1),
    #                      (1, int(img_h*0.9)),
    #                      (int(img_w*0.2), int(img_h*0.5)),
    #                      (int(img_w*0.8), int(img_h*0.5)),
    #                      (img_w - 1, int(img_h*0.9) ),
    #                      (img_w - 1, img_h -1 )]],
    #                    dtype=np.int32)
    #img_masked, _ = region_of_interest(line_img, vertices)
"""
"""
def region_of_interest(img, vertices):
    #Applies an image mask.
    #
    #Only keeps the region of the image defined by the polygon
    #formed from `vertices`. The rest of the image is set to black.

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, mask
"""

def PID(globalErrorVec, KP=1, KI=0, KD=0):
    errorVec = globalErrorVec

    if len(errorVec) == 0:
        return 0
    elif len(errorVec) == 1:
        errorVec.insert(0, [0, time.time() - 1])  # Add leading zero

    # Find necessary recent errors and time stamps
    lastError = errorVec[len(errorVec) - 1][0]
    secondToLastError = errorVec[len(errorVec) - 2][0]

    lastTime = errorVec[len(errorVec) - 1][1]
    secondToLastTime = errorVec[len(errorVec) - 2][1]
    timeOffset = 0.5

    # Calculate the three parts
    proportional = KP * lastError
    integral = KI * integralCalc(errorVec, lastTime - timeOffset)
    denominator = float(lastTime) - secondToLastTime
    if denominator == 0:
        denominator = 0.01
    derivative = KD * (float(lastError) - secondToLastError) / denominator

    # Add up the 3 parts
    output = proportional + integral + derivative
    return output


def integralCalc(dataVec, tStart):
# Intgrate from tStart to final value
    totalArea = 0
    startIndex = len(dataVec) - 1
    while dataVec[startIndex -1][1] >= tStart:
        if startIndex == 0:
            break
        else:
            startIndex -= 1

    index = startIndex

    while index < len(dataVec) - 1:
        avgHeight = (dataVec[index][0] + dataVec[index + 1][0]) / 2 #Error
        width = dataVec[index + 1][1] - dataVec[index][1] #Time
        area = avgHeight * width
        totalArea += area
        index += 1

    return totalArea

