import cv2
import numpy as np
#from hand_recognition import *
#import cv2
#import numpy as np
import math
import matplotlib as plt

def capture_histogram(source):
    cap = cv2.VideoCapture(source)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (1000, 600))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Place region of the hand inside box & press `A`',
                    (5, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (500, 100), (580, 180), (10, 205, 10), 2)
        box = frame[105:175, 505:575]

        cv2.imshow("Capture Histogram", frame)
        key = cv2.waitKey(10)
        if key == ord('a'):
            object_color = box
            cap.release()
            cv2.destroyAllWindows()
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    if(object_color.any()):
        object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
        object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
                               [12, 15], [0, 180, 0, 256])

        cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
        return object_hist
    else:
        return None

def locate_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    if(object_hist.any()):
        object_segment = cv2.calcBackProject(
            [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

        _, segment_thresh = cv2.threshold(
            object_segment, 70, 255, cv2.THRESH_BINARY)

        # apply some image operations to enhance image
        kernel = None
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        filtered = cv2.filter2D(segment_thresh, -1, disc)

        eroded = cv2.erode(filtered, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # masking
        masked = cv2.bitwise_and(frame, frame, mask=closing)

        return closing, masked, segment_thresh,hsv_frame,object_segment,disc,filtered,eroded,dilated,closing
    else:
        return None 

def detect_hand(frame, hist):
    return_value = {}

    detected_hand, masked, raw ,hsv,obj_seg,disc,filtere,er,di,cl= locate_object(frame, hist)
    return_value["binary"] = detected_hand
    return_value["masked"] = masked
    return_value["raw"] = raw
    return_value["hsv"]=hsv
    return_value["obj_seg"]=obj_seg
    return_value["disc"]=disc
    return_value["filtere"]=filtere
    return_value["er"]=er
    return_value["di"]=di
    return_value["cl"]=cl
    #return return_value
    contours,hierarchy = cv2.findContours(detected_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Contours ", contours)
    #print(type(contours))
    #print(type(contours[0]))
    #print(contours[0][0][0][1])
    #print ((contours[0]))
    palm_area = 0
    flag = None
    cnt = None

    # find the largest contour
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > palm_area:
            palm_area = area
            flag = i

    # we want our contour to have a minimum area of 10000
    # this number might be different depending on camera, distance of hand
    # from screen, etc.
    if flag is not None and palm_area > 10000:
        cnt = contours[flag]
        return_value["contours"] = cnt
        cpy = frame.copy()
        cv2.drawContours(cpy, [cnt], 0, (0, 255, 0), 2)
        return_value["boundaries"] = cpy
        return True, return_value
    else:
        return False, return_value
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
def extract_fingertips(hand):
    cnt = hand["contours"]
    points = []
    hull = cv2.convexHull(cnt, returnPoints=False)
    #print (len(hull))
    #print ("Hull"+str(type(hull)))
    #cv2.imshow("Hull",hull)
    defects = cv2.convexityDefects(cnt, hull)
    print ("HUll",len(hull))
    print ("cnt",len(cnt))
    #print(defects.shape)
    #cv2.imshow("defects",defects)
    #print ("defects"+str(type(defects)))
    # get all the "end points" using the defects and contours
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        end = tuple(cnt[e][0])
        points.append(end)

    # filter out the points which are too close to each other
    #print(points)
    filtered = filter_points(points, 50)

    # sort the fingertips in order of increasing value of the y coordinate
    filtered.sort(key=lambda point: point[1])

    # return the fingertips, at most 5.
    return [pt for idx, pt in zip(range(5), filtered)]


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)


def filter_points(points, filterValue):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] and points[j] and dist(points[i], points[j]) < filterValue:
                points[j] = None
    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered


def plot(frame, points):
    radius = 5
    colour = (0, 0, 255)
    thickness = -1
    for point in points:
        cv2.circle(frame, point, radius, colour, thickness)


hist = capture_histogram(0)
cap = cv2.VideoCapture(0)
screen = np.zeros((600, 1000))
curr = None
prev = None
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    if not ret:
        break
    hand=detect_hand(frame,hist)[1]
    hand_image = hand["boundaries"]
    pts=extract_fingertips(hand)
    #print (pts)
    plot(frame,pts)
    cv2.imshow("points",frame)
    prev=curr
    curr=pts[0]
    if prev and curr:
        cv2.line(screen, prev, curr, (255, 255, 255), 5)
    cv2.imshow("Drawing", screen)
    cv2.imshow("Hand Detector", hand_image)
#     else:
#         cv2.imshow("Hand dectector",frame)
    #hand["contours"]=np.uint8(hand["contours"])
    #print (hand["contours"].depth(),hand["boundaries"].depth())
#     cv2.imshow("Raw",hand["raw"])
#     cv2.imshow("Enhanced Binary",hand["binary"])
#     cv2.imshow("Masked",hand["masked"])
#     cv2.imshow("hsv",hand["hsv"])
#     cv2.imshow("obj_seg",hand["obj_seg"])
#cv2.imshow("disc",hand["disc"])
#     cv2.imshow("filtere",hand["filtere"])
#     cv2.imshow("er",hand["er"])
#     cv2.imshow("di",hand["di"])
#     cv2.imshow("cl",hand["cl"])
    #cv2.imshow("contour",hand["contours"])
    #cv2.imshow("boundaries",hand["boundaries"])
    #print(hand["contours"])
    k=cv2.waitKey(10)
    if k == ord('q'):
        break
#plt.hist(hist)
cap.release()
cv2.destroyAllWindows()