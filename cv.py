import numpy as np
import cv2


def nothing(a):
    return a


def MinRect(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def MaxContourArea(cont):
    maxArea = 0
    maxCnt = np.array([[[0,  0]]], dtype=np.int32)
    for cnt in cont:
        cntArea = cv2.contourArea(cnt)
        if cntArea > maxArea:
            maxArea = cntArea
            maxCnt = cnt
    return maxCnt


def ColorTrack(image, kernel_size, upper_color, lower_color):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = np.copy(image)
    blur = cv2.medianBlur(img, kernel_size)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    maskCont = cv2.dilate(mask, kernel, iterations=1)
    maskCont = cv2.medianBlur(maskCont, kernel_size)
    maskCont = cv2.morphologyEx(maskCont, cv2.MORPH_CLOSE, kernel)
    maskCont = cv2.morphologyEx(maskCont, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(maskCont, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = MaxContourArea(contours)
    epsilon = 0.008 * cv2.arcLength(maxContour, True)
    approxMaxContour = cv2.approxPolyDP(maxContour, epsilon, True)
    return approxMaxContour, maskCont


'''
cv2.namedWindow("frame")

cv2.createTrackbar('H_Up', 'frame', 180, 180, nothing)
cv2.createTrackbar('S_Up', 'frame', 255, 255, nothing)
cv2.createTrackbar('V_Up', 'frame', 255, 255, nothing)

cv2.createTrackbar('H_Down', 'frame', 0, 180, nothing)
cv2.createTrackbar('S_Down', 'frame', 0, 255, nothing)
cv2.createTrackbar('V_Down', 'frame', 0, 255, nothing)
'''

upperPen = np.array((180, 30, 115))
lowerPen = np.array((0, 0, 0))

pic_name = 'picture2.jpg'

picture = cv2.imread(pic_name, 1)
maxContourPen, maskP = ColorTrack(picture, 3, upperPen, lowerPen)
# cv2.drawContours(picture, [maxContourPen], 0, (255, 0, 0), 3)
cv2.drawContours(picture, [MinRect(maxContourPen)], 0, (0, 0, 255), 2)
# cv2.imshow('maskP', maskP)
cv2.imshow('img', picture)


while 1:
    '''
    picture = cv2.imread(pic_name, 1)

    upper = (cv2.getTrackbarPos('H_Up', 'frame'), cv2.getTrackbarPos('S_Up', 'frame'), cv2.getTrackbarPos('V_Up', 'frame'))
    lower = (cv2.getTrackbarPos('H_Down', 'frame'), cv2.getTrackbarPos('S_Down', 'frame'), cv2.getTrackbarPos('V_Down', 'frame'))

    maxContourTest, maskT = ColorTrack(picture, 3, upper, lower)
    cv2.drawContours(picture, [maxContourTest], 0, (0, 100, 0), 3)
    cv2.drawContours(picture, [MinRect(maxContourTest)], 0, (100, 0, 0), 2)

    cv2.imshow('test_img', picture)
    cv2.imshow('maskT', maskT)
    '''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
