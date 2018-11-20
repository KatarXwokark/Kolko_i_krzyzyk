import cv2
import numpy as np

cam = cv2.VideoCapture(1)

cv2.namedWindow("TicTacToe")

img_counter = 0

vision = False

def line_equation(x1, y1, x2, y2):
    temp = np.zeros((1, 2))
    temp[0][0] = (y1 - y2) / (x1 - x2)
    temp[0][1] = (y2 * x1 - x2 * y1) / (x1 - x2)
    return temp

def detection(crosses, circles, x1, y1, x2, y2, x3, y3, x4, y4):
    xy12 = line_equation(x1, y1, x2, y2)
    xy23 = line_equation(x2, y2, x3, y3)
    xy34 = line_equation(x3, y3, x4, y4)
    xy41 = line_equation(x4, y4, x1, y1)
    #print(xy12, xy23, xy34, xy41)
    if xy12[0][1] < xy34[0][1]:
        temp = xy34
        xy34 = xy12
        xy12 = temp
    if xy23[0][1] < xy41[0][1]:
        temp = xy23
        xy23 = xy41
        xy41 = temp
    for cross in crosses:
        if cross[0][1] <= xy12[0][0] * cross[0][0] + xy12[0][1] and cross[0][1] <= xy23[0][0] * cross[0][0] + xy23[0][1]:
            if cross[0][1] >= xy34[0][0] * cross[0][0] + xy34[0][1] and cross[0][1] >= xy41[0][0] * cross[0][0] + xy41[0][1]:
                return ("X", cross)
    for circle in circles:
        if circle[0][1] <= xy12[0][0] * circle[0][0] + xy12[0][1] and circle[0][1] <= xy23[0][0] * circle[0][0] + xy23[0][1]:
            if circle[0][1] >= xy34[0][0] * circle[0][0] + xy34[0][1] and circle[0][1] >= xy41[0][0] * circle[0][0] + xy41[0][1]:
                return ("O", circle[0])
    return (" ", np.zeros((3)))

while True:
    plansza = [["","",""],["","",""],["","",""]]

    ret, frame = cam.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    retTresh, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # img = cv2.Canny(img, 50, 150, apertureSize=3)

    img = cv2.dilate(img, (5, 5), iterations=5)

    img = cv2.erode(img, (5, 5), iterations=5)

    lines = cv2.HoughLines(img, 1, np.pi / 30, 200)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                            param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)

    if lines is not None and circles is not None and len(lines) < 50:
        newlines = np.zeros((4, 1, 2))
        newlines[0] = lines[0]
        moe = 0.1
        k = 1
        for i in range(1, len(lines)):
            if lines[i][0][1] > newlines[0][0][1] - moe and lines[i][0][1] < newlines[0][0][1] + moe:
                if newlines[1][0][0] == 0 and newlines[1][0][1] == 0:
                    newlines[1] = lines[i]
                    k += 1
            elif lines[i][0][1] > newlines[0][0][1] - moe + np.pi/2 and lines[i][0][1] < newlines[0][0][1] + moe + np.pi/2:
                if newlines[2][0][0] == 0 and newlines[2][0][1] == 0:
                    newlines[2] = lines[i]
                    k += 1
                elif newlines[3][0][0] == 0 and newlines[3][0][1] == 0:
                    newlines[3] = lines[i]
                    k += 1
        if k == 4:
            for i in range(4):
                rho, theta = newlines[i][0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                newlines[i] = line_equation(x1, y1, x2, y2)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            xy = np.zeros((4, 4, 2))
            xy[1][1][1] = -(newlines[2][0][1] - newlines[0][0][1]) / (newlines[2][0][0] - newlines[0][0][0])
            xy[1][1][0] = newlines[0][0][0] * xy[1][1][1] + newlines[0][0][1]
            xy[1][2][1] = -(newlines[3][0][1] - newlines[0][0][1]) / (newlines[3][0][0] - newlines[0][0][0])
            xy[1][2][0] = newlines[0][0][0] * xy[1][2][1] + newlines[0][0][1]
            xy[2][1][1] = -(newlines[2][0][1] - newlines[1][0][1]) / (newlines[2][0][0] - newlines[1][0][0])
            xy[2][1][0] = newlines[1][0][0] * xy[2][1][1] + newlines[1][0][1]
            xy[2][2][1] = -(newlines[3][0][1] - newlines[1][0][1]) / (newlines[3][0][0] - newlines[1][0][0])
            xy[2][2][0] = newlines[1][0][0] * xy[2][2][1] + newlines[1][0][1]
            xy[0][1][1] = xy[1][1][1] + (xy[1][1][1] - xy[2][1][1])
            xy[0][1][0] = xy[1][1][0] + (xy[1][1][0] - xy[2][1][0])
            xy[0][2][1] = xy[1][2][1] + (xy[1][2][1] - xy[2][2][1])
            xy[0][2][0] = xy[1][2][0] + (xy[1][2][0] - xy[2][2][0])
            xy[1][0][1] = xy[1][1][1] + (xy[1][1][1] - xy[1][2][1])
            xy[1][0][0] = xy[1][1][0] + (xy[1][1][0] - xy[1][2][0])
            xy[2][0][1] = xy[2][1][1] + (xy[2][1][1] - xy[2][2][1])
            xy[2][0][0] = xy[2][1][0] + (xy[2][1][0] - xy[2][2][0])
            xy[3][1][1] = xy[2][1][1] + (xy[2][1][1] - xy[1][1][1])
            xy[3][1][0] = xy[2][1][0] + (xy[2][1][0] - xy[1][1][0])
            xy[3][2][1] = xy[2][2][1] + (xy[2][2][1] - xy[1][2][1])
            xy[3][2][0] = xy[2][2][0] + (xy[2][2][0] - xy[1][2][0])
            xy[1][3][1] = xy[1][2][1] + (xy[1][2][1] - xy[1][1][1])
            xy[1][3][0] = xy[1][2][0] + (xy[1][2][0] - xy[1][1][0])
            xy[2][3][1] = xy[2][2][1] + (xy[2][2][1] - xy[2][1][1])
            xy[2][3][0] = xy[2][2][0] + (xy[2][2][0] - xy[2][1][0])
            xy[0][0][1] = xy[0][1][1] + (xy[0][1][1] - xy[0][2][1])
            xy[0][0][0] = xy[0][1][0] + (xy[0][1][0] - xy[0][2][0])
            xy[0][3][1] = xy[0][2][1] + (xy[0][2][1] - xy[0][1][1])
            xy[0][3][0] = xy[0][2][0] + (xy[0][2][0] - xy[0][1][0])
            xy[3][0][1] = xy[3][1][1] + (xy[3][1][1] - xy[3][2][1])
            xy[3][0][0] = xy[3][1][0] + (xy[3][1][0] - xy[3][2][0])
            xy[3][3][1] = xy[3][2][1] + (xy[3][2][1] - xy[3][1][1])
            xy[3][3][0] = xy[3][2][0] + (xy[3][2][0] - xy[3][1][0])

            for i in range(4):
                for j in range(4):
                    for n in range(-2*(i+1), 3*(i+1)):
                        for m in range(-2*(j+1), 3*(j+1)):
                            if int(xy[i][j][0]) + n >= 0 and int(xy[i][j][0]) + n < len(frame) and int(
                                    xy[i][j][1]) + n >= 0 and int(xy[i][j][1]) + n < len(frame[0]):
                                frame[int(xy[i][j][0]) + n][int(xy[i][j][1]) + m] = (0, 255, 0)

            newcircles = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    cv2.line(frame, (int(xy[0+i][0+j][1]), int(xy[0+i][0+j][0])),
                             (int(xy[0+i][1+j][1]), int(xy[0+i][1+j][0])), (0, 0, 255), 2)
                    cv2.line(frame, (int(xy[0 + i][1 + j][1]), int(xy[0 + i][1 + j][0])),
                             (int(xy[1 + i][1 + j][1]), int(xy[1 + i][1 + j][0])), (0, 0, 255), 2)
                    cv2.line(frame, (int(xy[1 + i][1 + j][1]), int(xy[1 + i][1 + j][0])),
                             (int(xy[1 + i][0 + j][1]), int(xy[1 + i][0 + j][0])), (0, 0, 255), 2)
                    cv2.line(frame, (int(xy[1 + i][0 + j][1]), int(xy[1 + i][0 + j][0])),
                             (int(xy[0 + i][0 + j][1]), int(xy[0 + i][0 + j][0])), (0, 0, 255), 2)
                    temp = detection([], circles, xy[0+i][0+j][1], xy[0+i][0+j][0], xy[0+i][1+j][1],
                              xy[0+i][1+j][0], xy[1+i][1+j][1], xy[1+i][1+j][0], xy[1+i][0+j][1], xy[1+i][0+j][0])
                    plansza[i][j] = temp[0]
                    newcircles[i][j] = temp[1]
            if plansza != [[" "," "," "],[" "," "," "],[" "," "," "]]:
                print(plansza)

            if newcircles is not None:
                newcircles = np.uint16(np.around(newcircles))
                for i in range(3):
                    for j in newcircles[i]:
                        cv2.circle(frame, (j[0], j[1]), j[2], (0, 255, 0), 2)
                        cv2.circle(frame, (j[0], j[1]), 2, (0, 0, 255), 3)

    if vision:
        cv2.imshow("TicTacToe", frame)
    else:
        cv2.imshow("TicTacToe", img)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("ESC hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "photos/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    elif k % 256 == 118:
        if vision:
            vision = False
        else:
            vision = True

cam.release()

cv2.destroyAllWindows()
