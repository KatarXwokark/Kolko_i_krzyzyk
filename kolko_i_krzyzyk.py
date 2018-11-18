import cv2
import numpy as np

cam = cv2.VideoCapture(1)

cv2.namedWindow("TicTacToe")

img_counter = 0

vision = False

while True:
    ret, frame = cam.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    retTresh, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # img = cv2.Canny(img, 50, 150, apertureSize=3)

    img = cv2.dilate(img, (5, 5), iterations=5)

    img = cv2.erode(img, (5, 5), iterations=5)

    lines = cv2.HoughLines(img, 1, np.pi / 30, 200)

    if lines is not None and len(lines) < 50:
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
                temp = np.zeros((1, 2))
                temp[0][0] = (y1 - y2)/(x1 - x2)
                temp[0][1] = (y2 * x1 - x2 * y1) / (x1 - x2)
                newlines[i] = temp
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(newlines)
            xy = np.zeros((4,2))
            xy[0][1] = -(newlines[2][0][1] - newlines[0][0][1]) / (newlines[2][0][0] - newlines[0][0][0])
            xy[0][0] = newlines[0][0][0] * xy[0][1] + newlines[0][0][1]
            xy[1][1] = -(newlines[3][0][1] - newlines[0][0][1]) / (newlines[3][0][0] - newlines[0][0][0])
            xy[1][0] = newlines[0][0][0] * xy[1][1] + newlines[0][0][1]
            xy[2][1] = -(newlines[2][0][1] - newlines[1][0][1]) / (newlines[2][0][0] - newlines[1][0][0])
            xy[2][0] = newlines[1][0][0] * xy[2][1] + newlines[1][0][1]
            xy[3][1] = -(newlines[3][0][1] - newlines[1][0][1]) / (newlines[3][0][0] - newlines[1][0][0])
            xy[3][0] = newlines[1][0][0] * xy[3][1] + newlines[1][0][1]
            h = 0
            print(xy)
            for i in range(4):
                if xy[i][1] <= 0 or xy[i][1] >= len(frame):
                    h = 1
                if xy[i][0] <= 0 or xy[i][0] >= len(frame[0]):
                    h = 1
            if h == 0:
                for i in range(4):
                    for j in range(-2, 3):
                        for n in range(-2, 3):
                            frame[int(xy[i][0]) + j][int(xy[i][1]) + n] = (0, 255, 0)

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
