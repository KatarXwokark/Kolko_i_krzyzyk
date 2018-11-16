import cv2
import numpy as np

cam = cv2.VideoCapture(0)

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

    # img = cv2.erode(img, (5, 5), iterations=5)

    lines = cv2.HoughLines(img, 1, np.pi / 30, 200)

    if lines is not None and len(lines) < 50:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

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
