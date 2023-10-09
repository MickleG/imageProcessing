import cv2
import math
import numpy as np

def houghing(input_image, index):
    img = cv2.imread(input_image)

    dst = cv2.Canny(img, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 350, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


    # linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 150, None, 50, 10)

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Source", 659, 853)

    cv2.namedWindow("Standard Hough " + str(index), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Standard Hough " + str(index), 659, 853)

    # cv2.namedWindow("P Hough", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("P Hough", 659, 853)

    # cv2.imshow("Source", img)
    cv2.imshow("Standard Hough " + str(index), cdst)
    # cv2.imshow("P Hough", cdstP)

    cv2.waitKey()

def contrast(input_image, index):
    img = cv2.imread(input_image, -1)

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    # bg_merge = []
    # result_binary_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        # bg_merge.append(bg_img)
        diff_img = cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        # result_binary_planes.append(binary_img)
    
    # bg_final_img = cv2.merge(bg_merge)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    result_norm_gray = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)

    ret, binary_img = cv2.threshold(result_norm_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # binary_img2 = cv2.adaptiveThreshold(result_norm_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 20)

    kernel = np.ones((2, 2), np.uint8)

    # cv2.namedWindow("standard " + str(index), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("standard " + str(index), 659, 853)

    # cv2.namedWindow("thresholded " + str(index), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("thresholded " + str(index), 659, 853)

    cv2.namedWindow("eroded_dilated " + str(index), cv2.WINDOW_NORMAL)

    cv2.resizeWindow("eroded_dilated " + str(index), 659, 853)

    # cv2.namedWindow("result_norm " + str(index), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("result_norm " + str(index), 659, 853)


    # cv2.namedWindow("adaptive " + str(index), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("adaptive " + str(index), 659, 853)

    # cv2.imshow("text_filter " + str(index), bg_final_img)
    # cv2.imshow("standard " + str(index), img)
    # cv2.imshow("result_norm " + str(index), result_norm)
    # cv2.imshow("thresholded " + str(index), binary_img)
    cv2.imshow("eroded_dilated " + str(index), cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel))
    
    # cv2.imshow("adaptive " + str(index), binary_img2)

    cv2.waitKey()

for i in range(1, 9):
    index = i
    if(i < 10):
        input_image_name = "data/data1/000" + str(i) + ".png"
    else:
        input_image_name = "data/data1/00" + str(i) + ".png"
    # houghing(input_image_name, index)
    contrast(input_image_name, index)

    cv2.destroyAllWindows()
    # print(input_image_name)

# input_image_name = "data/bad.jpg"

contrast(input_image_name, 1)

cv2.destroyAllWindows()