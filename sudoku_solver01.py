import cv2
import numpy as np
import math


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def clear_image_spots(frame_to_clear):
    # Creating kernel

    # first try
    # kernel = np.ones((3, 3), np.uint8)
    # Using cv2.erode() method
    # eroded_img = cv2.erode(frame_to_clear, kernel, cv2.BORDER_REFLECT)
    # Using cv2.dilate() method
    # Dilated_img = cv2.dilate(eroded_img, kernel, cv2.BORDER_REFLECT)

    # second try
    ## opening (erosion followed by dilation) and closing(the opposite)
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(frame_to_clear, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # cleared_img = closing

    # third try
    cleared_img = cv2.bilateralFilter(frame_to_clear, 10, 20, 20)  # apply bilateral filter

    # forth try
    # kernel = np.ones((2, 2), np.uint8)
    # cleared_img = cv2.adaptiveThreshold(frame_to_clear, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cleared_img = cv2.morphologyEx(cleared_img, cv2.MORPH_CLOSE, kernel)

    return cleared_img


def boxed_frame(big_img, size_sub_window_0to1):
    # taken from
    # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # answer by https://stackoverflow.com/users/4907913/thewaywewere

    height, width, channels = big_img.shape
    min_dim = min(height, width)  # the minimal dimension
    half_window_size = math.floor(min_dim * size_sub_window_0to1 / 2)
    mid_frame = (math.ceil(height / 2), math.ceil(width / 2))

    # creating list of (x,y) tuples for the edges of Range Of Interest (ROI)
    y_idx = np.add(mid_frame[0], [-half_window_size, -half_window_size, +half_window_size, +half_window_size])
    x_idx = np.add(mid_frame[1], [-half_window_size, +half_window_size, +half_window_size, -half_window_size])
    points = [pt for pt in zip(x_idx, y_idx)]

    # plot ROI on image
    line_width = 1
    sq_marked_img = big_img
    sq_marked_img = cv2.rectangle(sq_marked_img, points[0], points[2], (100, 100, 100), line_width)

    # drew that way but there is a shelf code for it
    # for k in range(len(points)):
    #     sq_marked_img = cv2.line(sq_marked_img, points[k - 1], points[k], (255, 0, 0), line_width)

    sub_img = big_img[points[0][1]:points[2][1],
              points[0][0]:points[2][0]]  # sub_img = img[y(pt1):y(pt3), x(pt1):x(pt3)]

    sub_img = image_resize(sub_img, height=500)

    return sq_marked_img, sub_img


def line_dist(line1, line2):
    pass


def line_theta(line):
    return math.atan(   (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])   )


def unite_lines(lines):
    try:
        # sort by cos(theta) of the line
        united_lns = sorted(lines,
                            key=lambda line: math.cos(line_theta(line)) )
        grouped_lines = united_lns                                            ### edit this ###
        for line in lines:
            x1, y1, x2, y2 = line[0]
            frame_to_edit = cv2.line(frame_to_edit, (x1, y1), (x2, y2), (255, 0, 0), 2)
        lines = grouped_lines
    except:
        print('nan found')
    return lines


def edit_frame(frame_to_edit):

    gray = cv2.cvtColor(frame_to_edit, cv2.COLOR_BGR2GRAY)  # gray color

    # (thresh, frame_to_edit) = cv2.threshold(frame_to_edit, 127, 255, cv2.THRESH_BINARY)   # black and white img

    spots_cleared_frame = clear_image_spots(gray)  # clear image spots
    cv2.imshow('cleared_frame', spots_cleared_frame)
    # find edges
    frame_egdes = cv2.Canny(spots_cleared_frame, 50, 80)
    cv2.imshow('frame_egdes', frame_egdes)

    # find contours
    ret, thresh = cv2.threshold(frame_egdes, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_egdes = cv2.drawContours(frame_egdes, contours, -1, (255, 255, 255), 6)
    cv2.imshow('frame_egdes', frame_egdes)

    # find lines
    minLineLength = 100
    maxLineGap = 100
    # lines = cv2.HoughLinesP(frame_egdes, 1, np.pi / 180, 20, minLineLength, maxLineGap)
    lines = cv2.HoughLinesP(frame_egdes, 1, 5 * np.pi / 180, 20, minLineLength, maxLineGap)

    # unite lines
    # lines = unite_lines(lines)
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            frame_to_edit = cv2.line(frame_to_edit, (x1, y1), (x2, y2), (255, 0, 0), 2)
    except:
        print('nan found')

    return frame_to_edit


if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break
        range_sub_rect = 2 / 3
        frame_box, sub_frame = boxed_frame(frame, range_sub_rect)
        cv2.imshow('feed', frame_box)

        edited_frame = edit_frame(sub_frame)
        cv2.imshow('edited_frame', edited_frame)


    cap.release()  # release the camera
    cv2.destroyAllWindows()  # closes the current windows
