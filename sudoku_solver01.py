import cv2
import numpy as np
import math


def find_num(img):
    cropped_image = [0]
    height, width = img.shape

    num_edges = cv2.Canny(img, 50, 80)  # edges
    ret, thresh = cv2.threshold(num_edges, 127, 255, 0)
    contours, hierarchy = \
        cv2.findContours(thresh, cv2.cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # only the external contour
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)  # This will find out co-ord for plate
        if 0.1 * width <= w <= 0.95 * width and 0.3 * height <= h <= 0.8 * height:
            cropped_image = img[y:y + h, x:x + w]  # Create new image

            break

    return cropped_image

def get_numbers_in_fig(new_img, size=9):
    x_len, y_len = new_img.shape

    # size of mini image
    x_mini = x_len // size
    y_mini = y_len // size

    # sud_array = np.empty((size, size,1))
    sud_array = [[0] * size for i in range(size)]

    # create array of figures
    for c in range(size):
        for r in range(size):
            sud_array[r][c] = new_img[x_mini * r: x_mini * (r + 1), y_mini * c: y_mini * (c + 1)]
            sud_array[r][c] = find_num(sud_array[r][c])

            # test

            if len(sud_array[r][c]) != 1:
                cv2.imshow('is it a number?', sud_array[r][c])
                cv2.waitKey(1000)

    return sud_array


def order_points(pts):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = np.array(pts).sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    try:
        transverse = np.linalg.inv(M)
        if True:
            pass   # check trans did good
    except:
        transverse = -1

    # return the warped image
    # and the transverse coordinate transform

    return warped, transverse


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
    line_width = 5
    sq_marked_img = big_img
    sq_marked_img = cv2.rectangle(sq_marked_img, points[0], points[2], (200, 50, 200), line_width)

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
    return math.atan((line[0][3] - line[0][1]) / (line[0][2] - line[0][0]))


def unite_lines(lines):
    try:
        # sort by cos(theta) of the line
        united_lns = sorted(lines,
                            key=lambda line: math.cos(line_theta(line)))
        grouped_lines = united_lns  ### edit this ###
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
    frame_edges = cv2.Canny(spots_cleared_frame, 50, 80)
    # cv2.imshow('frame_edges', frame_edges)

    # find contours
    ret, thresh = cv2.threshold(frame_edges, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = \
        cv2.findContours(thresh, cv2.cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # only the external contour

    # show contours
    # frame_edges = cv2.drawContours(frame_edges, contours, -1, (255, 255, 255), 6)
    # cv2.imshow('frame_edges', frame_edges)

    # sort contours by area - this part is based on :
    #           https://github.com/AjayAndData/Licence-plate-detection-and-recognition---using-openCV-only
    #           /blob/master/Car%20Number%20Plate%20Detection.py
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max(5, len(contours))]

    # we need the maximal squared area
    idx = 1
    for cntr in contours:
        perimeter = cv2.arcLength(cntr, True)
        approx = cv2.approxPolyDP(cntr, 0.02 * perimeter, True)
        # print ("approx = ",approx)
        if len(approx) == 4 and cv2.contourArea(cntr) > 0.5 * np.size(gray):  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            # x, y, w, h = cv2.boundingRect(cntr)  # This will find out co-ord for plate
            # new_img = gray[y:y + h, x:x + w]  # Create new image

            # my edition. stretching a Quadrilateral to square and get the transverse transform
            # new_img, transverse_trans = four_point_transform(gray, pts=[approx[i][0] for i in range(4)]) # original gray
            new_img, transverse_trans = \
                four_point_transform(spots_cleared_frame, pts=[approx[i][0] for i in range(4)])  # cleared image

            if np.size(new_img) > 0.5 * np.size(gray):  # only for large figures display them
                cv2.imshow('Cropped Images /' + str(idx) + '.png', new_img)  # display new image
                cv2.waitKey(0)
                # cv2.imwrite('Cropped Images-Text/' + str(idx) + '.png', new_img)  # Store new image
                # idx += 1

                # pass the image to image recognition solver
                soduko_arr = get_numbers_in_fig(new_img)
            break

    # find lines
    # minLineLength = 100
    # maxLineGap = 100
    # # lines = cv2.HoughLinesP(frame_egdes, 1, np.pi / 180, 20, minLineLength, maxLineGap)
    # lines = cv2.HoughLinesP(frame_egdes, 1, 5 * np.pi / 180, 20, minLineLength, maxLineGap)
    #
    # # unite lines
    # # lines = unite_lines(lines)
    # try:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         frame_to_edit = cv2.line(frame_to_edit, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # except:
    #     print('nan found')

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
        # cv2.imshow('edited_frame', edited_frame)

    cap.release()  # release the camera
    cv2.destroyAllWindows()  # closes the current windows
