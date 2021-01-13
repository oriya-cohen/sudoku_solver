import numpy as np
import cv2


class line_obj:
    def __init__(self, *argv):
        self.x1, self.y1, self.x2, self.y2 = argv[0]
        self.dif_x = float(self.x2 - self.x1)
        self.dif_y = float(self.y2 - self.y1)
        self.length_line = np.sqrt(np.power(self.dif_x, 2) + np.power(self.dif_y, 2))
        self.theta = np.arctan(np.divide(self.dif_y, self.dif_x))

    def line_extend(self):
        new_x1 = self.x1 - int(10 * self.dif_x / self.length_line)
        new_x2 = self.x2 + int(10 * self.dif_x / self.length_line)
        new_y1 = self.y1 - int(10 * self.dif_y / self.length_line)
        new_y2 = self.y2 + int(10 * self.dif_y / self.length_line)

        line_extend = [[new_x1, new_y1, new_x2, new_y2]]
        return line_extend

    def line_theta(self):
        return self.theta


def get_height():
    height = img_dimension[0]
    return height


# get figure
print('starting ')

# load fig grey scale
original_img = cv2.imread(r'D:\\PyProjects\\sudoku\\pics\\her1.jpeg', 0)
(thresh, BW_image) = cv2.threshold(original_img, 127, 255, cv2.THRESH_BINARY)

# d - size
img_dimension = BW_image.shape

height = get_height()

ratio = 450 / height
d_size = [ratio * img_dimension[0], ratio * img_dimension[1]]
d_size = [int(i) for i in d_size[::-1]]

# resize image
resized_img = cv2.resize(BW_image, tuple(d_size))
cv2.imshow('1. resized_img', resized_img)
# cv2.waitKey(0) & 0xFF


# Taking a matrix of size 3 as the kernel 
kernel = np.ones((3, 3), np.uint8)
# The first parameter is the original image, 
# kernel is the matrix with which image is  
# convolved and third parameter is the number  
# of iterations, which will determine how much  
# you want to erode/dilate a given image.  
img_dilation = cv2.dilate(resized_img, kernel, iterations=1)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

# cv2.imshow('Input', img) 
cv2.imshow('2. Dilation', img_dilation)
cv2.imshow('3. Erosion', img_erosion)
# cv2.waitKey(0) & 0xFF

# blur image 

# kernel_size = 5
# blur_gray = cv2.GaussianBlur(resized_img,(kernel_size, kernel_size),0)
# blur_gray = resized_img
blur_gray = img_erosion

# show blurred scaled grayscale img
cv2.imshow('3. blured', blur_gray)
cv2.waitKey(0) & 0xFF

# identify area of sudoku

# apply canny algorithm to identify edges
low_threshold = 50
high_threshold = 100
apertureSize = 3
edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize)

# show the difference img
cv2.imshow('edges', edges)
cv2.waitKey(0) & 0xFF

# Run Hough on edge detected image
# parameters
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 100  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
# lines = [ line_extend(line) for line in lines]

# group lines by angle
# lines.sort( key = lambda x: line_obj(x[0]).line_theta()) #__________________ correct it later
line_obj_vec = [line_obj(k[0]) for k in lines]

# New image to put lines on
line_image = np.copy(blur_gray) * 0  # creating a blank to draw lines on

for line in lines:

    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

'''
for idx,dat_line in enumerate(lines):   # -> lines[0]
    
    rho,theta=dat_line[0]
    print(['line ',idx])
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
'''

cv2.imshow('Hough lines results', line_image)
cv2.waitKey(0) & 0xFF

# straiten the figure


# getting all 81 squere subfigures and put to a figure list


# for each sub figure run mnist-trained network and get the number to an array


# run an algorithem to solve sudoku


# display solution to screan.

print('The End')
# input("Press Enter to continue...")
