# https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/

import numpy as np
import cv2 as cv
# Create a black image
img = np.ones((512,512,3), np.uint8)
img = img * 255
# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)

 
# Filename 
filename = 'D:\\checks\\savedImage.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
cv.imwrite(filename, img) 




'''
import tkinter

root = tkinter.Tk()
canvas = tkinter.Canvas(root)
canvas.pack()

for i in range(10):
    canvas.create_line(50 * i, 0, 50 * i, 400)
    canvas.create_line(0, 50 * i, 400, 50 * i)
canvas.create_rectangle(100, 100, 200, 200, fill="blue")
canvas.create_line(50, 100, 250, 200, fill="red", width=10)

canvas.create_text(70,
              70,
              text="4")


root.mainloop()
'''

# class shmendric:

#    def __init__(self, *args, **kwargs):
#        self.x=args[0]
#        self.y=args[1]
#        # return super().__init__(*args, **kwargs)

#    def leng(self):
#        return self.x+self.y
    

#oriya = shmendric(3,5)

#print(oriya.leng())

