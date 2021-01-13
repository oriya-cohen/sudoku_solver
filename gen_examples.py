# generate sudoku examples

import numpy as np
import cv2

class sud_figure:
    
    def __init__(self, *args, **kwargs):
        self.size_of_sudoku = args(0)
        self.contain_numbers = np.zeros([self.size_of_sudoku, self.size_of_sudoku])
        self.values_at_place = np.zeros([self.size_of_sudoku, self.size_of_sudoku])
        self.fig = np.ones((300,300,1), np.uint8)*255
        

        return super().__init__(*args, **kwargs)