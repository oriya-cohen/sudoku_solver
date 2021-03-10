import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

class ImageGenDataset:
    def __init__(self, labels=[k for k in range(1, 10)], num_labels=10, data_size=10, img_hight=28, img_width=28):
        self.img_hight, self.img_width = img_hight, img_width
        self.loc = 0
        self.max_data_size = data_size
        self.data_labels = np.zeros(data_size)
        self.image_tensor = np.zeros([data_size, img_hight, img_width])
        self.num_labels = num_labels
        self.ttf_font_locs = self.get_fonts(path=os.path.join('..', 'data', 'Google_fonts', 'fonts-master'))
        self.path_list = []
        self.labels = labels

    def gen_dir(self):
        for label in range(1, self.num_labels):
            new_path = os.path.join('..', 'data', 'digit_fonts', str(label))
            if not os.path.exists(new_path):
                os.makedirs(new_path)

    def add_example(self, label, img):
        self.data_labels[self.loc] = label
        self.image_tensor[self.loc] = img
        self.loc += 1

    def get_fonts(self, path=r'C:', path_list=[], extention=".ttf"):
        for root, dirs, files in os.walk(path):
            for file_ in files:
                if file_.endswith(extention):
                    path_list.append(os.path.join(root, file_))
        self.path_list = path_list
        return path_list

    def gen_all_examples(self):

        self.gen_dir()

        # google fonts
        ttf_font_paths = self.get_fonts(path=os.path.join('..', 'data', 'Google_fonts', 'fonts-master'))
        num_fonts = ttf_font_paths.__len__()
        image_size = (40, 40)

        # Write Text on Image
        for font_path in ttf_font_paths:
            # pick a font
            font = ImageFont.truetype(font_path, 28, encoding="unic")
            font_name = os.path.basename(os.path.normpath(font_path))
            font_name = font_name[:-4]
            for label in self.labels:
                text = str(label)
                try:
                    text_width, text_height = font.getsize(text)
                    maximal_size = max((text_width, text_height))
                    img = Image.new('L', (maximal_size * 3, maximal_size * 3))

                    # printing a number
                    draw = ImageDraw.Draw(img)

                    draw.text((maximal_size, maximal_size), str(label), 255, font)
                except:
                    continue

                # np image for processing
                np_im = np.array(img)

                try:
                    y_start, y_end = first_last_nonzero(np.sum(np_im, 1) > 0)
                    x_start, x_end = first_last_nonzero(np.sum(np_im, 0) > 0)
                    center_x, center_y, = (x_end + x_start)//2 , (y_end + y_start)//2
                except:
                    # img.show()
                    # print("error detected")
                    continue

                # cropped_np_im = np_im[y_start:y_end, x_start:x_end]
                cropped_np_im = np_im[(center_y - maximal_size//2):(center_y + maximal_size//2),
                                      (center_x - maximal_size//2):(center_x + maximal_size//2)]

                # show the cropped resized image
                image_saving_size = (self.img_hight, self.img_width)
                image = cv2.resize(cropped_np_im, image_saving_size)

                # show the image
                # Image.fromarray(cropped_np_im).show()

                # save the image
                save_loc = os.path.join('..', 'data', 'digit_fonts', str(label))
                figure_name = font_name + ' ' + str(label) + '.png'
                cv2.imwrite(os.path.join(save_loc, figure_name), image)


def first_last_nonzero(boolean_vector):
    first = last = -1
    for idx, val in enumerate(boolean_vector):
        if val == True and first == -1:
            first = idx
        if val == False and first != -1:
            last = idx
            return first, last


# create data
image_data_generator = ImageGenDataset()
image_data_generator.gen_all_examples()

