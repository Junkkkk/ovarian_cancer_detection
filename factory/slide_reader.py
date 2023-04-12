import openslide
import numpy as np
from PIL import Image
import cv2


class WSI:
    def __init__(self, wsi_path, level):
        self.wsi_path = wsi_path
        self.level = level
        self.width, self.height, self.wsi = self.get_wsi()

        # self.region_img = self.get_region_img()
        # self.foreground_img = self.get_foreground_img()

    def get_wsi(self):
        wsi = openslide.OpenSlide(self.wsi_path)
        width, height = wsi.level_dimensions[self.level]
        return width, height, wsi

    def get_whole_img(self, size=1000):
        whole_img = self.wsi.get_thumbnail(size=(size, size))
        whole_img = np.asarray(whole_img)
        return whole_img

    def get_patch_img(self, start_coord, size):

        patch_img = self.wsi.read_region(start_coord, self.level, (size, size),)
        patch_img = cv2.cvtColor(np.array(patch_img), cv2.COLOR_RGBA2RGB)

        return patch_img
