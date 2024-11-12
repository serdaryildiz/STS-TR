import os
from typing import List

import numpy
from PIL import ImageFont


class Font:
    def __init__(self, args: dict):
        # self.fonts = args["fonts"]
        self.fonts = [os.path.join(args["fonts"], l)for l in os.listdir(args["fonts"])]
        print(self.fonts)
        self.minSize = args["minSize"]
        self.maxSize = args["maxSize"]
        return

    def getRandomFont(self):
        path = numpy.random.choice(self.fonts)
        size = self.getSize()
        font = ImageFont.truetype(path, size=size)
        return font

    def getSize(self):
        size = numpy.random.randint(low=self.minSize, high=self.maxSize)
        return size
