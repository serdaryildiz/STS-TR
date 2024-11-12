import os
import cv2
import numpy
from PIL import Image

from Texture import Painter


class BackgroundBlender:

    def __init__(self, args):
        self.args = args
        self.root = args["BackgroundTexture"]["root"]
        self.distanceTh = args["BackgroundTexture"]["distanceTh"]
        self.numColor = args["BackgroundTexture"]["numColor"]
        self.resizeSize = (128, 32)
        self.imagePaths = None
        self.oneColorP = args["BackgroundTexture"]["oneColorP"]
        return

    def __call__(self, image: numpy.ndarray):
        background = self.getAppropriateBackground(image)
        if background is not None:
            image = self.blender(image=image, background=background)
        return image

    @staticmethod
    def blender(image: numpy.ndarray, background: numpy.ndarray):
        h, w, _ = image.shape
        mask = image[..., 3] > 0
        mask = mask.reshape(h, w, 1).repeat(4, axis=2)
        image = image * mask

        image = Image.fromarray(image.astype(numpy.uint8))
        background = Image.fromarray(background.astype(numpy.uint8))
        mask = Image.fromarray(numpy.array(image)[..., 3])
        image = Image.composite(image, background, mask)
        image = numpy.array(image)

        return image.astype(numpy.uint8)

    def getAppropriateBackground(self, image: numpy.ndarray):
        """
            returns appropriate background image for text image
        :param image: text image
        :return: background image
        """
        h, w, _ = image.shape
        imgResized = cv2.resize(image, self.resizeSize, interpolation=cv2.INTER_CUBIC)

        background = None
        appropriate = False
        counter = 0
        while not appropriate:
            background = self.getBackgroundImage(w=w, h=h)
            appropriate = self.isAppropriate(imageResized=imgResized, background=background)
            counter += 1
            if counter > 10:
                background = None
                break
        return background

    def isAppropriate(self, imageResized: numpy.ndarray, background: numpy.ndarray):
        """
            check is background appropriate
        :param imageResized: resized image
        :param background: background
        :return: True / False
        """
        h, w, c = self.resizeSize[1], self.resizeSize[0], 4
        assert imageResized.shape == (h, w, c)

        mask = imageResized[..., 3] > 0
        mask = mask.astype(int).reshape(h, w, 1).repeat(4, axis=2)
        bgResized = cv2.resize(background, self.resizeSize, interpolation=cv2.INTER_CUBIC)

        imageResized = imageResized * mask
        bgResized = bgResized * (1 - mask)

        merged = numpy.concatenate((imageResized, bgResized), axis=0)

        # old version
        # merged = Image.fromarray(merged.astype(numpy.uint8)[..., :3]).quantize(colors=self.numColor)
        # merged = numpy.array(merged)

        merged = self.rgb2gray(merged)
        merged = merged / 255
        merged = merged * self.numColor
        merged = merged.astype(numpy.uint8)

        bgHist = numpy.zeros((self.numColor+1,))
        imgHist = numpy.zeros((self.numColor+1,))

        colors, counts = numpy.unique(merged[:h], return_counts=True)
        for i, c in enumerate(colors):
            imgHist[c] = counts[i]

        maxCount = numpy.max(imgHist)

        colors, counts = numpy.unique(merged[h:], return_counts=True)
        for i, c in enumerate(colors):
            bgHist[c] = counts[i]

        if numpy.max(counts) > maxCount:
            maxCount = numpy.max(bgHist)

        bgHist = bgHist / maxCount
        imgHist = imgHist / maxCount
        distance = numpy.sum(numpy.abs(imgHist[1:] - bgHist[1:]))

        return distance >= self.distanceTh

    def getBackgroundImage(self, w: int, h: int):
        """
            produce background image for text image
        :param w: width of text image
        :param h: height of text image
        :return: image of texture
        """
        texture = None
        while texture is None:
            if numpy.random.rand() <= self.oneColorP:
                texture = numpy.empty((h, w, 4))
                texture[..., :] = Painter.getColor()
                noise = numpy.random.randint(low=-10, high=10, size=texture.shape)
                texture += noise
                texture = numpy.clip(texture, a_min=0, a_max=255)
            else:
                # if texture is None, there isn't any appropriate w and h
                path = self.getImagePath()
                texture = numpy.array(Image.open(path).convert("RGBA"))
                texture = self.getBackgroundCrop(texture=texture, w=w, h=h)
        return texture

    @staticmethod
    def getBackgroundCrop(texture: numpy.ndarray, w: int, h: int):
        """
            crop the background image
        :param texture: texture image
        :param w: width of image
        :param h: height of image
        :return: crop of texture
        """
        hTexture, wTexture, _ = texture.shape
        assert hTexture < 1000, wTexture<1000
        if h >= hTexture or w >= wTexture:
            return None

        rangeH = hTexture - h
        rangeW = wTexture - w
        x = numpy.random.randint(low=0, high=rangeW)
        y = numpy.random.randint(low=0, high=rangeH)
        return texture[y:y + h, x:x + w, ...]

    def getImagePath(self):
        """
            select randomly an image path for given dataset
        :return: path
        """
        if self.imagePaths is None:
            imagePaths = os.listdir(self.root)
            self.imagePaths = [os.path.join(self.root, p) for p in imagePaths]

        path = numpy.random.choice(self.imagePaths)
        return path

    def rgb2gray(self, image: numpy.ndarray):
        """
            convert RGB to Gray
        :return: image
        """
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray