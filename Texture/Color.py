import os

import numpy
from PIL import Image
import blend_modes


class Painter:
    """
        Give a random color to given image
    """

    def __init__(self):
        return

    def __call__(self, image: numpy.ndarray):
        """
            apply a random color to given image
        :param image: image
        :return: painted image
        """
        alpha = image[..., 3]
        src = numpy.empty_like(image)
        src[..., :] = self.getColor()
        src = Image.fromarray(src)
        dst = Image.fromarray(image)
        out = Image.alpha_composite(dst, src)
        out = numpy.array(out)
        out[..., 3] = alpha
        return out

    @staticmethod
    def getColor():
        """
         get random color
        :return: color ( R, G, B)
        """
        r = numpy.random.randint(low=0, high=255)
        g = numpy.random.randint(low=0, high=255)
        b = numpy.random.randint(low=0, high=255)
        alpha = 255
        return numpy.array([r, g, b, alpha])


class TextureMixer:
    """
        Add Texture to Text Image using blend_modes package.
        more info : https://blend-modes.readthedocs.io/en/latest/reference.html
        blend functions : https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Difference
    """

    def __init__(self, p: int, root: str, maxOpacity: float):
        self.p = p
        self.root = root
        assert 0 <= maxOpacity <= 1
        self.maxOpacity = maxOpacity
        self.imagePaths = None
        self.blendFunctions = [
            blend_modes.addition,
            blend_modes.divide,
            blend_modes.subtract,
            blend_modes.difference,
            blend_modes.darken_only,
            blend_modes.lighten_only
        ]
        return

    def __call__(self, image: numpy.ndarray):
        if self.isRun():
            h, w, _ = image.shape
            texture = self.getTextureImage(w=w, h=h)
            # there is appropriate texture image
            if texture is not None:
                image = self.blend(textImage=image, texture=texture)
        return image

    def blend(self, textImage: numpy.ndarray, texture: numpy.ndarray):
        """
            blend texture and text image
        :param textImage: text image
        :param texture: texture image
        :return: blended image
        """
        alpha = textImage[..., 3]
        blender = self.getBlendFunction()
        opacity = self.getOpacity()
        image = blender(texture.astype(float), textImage.astype(float), opacity)
        image[..., 3] = alpha
        return image.astype(numpy.uint8)

    def getOpacity(self):
        opacity = numpy.random.randint(low=0.6, high=int(self.maxOpacity * 100)) / 100
        return opacity

    def getBlendFunction(self):
        """
            select blend functions
        :return: blend function
        """
        return numpy.random.choice(self.blendFunctions)

    def getTextureImage(self, w: int, h: int):
        """
            produce texture image for text image
        :param w: width of text image
        :param h: height of text image
        :return: image of texture
        """
        # if texture is None, there isn't any appropriate w and h
        texture = None
        counter = 0
        while texture is None:
            path = self.getImagePath()
            texture = numpy.array(Image.open(path).convert("RGBA"))
            texture = self.getTextureCrop(texture=texture, w=w, h=h)
            counter += 1
            if counter > 10:
                texture
                break
        return texture

    @staticmethod
    def getTextureCrop(texture: numpy.ndarray, w: int, h: int):
        """
            crop the texture image
        :param texture: texture image
        :param w: width of image
        :param h: height of image
        :return: crop of texture
        """
        hTexture, wTexture, _ = texture.shape
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
            imagePaths = []
            dirs = os.listdir(self.root)
            for dirName in dirs:
                path = os.path.join(self.root, dirName)
                imagePaths += [os.path.join(path, p) for p in os.listdir(path)]
            self.imagePaths = imagePaths

        path = numpy.random.choice(self.imagePaths)
        return path

    def isRun(self):
        rand = numpy.random.rand()
        if rand <= self.p:
            return True
        else:
            return False
