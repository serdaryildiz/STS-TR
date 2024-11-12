import cv2
import numpy
from typing import List
from math import pi

from wand.image import Image as WImage


class CustomAugmentation:
    def __init__(self, p):
        self.p = p

    def isRun(self):
        rand = numpy.random.rand()
        if rand <= self.p:
            return True
        else:
            return False

    def apply(self, image: numpy.ndarray):
        raise NotImplementedError()


class PadLeftRight(CustomAugmentation):
    def __init__(self, p, pad: List[float]):
        super(PadLeftRight, self).__init__(p)
        assert len(pad) == 2
        assert pad[0] < pad[1]
        self.pad = numpy.array(pad)

    def apply(self, image: numpy.ndarray):
        if self.isRun():
            h, w, c = image.shape
            assert c == 4, "Image has to be RGBA"
            padLeft, padRight = self.getPadSizes(w=w)
            image = numpy.concatenate((numpy.zeros((h, padLeft, 4), dtype=numpy.uint8),
                                       image,
                                       numpy.zeros((h, padRight, 4), dtype=numpy.uint8)), axis=1)
        return image

    def getPadSizes(self, w: int):
        # assert w > 5
        pad = numpy.ceil(self.pad * w).astype(int)
        padLeft = numpy.random.randint(low=pad[0], high=pad[1]+1)
        padRight = numpy.random.randint(low=pad[0], high=pad[1]+1)
        assert padLeft != 0 and padRight != 0
        return padLeft, padRight


class ResizeChar(CustomAugmentation):
    def __init__(self, p, ratio: list, minW: int, minH: int):
        super(ResizeChar, self).__init__(p)
        assert len(ratio) == 2
        assert ratio[0] < ratio[1]
        self.ratio = numpy.array(ratio).astype(int) * 100
        self.minW = minW
        self.minH = minH

    def apply(self, image: numpy.ndarray):
        if self.isRun():
            h, w, _ = image.shape
            newW, newH = self.getNewWH(w, h)
            if newW != w or newH != h:
                image = cv2.resize(image, (newW, newH), interpolation=cv2.INTER_NEAREST)
        return image

    def getNewWH(self, w: int, h: int):
        r = numpy.random.randint(low=self.ratio[0], high=self.ratio[1]) / 100
        newW = int(w * r)
        newH = int(h * r)
        if newW < self.minW or newH < self.minH:
            newH = h
            newW = w
        return newW, newH


class AffineTransform(CustomAugmentation):

    def __init__(self, p, maxRotate: int, maxTranslate: int):
        super().__init__(p)
        self.maxRotate = maxRotate
        self.maxTranslate = maxTranslate

    def apply(self, image: numpy.ndarray):
        if self.isRun():
            args = self.getAffineMatrixArgs()
            image = WImage.from_array(numpy.array(image))
            image.virtual_pixel = 'transparent'
            image.distort('affine_projection', args)
            image = numpy.array(image)
        return image

    def getAffineMatrixArgs(self):
        rotateX = numpy.random.randint(low=-self.maxRotate, high=self.maxRotate) / 100
        rotateY = numpy.random.randint(low=-self.maxRotate, high=self.maxRotate) / 100
        scaleX = 1
        scaleY = 1
        translateX = numpy.random.randint(low=-self.maxTranslate, high=self.maxTranslate)
        translateY = numpy.random.randint(low=-self.maxTranslate, high=self.maxTranslate)
        return scaleX, rotateX, rotateY, scaleY, translateX, translateY


class WrapText(CustomAugmentation):
    def __init__(self, p, arcAngle: List[int], rotateAngle: List[int]):
        super().__init__(p)
        assert arcAngle[0] < arcAngle[1]
        assert rotateAngle[0] < rotateAngle[1]
        self.arcAngle = numpy.array(arcAngle)
        self.rotateAngle = numpy.array(rotateAngle)

    def apply(self, image: numpy.ndarray):
        if self.isRun():
            arc, rotate = self.getAngles()
            image = WImage.from_array(numpy.array(image))
            image.virtual_pixel = 'transparent'
            image.distort('arc', (arc, rotate))
            image = numpy.array(image)
        return image

    def getAngles(self):
        arc = numpy.random.randint(low=self.arcAngle[0], high=self.arcAngle[1])
        rotate = numpy.random.randint(low=self.rotateAngle[0], high=self.rotateAngle[1])
        return arc, rotate


class Transformation3D(CustomAugmentation):
    """
    Reference : https://github.com/eborboihuc/rotate_3d
    """

    def __init__(self, p, maxTheta, maxPhi, maxGamma):
        super().__init__(p)
        self.maxTheta = maxTheta
        self.maxPhi = maxPhi
        self.maxGamma = maxGamma
        self.pParam = numpy.sqrt(p) // 2

    def apply(self, image: numpy.ndarray):
        if self.isRun():
            h, w, _ = image.shape
            theta, phi, gamma, dx, dy, dz = self.getParams()
            rTheta, rPhi, rGamma = self.get_rad(theta, phi, gamma)
            d = numpy.sqrt(h ** 2 + w ** 2)
            focal = d / (2 * numpy.sin(rGamma) if numpy.sin(rGamma) != 0 else 1)
            dz = focal
            H = self.getH(focal, w, h, rTheta, rPhi, rGamma, dx, dy, dz)
            image = self.wrap(image, H)
        return image

    def wrap(self, image: numpy.ndarray, H: numpy.array):
        h, w, _ = image.shape
        corners = numpy.array([[
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ]], dtype=numpy.float32)
        corners = cv2.perspectiveTransform(corners, H)[0]
        bx, by, bboxWidth, bboxHeight = cv2.boundingRect(corners)
        th = numpy.array([
            [1, 0, -bx],
            [0, 1, -by],
            [0, 0, 1]
        ])
        mat = th @ H
        image = cv2.warpPerspective(image, mat, (bboxWidth, bboxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return image

    def getParams(self):
        if self.pParam < numpy.random.rand():
            theta = numpy.random.randint(low=-self.maxTheta, high=self.maxTheta)  # rotation around x-axis
        else:
            theta = 0
        if self.pParam < numpy.random.rand():
            phi = numpy.random.randint(low=-self.maxPhi, high=self.maxPhi)
        else:
            phi = 0
        if self.pParam < numpy.random.rand():
            gamma = numpy.random.randint(low=-self.maxGamma, high=self.maxGamma)
        else:
            gamma = 0
        dx = 0
        dy = 0
        dz = 0
        return theta, phi, gamma, dx, dy, dz

    def getH(self, f, w, h, theta, phi, gamma, dx, dy, dz):
        # Projection 2D -> 3D matrix
        A1 = numpy.array([[1, 0, -w / 2],
                          [0, 1, -h / 2],
                          [0, 0, 1],
                          [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = numpy.array([[1, 0, 0, 0],
                          [0, numpy.cos(theta), -numpy.sin(theta), 0],
                          [0, numpy.sin(theta), numpy.cos(theta), 0],
                          [0, 0, 0, 1]])

        RY = numpy.array([[numpy.cos(phi), 0, -numpy.sin(phi), 0],
                          [0, 1, 0, 0],
                          [numpy.sin(phi), 0, numpy.cos(phi), 0],
                          [0, 0, 0, 1]])

        RZ = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                          [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = numpy.dot(numpy.dot(RX, RY), RZ)

        # Translation matrix
        T = numpy.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = numpy.array([[f, 0, w / 2, 0],
                          [0, f, h / 2, 0],
                          [0, 0, 1, 0]])

        # Final transformation matrix
        return numpy.dot(A2, numpy.dot(T, numpy.dot(R, A1)))

    def get_rad(self, theta, phi, gamma):
        return (self.deg_to_rad(theta),
                self.deg_to_rad(phi),
                self.deg_to_rad(gamma))

    def deg_to_rad(self, deg):
        return deg * pi / 180.0


class CustomSequenceAugmentations:
    def __init__(self, augmentations: List[CustomAugmentation]):
        self.augmentations = augmentations

    def __call__(self, image: numpy.ndarray):
        for aug in self.augmentations:
            image = aug.apply(image)
        return image
