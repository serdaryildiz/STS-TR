from typing import List

import cv2
import numpy
import imgaug.augmenters as iaa

from Augmentations import PadLeftRight, CustomSequenceAugmentations, ResizeChar


class CharImageAugmentations:
    def __init__(self, args):
        self.augmentationSequence = self.getAugmentations(args)
        self.customAugmentationSequence = self.getCustomAugmentations(args)

    def apply(self, image: numpy.ndarray, bboxes: list):
        crops = []
        cropH = []
        for bbox in bboxes:
            [x1, y1, x2, y2] = bbox
            crop = image[:, x1:x2]
            crop = self.augmentationSequence(image=crop)
            crop = self.customAugmentationSequence(image=crop)
            crops.append(crop)
            cropH.append(crop.shape[0])
        image = self.concatenateCrops(crops, cropH)
        return image

    @staticmethod
    def concatenateCrops(crops: List[numpy.ndarray], cropH: List[int]):
        def getPadHSize(maxPad):
            if maxPad == 0:
                pad = 0
            else:
                pad = numpy.random.randint(low=0, high=maxPad)
            return pad

        maxH = max(cropH)
        minH = min(cropH)
        if maxH == minH:
            image = numpy.concatenate(crops, axis=1)
        else:
            tmp = []
            pads = {}

            for crop in crops:
                h, w, c = crop.shape
                newCrop = numpy.zeros((maxH, w, 4), dtype=numpy.uint8)
                if h in list(pads.keys()):
                    pad = pads[h]
                else:
                    pad = getPadHSize(maxPad=(maxH - h))
                    pads[h] = pad

                newCrop[pad:h + pad, ...] = crop
                tmp.append(newCrop)
            image = numpy.concatenate(tmp, axis=1)
        return image

    @staticmethod
    def getAugmentations(args):
        sequence = []
        augmentations = args['geometricAugmentations']
        for augName in list(augmentations.keys()):
            p = augmentations[augName]['p']
            if 'ElasticTransformation' in augName:
                newAugmentation = iaa.Sometimes(p, iaa.ElasticTransformation(alpha=(augmentations[augName]['min_alpha'],
                                                                                    augmentations[augName][
                                                                                        'max_alpha']),
                                                                             sigma=(augmentations[augName]['min_sigma'],
                                                                                    augmentations[augName][
                                                                                        'max_sigma']),
                                                                             mode=augmentations[augName]['mode']))
            elif augName == '':
                newAugmentation = None
                pass
            else:
                raise Exception("Unknown augmentation type pn char image augmentations!")

            sequence.append(newAugmentation)

        return iaa.Sequential(sequence, random_order=True)

    @staticmethod
    def getCustomAugmentations(args):
        sequence = []
        augmentations = args['customAugmentations']
        for augName in list(augmentations.keys()):
            p = augmentations[augName]['p']
            if "PadLeftRight" in augName:
                newAugmentation = PadLeftRight(p, pad=[augmentations[augName]['min_pad'],
                                                       augmentations[augName]['max_pad']])
            elif "ResizeChar" in augName:
                newAugmentation = ResizeChar(p,
                                             ratio=[augmentations[augName]['min_ratio'],
                                                    augmentations[augName]['max_ratio']],
                                             minW=augmentations[augName]['min_W'],
                                             minH=augmentations[augName]['min_H'])
            else:
                raise Exception("Unknown augmentation type pn char image augmentations!")
            sequence.append(newAugmentation)

        return CustomSequenceAugmentations(sequence)
