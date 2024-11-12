import copy
from typing import List

import numpy
import tqdm as tqdm
from PIL import Image

from Augmentations import TextImageAugmentations, CharImageAugmentations
from Components.BackgroundBlender import BackgroundBlender
from Components.CharImage import CharImage
from utils import saveRGBAImage


class TextImage:
    def __init__(self, characters: List[CharImage], cfgCharAugmentations=None, cfgTextAugmentations=None, cfgBackground=None):
        self.characters = characters
        self.charLength = len(characters)

        self.wordImage = None
        self.charBBoxes = None

        self.mergeCharacters()

        self.charImageAugmentations = CharImageAugmentations(cfgCharAugmentations)
        self.textImageAugmentations = TextImageAugmentations(cfgTextAugmentations)
        self.blendBackground = BackgroundBlender(cfgBackground)
        return

    def getSamples(self, N: [list, tuple]):
        """
            generate text images using character list
        :param N: number of generated image (how many char, how many text, how many background)
        :return: image list (length == n)
        """
        assert len(N) == 3
        tbar = tqdm.tqdm(total=N[0]*N[1]*N[2], colour='CYAN')
        tbar.set_postfix_str(f" str: {''.join(c.text for c in self.characters)}")
        images = []
        for i in range(N[0]):
            # get base text image
            image, charBboxes = self.getWordImage()
            saveRGBAImage(image+(255-image[..., 3:4]), f"tests/results/{i}_raw_text_img.jpg")
            # augment base chars
            image = self.charImageAugmentations.apply(image=image, bboxes=charBboxes)
            saveRGBAImage(image+(255-image[..., 3:4]), f"tests/results/{i}_CharAugmented_text_img.jpg")
            for j in range(N[1]):
                # augment text image
                image_ = copy.deepcopy(image)
                image_ = self.textImageAugmentations.apply(image=image_)
                saveRGBAImage(image_, f"tests/results/{i}-{j}_ImageAugmented_text_img.jpg")
                for k in range(N[2]):
                    # blend background
                    image__ = copy.deepcopy(image_)
                    image__ = self.blendBackground(image__)
                    saveRGBAImage(image__, f"tests/results/{i}-{j}-{k}_last_img.jpg")
                    images.append(image__)
                    tbar.update(1)
        return images

    def mergeCharacters(self):
        image_arr = []
        bbox_arr = []

        word_width = 0
        word_height = 0
        for i in range(self.charLength):
            image, [x1, y1, x2, y2] = self.characters[i].getImage()

            assert x1 == 0
            bbox = [x1 + word_width, y1, x2 + word_width, y2]

            if word_height == 0:
                word_height = image.shape[0]
            assert image.shape[0], word_height

            word_width += x2 - x1
            bbox_arr.append(bbox)
            image_arr.append(image)

        self.wordImage = numpy.concatenate(image_arr, axis=1)
        self.charBBoxes = bbox_arr
        return

    def getWordImage(self):
        return copy.deepcopy(self.wordImage), self.charBBoxes



