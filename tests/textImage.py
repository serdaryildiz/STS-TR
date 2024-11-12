import os

import cv2
import numpy
from PIL import ImageFont, Image

from Components.CharImage import CharImage
from Components.TextImage import TextImage
from Components.TextProducer import TextProducer
from Texture import Font
from utils import readYAML, saveRGBAImage


def demo1():
    cfgCharAugmentations = "configs/charAugmentations.yaml"
    cfgCharAugmentations = readYAML(cfgCharAugmentations)

    cfgTextAugmentations = "configs/textAugmentations.yaml"
    cfgTextAugmentations = readYAML(cfgTextAugmentations)

    cfgBackground = "configs/background.yaml"
    cfgBackground = readYAML(cfgBackground)

    cfg_TextProducer = "configs/textProducer.yaml"
    cfg_TextProducer = readYAML(cfg_TextProducer)

    textProducer = TextProducer(cfg_TextProducer)
    cnt = 0
    for _ in range(1000):
        # text = "ÖĞRENCİ"
        text = textProducer.getText()
        root = "./sources/fonts/"
        fonts = [os.path.join(root, l) for l in os.listdir(root)]
        for f in sorted(fonts):
            print(f)
            char_list = []
            size = 49
            font = ImageFont.truetype(f, size=size)

            for c in text:
                charImg = CharImage(text=c,
                                    font=font,
                                    colorType="OneColor",
                                    color=(0, 0, 0, 255),
                                    bold=False)
                char_list.append(charImg)

            txtImage = TextImage(char_list, cfgCharAugmentations, cfgTextAugmentations, cfgBackground)

            samples = txtImage.getSamples((1, 1, 1))

            for s in samples:
                saveRGBAImage(s, f"tests/results/{text}-{cnt}.jpg")
                cnt += 1
        # img = Image.fromarray(txtImage.wordImage)
        # img.show()
        # # print(f)
        # cv2.imshow(" ", numpy.array(img))
        # cv2.waitKey(0)


        # for i in range(len(text)):
        #     [x1, y1, x2, y2] = txtImage.charBBoxes[i]
        #     Image.fromarray(txtImage.wordImage[y1:y2, x1:x2]).show()

    return


def demo2():
    cfgCharAugmentations = "configs/charAugmentations.yaml"
    cfgCharAugmentations = readYAML(cfgCharAugmentations)

    cfgTextAugmentations = "configs/textAugmentations.yaml"
    cfgTextAugmentations = readYAML(cfgTextAugmentations)

    cfgBackground = "configs/background.yaml"
    cfgBackground = readYAML(cfgBackground)

    text = "Çağdaş!"
    char_list = []

    size = 45
    font = ImageFont.truetype("sources/fonts/Ubuntu-Regular.ttf", size=size)

    for c in text:
        charImg = CharImage(text=c,
                            font=font,
                            colorType="OneColor",
                            color=(0, 0, 0, 255),
                            bold=False)
        char_list.append(charImg)

    txtImage = TextImage(char_list, cfgCharAugmentations, cfgTextAugmentations, cfgBackground)

    samples = txtImage.getSamples((2, 2, 2))

    return


if __name__ == '__main__':
    # test1()
    demo1()
    # demo2()
