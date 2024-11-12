import numpy

from Augmentations import WrapText, CustomSequenceAugmentations, AffineTransform
from Augmentations.Augmentations import Transformation3D
from Texture import Painter, TextureMixer


class TextImageAugmentations:
    def __init__(self, args):
        self.augmentations = []
        self.layoutAugmentation = self.getCustomLayoutAugmentations(args)
        self.painter = Painter()
        self.texture = self.getTextureMixer(args)

    @staticmethod
    def getTextureMixer(args):
        p = args["Texture"]["TextureMixer"]["p"]
        root = args["Texture"]["TextureMixer"]["root"]
        maxOpacity = args["Texture"]["TextureMixer"]["maxOpacity"]
        return TextureMixer(p=p, root=root, maxOpacity=maxOpacity)

    def apply(self, image: numpy.ndarray):
        image = self.layoutAugmentation(image=image)
        image = self.painter(image)
        image = self.texture(image)
        return image

    @staticmethod
    def getCustomLayoutAugmentations(args):
        sequence = []
        augmentations = args['customLayoutAugmentation']
        for augName in list(augmentations.keys()):
            p = augmentations[augName]['p']
            if "WrapText" in augName:
                newAugmentation = WrapText(p,
                                           arcAngle=[augmentations[augName]["minArcAngle"],
                                                     augmentations[augName]["maxArcAngle"]],
                                           rotateAngle=[augmentations[augName]["minRotateAngle"],
                                                        augmentations[augName]["maxRotateAngle"]])
            elif "AffineTransform" in augName:
                newAugmentation = AffineTransform(p,
                                                  augmentations[augName]["maxRotate"],
                                                  augmentations[augName]["maxTranslate"])
            elif "Transformation3D" in augName:
                newAugmentation = Transformation3D(p,
                                                   augmentations[augName]["maxTheta"],
                                                   augmentations[augName]["maxPhi"],
                                                   augmentations[augName]["maxGamma"])
            else:
                raise Exception("Unknown augmentation type pn char image augmentations!")
            sequence.append(newAugmentation)
        return CustomSequenceAugmentations(sequence)
