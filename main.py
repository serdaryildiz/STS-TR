from Components.CharImage import CharImage
from Components.TextImage import TextImage
from Components.TextProducer import TextProducer
from Texture import Font
from utils import readYAML, ImageWriter


def main(cfg):
    """
        main function
    :param cfg: configs
    """
    (cfg_Base, cfg_CharAugmentations, cfg_TextAugmentations, cfg_Background, cfg_TextProducer) = cfg

    font = Font(cfg_Base["FONT"])

    textProducer = TextProducer(cfg_TextProducer)
    writer = ImageWriter(root=cfg_Base["root"], isLMDB=cfg_Base["isLMDB"])

    for _ in range(cfg_Base["numUniqueText"]):
        text = textProducer.getText()
        fontSample = font.getRandomFont()
        try:
            generator(text, fontSample, cfg_Base, cfg_CharAugmentations, cfg_TextAugmentations, cfg_Background, writer.writeSamples)
        except Exception as e:
            print(e)
    return


def generator(text, font, cfg_Base, cfg_CharAugmentations, cfg_TextAugmentations, cfg_Background, writer):
    """
        Image generator
    :param text: text
    :param font: font
    :param cfg_Base: base config
    :param cfg_CharAugmentations: char augmentation config
    :param cfg_TextAugmentations: text augmentation config
    :param cfg_Background: background augmentation config
    :param writer: writer function pointer
    """

    # get character images
    char_list = []
    for c in text:
        charImg = CharImage(text=c,
                            font=font,
                            colorType="OneColor",
                            color=(0, 0, 0, 255),
                            bold=False)
        char_list.append(charImg)

    # get text images
    txtImage = TextImage(char_list, cfg_CharAugmentations, cfg_TextAugmentations, cfg_Background)

    # get samples
    samples = txtImage.getSamples(cfg_Base["getSamples"])

    # write samples
    writer(text, samples)
    return samples[0]


if __name__ == '__main__':
    cfgBase = "configs/base.yaml"
    cfgCharAugmentations = "configs/charAugmentations.yaml"
    cfgTextAugmentations = "configs/textAugmentations.yaml"
    cfgBackground = "configs/background.yaml"
    cfgTextProducer = "configs/textProducer.yaml"

    cfgBase = readYAML(cfgBase)
    cfgCharAugmentations = readYAML(cfgCharAugmentations)
    cfgTextAugmentations = readYAML(cfgTextAugmentations)
    cfgBackground = readYAML(cfgBackground)
    cfgTextProducer = readYAML(cfgTextProducer)

    cfg = (cfgBase, cfgCharAugmentations, cfgTextAugmentations, cfgBackground, cfgTextProducer)

    main(cfg)
