import numpy
from PIL import ImageFont, Image, ImageDraw


class CharImage:
    def __init__(self, text: str,
                 font: ImageFont,
                 colorType: str = "OneColor",
                 color: list or tuple = None,
                 bold: bool = False):
        """
            Text Image Class
        :param text: text which will draw on an image
        :param font: text font
        :param colorType: texture or OneColor
        :param color: text colour
        :param bold: bold or not
        """
        if colorType != "texture":
            assert len(color) == 4, "Color has to be 4 channel because of RGBA Image"

        self.text = text
        self.font = font
        self.colorType = colorType
        if colorType == "texture":
            self.color = (0, 0, 0, 255)
        else:
            self.color = color
        self.bold = bold

        self.stroke_width = 1
        self.direction = "ltr"

        self.bbox = None
        self.image = None

        # offsets
        self.x = 0
        self.y = 0
        self.ascent = None
        self.descent = None
        self.centerx = None
        self.centery = None

        self.width = None
        self.height = None
        return

    def getImage(self):
        if self.image is None:
            bbox = self.getBbox()
            width, height = bbox[2:]

            image = Image.new("RGBA", (width, height))
            draw = ImageDraw.Draw(image)
            draw.text(xy=(self.x, self.y),  # Top left corner
                      text=self.text,
                      fill=self.color,
                      font=self.font,
                      stroke_width=self.stroke_width,
                      direction=self.direction)

            image = numpy.array(image, dtype=numpy.uint8)

            self.image = image
            self.bbox = bbox

        return self.image, self.bbox

    def getBbox(self):
        ascent, descent = self.font.getmetrics()
        width = self.font.getsize(self.text, direction=self.direction)[0]
        height = ascent + descent
        bbox = [self.x, height-ascent, width, height]

        self.ascent = ascent
        self.descent = descent
        self.width = width
        self.height = height

        # can be calculated based on ascent and decent
        self.centerx = width // 2
        self.centery = height // 2

        return bbox


def test1():
    size = 49
    font = ImageFont.truetype("../sources/fonts/Ubuntu-Regular.ttf", size=size)
    txtImg = CharImage(text="Ö",
                       font=font,
                       colorType="OneColor",
                       color=(0, 0, 0, 255),
                       bold=False)

    image, bbox = txtImg.getImage()
    print(image.shape)

    txtImg2 = CharImage(text="Q",
                        font=font,
                        colorType="OneColor",
                        color=(0, 0, 0, 255),
                        bold=False)

    image2, bbox2 = txtImg2.getImage()
    [x1, y1, x2, y2] = bbox
    word = numpy.concatenate((image, image2), axis=1)
    # Image.fromarray(word).show()
    print(word.shape)
    assert word.shape[0] == image.shape[0] == image2.shape[0]

    firstWord = word[y1:y2, x1:x2, ...]
    Image.fromarray(firstWord).show()

if __name__ == '__main__':
    test1()
