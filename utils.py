import os
from typing import List
from datetime import datetime

import lmdb
import numpy
import yaml
from PIL import Image


def readYAML(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
    return dic


def saveRGBAImage(image: numpy.ndarray, path: str, raw: bool = True):
    if raw:
        image = Image.fromarray(image)
        new_image = Image.new("RGBA", image.size)  # Create a white rgba background
        new_image.paste(image, (0, 0), image)
        new_image.convert('RGB').save(path, "JPEG")
    else:
        raise NotImplementedError()
    return


class ImageWriter:
    """
        image writer
    """

    def __init__(self, root: str, isLMDB: bool):
        self.isLMDB = isLMDB
        self.root = root
        self.sep = "-*-"

        if self.isLMDB:
            path = os.path.join(root, "SyntheticTurkishStyleText_Samples")
            os.makedirs(path, exist_ok=True)

            path = os.path.join(root, "SyntheticTurkishStyleText")
            os.makedirs(path, exist_ok=True)
            self.env = lmdb.open(path, map_size=1099511627776)

    def writeSamples(self, text: str, samples: List[numpy.ndarray]):
        if self.isLMDB:
            now = datetime.now()
            date_time = now.strftime("%H-%M-%S")
            saveRGBAImage(samples[0], os.path.join(
                self.root,"SyntheticTurkishStyleText_Samples", f"{text}{self.sep}{0}{self.sep}{date_time}.jpg"))
            cache = {}
            for i, img in enumerate(samples):
                now = datetime.now()
                date_time = now.strftime("%H-%M-%S")
                imageKey = f"{text}{self.sep}{i}{self.sep}{date_time}"

                # because of jpeg compression
                path = os.path.join(self.root, f"temp.jpg")
                saveRGBAImage(img, path)
                with open(path, 'rb') as f:
                    imageBin = f.read()

                cache[imageKey.encode()] = imageBin
            self.writeCache(cache)
        else:
            for i, img in enumerate(samples):
                now = datetime.now()
                date_time = now.strftime("%H-%M-%S")
                path = os.path.join(self.root, f"{text}{self.sep}{i}{self.sep}{date_time}.jpg")
                saveRGBAImage(img, path)

    def writeCache(self, cache):
        with self.env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)
