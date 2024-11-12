import numpy


class TextProducer:
    """
        Text producer
    """
    lower2upper = {
        ord(u"i"): u"İ",
        ord(u"ı"): u"I"
    }

    atTheBeginning = ["(", "-", "%", "$", "#", "[", "*", "{"]
    atTheEnd = [")", "-", "%", "$", "#", "]", "*", "}", "!", "?"]
    atTheMiddle = ["-", "_", "&"]

    def __init__(self, args: dict):
        self.args = args
        self.datasets = args["datasets"]
        self.maxLength = args["maxLength"]
        self.pWord = args["pWord"]
        self.pLower10 = args["pLower10"]
        self.pAllUpperCase = args["pAllUpperCase"]
        self.pFirstUpperCase = args["pFirstUpperCase"]
        self.pAddNonAlphanumeric = args["pAddNonAlphanumeric"]

        self.words = set()
        self.length = None
        self._readDataset()
        return

    def getText(self):
        """
            get a text (number or word)
        :return: text
        """
        if numpy.random.rand() <= self.pWord:
            word = self._getWord()
            txt = self._augmentWord(word)
        else:
            txt = self._getNumber()
        return txt

    def _augmentWord(self, word):
        """
            Augment Text
        :param word: raw text
        :return: processed text
        """
        # lower case or upper case
        if numpy.random.rand() <= self.pAllUpperCase:
            word = word.translate(self.lower2upper).upper()
        elif numpy.random.rand() <= self.pFirstUpperCase:
            word = f"{word[0].translate(self.lower2upper).upper()}{word[1:]}"

        # add non-alphanumeric characters
        if numpy.random.rand() <= self.pAddNonAlphanumeric:
            rnd = numpy.random.rand()
            if rnd < 0.33:
                nonAlphanumericChar = numpy.random.choice(self.atTheBeginning)
                word = f"{nonAlphanumericChar}{word}"
            elif 0.33 <= rnd <= 0.66:
                mid = len(word) // 2
                nonAlphanumericChar = numpy.random.choice(self.atTheMiddle)
                word = f"{word[:mid]}{nonAlphanumericChar}{word[mid:]}"
            else:
                nonAlphanumericChar = numpy.random.choice(self.atTheEnd)
                word = f"{word}{nonAlphanumericChar}"

        return word

    def _getNumber(self):
        """
            get a number
        :return: string of a number
        """
        if numpy.random.rand() <= self.pLower10:
            num = numpy.random.randint(low=0, high=10)
        else:
            num = numpy.random.randint(low=10, high=1e9)
        return str(num)

    def _getWord(self):
        """
            get a word
        :return:
        """
        flag = True
        while flag:
            idx = numpy.random.randint(low=0, high=self.length)
            word = self.words[idx]
            if len(word) <= self.maxLength:
                flag = False
        return word

    def _readDataset(self):
        """
            read datasets and produce set of words
        :return:
        """
        for dPath in self.datasets:
            with open(dPath, 'r') as fp:
                words = fp.readlines()
                for w in words:
                    w = w.strip()
                    self.words.add(w)
        self.length = len(self.words)
        self.words = list(self.words)
        return
