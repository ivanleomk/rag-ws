import string


def strip_punctuation(s):
    return s.translate(str.maketrans("", "", string.punctuation))
