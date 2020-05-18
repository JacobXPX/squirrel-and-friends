import logging
import re
from functools import reduce

import dateutil.parser as dparser
import emoji
import nltk
from nltk.tokenize import word_tokenize
from numerizer import numerize
from spellchecker import SpellChecker

from ._regexes import *

nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p",
                    level=logging.INFO)


class textCleaner(object):
    def __init__(self):
        self.__name__ = "textCleaner"

    @staticmethod
    def rm_tag(text):
        """Remove tags.
        """

        res = HTML_TAG_REGEX.sub(r"", text)
        return res

    @staticmethod
    def tag_url(text, tag_with="URL"):
        """Rag url.
        """

        res = URL_REGEX.sub(tag_with, text)
        return res

    @staticmethod
    def tag_phone(text, tag_with="PHONE"):
        """Tag phone number.
        """

        res = PHONE_REGEX.sub(tag_with, text)
        return res

    @staticmethod
    def tag_email(text, tag_with="EMAIL"):
        """Tag email.
        """

        res = EMAIL_REGEX.sub(tag_with, text)
        return res

    @staticmethod
    def parse_cont(text):
        """Decontract and parse contractions.
        """

        for con, decon in SPECIAL_CONTRACTION.items():
            text = re.sub(con, decon, text)
        for con, decon in COMMON_CONTRACTION.items():
            text = re.sub(con, decon, text)
        return text

    @staticmethod
    def parse_abbr(text):
        """Parse abbreviations.
        """

        tokens = word_tokenize(text)

        def convert_abbr(w):
            if w.lower() in ABBREVIATION.keys():
                return ABBREVIATION[w.lower()]
            else:
                return w

        tokens = [convert_abbr(w) for w in tokens]
        res = " ".join(tokens)
        return res

    @staticmethod
    def parse_camel(text):
        """Decontract and parse camel words.
        """

        res = re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", text)
        return res

    @staticmethod
    def parse_emoji(text, use_aliases=False):
        """Parse emoji to meaningful words.
        Take advantage of emoji. https://pypi.org/project/emoji/
        """

        def replace(match):
            codes_dict = UNICODE_EMOJI_ALIAS if use_aliases else UNICODE_EMOJI
            val = codes_dict.get(match.group(0), match.group(0))
            return " " + val[1:-1].replace("_", " ") + " "

        res = re.sub(
            u"\ufe0f", "", (emoji.core.get_emoji_regexp().sub(replace, text)))
        return res

    @staticmethod
    def replace_currency(text):
        """Replace currency sign by 3-digit unit.
        """

        res = reduce(lambda text, cur:
                     text.replace(cur, CURRENCIES[cur] + " "),
                     CURRENCIES, text)
        return res

    @staticmethod
    def replace_number(text):
        """Replace number words to digit number.
        """

        return numerize(text)

    @staticmethod
    def tag_number(text, tag_with="<NUM>"):
        """Tag numbers.
        """

        tokens = word_tokenize(text)
        tokens = [w[0] if w[1] != "CD" else tag_with
                  for w in nltk.pos_tag(tokens)]
        res = " ".join(tokens)
        return res

    @staticmethod
    def tag_punct(text, tag_with={"!": "<exclamation>", "?": "<question>"}):
        """Tag meaningful words.
        """

        for sign, tag in tag_with.items():
            text = text.replace(sign, " " + tag)

        return text

    @staticmethod
    def rm_punct(text, ignore=None):
        """Remove punctuations.
        """

        text = text.replace("...", " ")
        ignore = [] if not ignore else ignore
        for p in [x for x in PUNCTUATION if x not in ignore]:
            text = text.replace(p+" ", p)
            text = text.replace(p, " ")
        return text

    @staticmethod
    def inner_strip(text):
        """Remove inner multiple space.
        """

        res = " ".join([char for char in text.split(" ") if char])
        return res

    @staticmethod
    def correct_spell(text):
        """Correct wrong spelled words.
        Notes:
            Poor performance, be careful while using it.
        """

        spell = SpellChecker()
        misspelled_words = spell.unknown(text.split())

        corrected_text = []
        for word in text.split():
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)

        res = " ".join(corrected_text)
        return res

    @staticmethod
    def replace_emoji(text):
        """Replace emoji by `-` linked word, surrounded by `:`.
        """

        res = emoji.demojize(text)
        return res

    @staticmethod
    def exact_datetime(text, fuzzy=True):
        """Extract datetime from text
        """

        res = dparser.parse(text, fuzzy=fuzzy)
        return res


def all_in_one(self, text):
    """Preprocess on the text including most of above steps
        and in proper order.
    """

    funcs = [
        self.rm_tag,
        self.tag_url,
        self.tag_phone,
        self.tag_email,
        self.parse_cont,
        self.parse_abbr,
        self.parse_camel,
        self.parse_emoji,
        self.replace_currency,
        self.replace_number,
        self.tag_number,
        self.tag_punct,
        self.rm_punct,
        self.inner_strip,
    ]

    for func in funcs:
        logging.info("start: "+func.__name__)
        text = func(text)

    res = text.lower()
    return res
