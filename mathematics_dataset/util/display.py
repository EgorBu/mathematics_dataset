# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functionality for displaying expressions.

SymPy provides a lot of functionality for displaying expressions, but it's
slightly too centered on being a symbolic maths engine to provides all our
needs. For example, it's impossible to display an unsimplified fraction like
3/6, or a decimal that isn't internally represented as a float and thus subject
to rounding.

Also provides some other convenience such as converting numbers to words, and
displaying percentages (properly formatted).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import decimal

# Dependency imports
import sympy
from num2words import num2words

_LANG = 'ru'

# For converting integers to words:
_INTEGER_LOW = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteeen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_INTEGER_MID = [
    "",
    "",
    "twenty",
    "thirty",
    "fourty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
_INTEGER_HIGH = [
    (int(1e12), "trillion"),
    (int(1e9), "billion"),
    (int(1e6), "million"),
    (int(1e3), "thousand"),
    (100, "hundred"),
]


# For converting rationals to words:
_SINGULAR_DENOMINATORS = [
    "",
    "",
    "half",
    "third",
    "quarter",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
]
_PLURAL_DENOMINATORS = [
    "",
    "",
    "halves",
    "thirds",
    "quarters",
    "fifths",
    "sixths",
    "sevenths",
    "eighths",
    "ninths",
    "tenths",
    "elevenths",
    "twelths",
    "thirteenths",
    "fourteenths",
    "fifteenths",
    "sixteenths",
    "seventeenths",
    "eighteenths",
    "nineteenths",
    "twentieths",
]


# For converting ordinals to words:
_ORD_ROOTS = [
    'нулев',
    'перв',
    'втор',
    'трет',
    'четверт',
    'пят',
    'шест',
    'седьм',
    'восьм',
    'девят',
    'десят',
    'одиннадцат',
    'двенадцат',
    'тринадцат',
    'четырнадцат',
    'пятнадцат',
    'шестнадцат',
    'семнадцат',
    'восемнадцат',
    'девятнадцат',
]


# Numbers with 3 in the end is exception in terms of cases.
# `третИЙ` - _ord_cases_3[True]["nomn"]["male"]
# _ord_cases_3: Dict
#     - is_singular: Dict[Bool, Dict]
#         - case: Dict[str, Optional[Dir, str]]
#             - *gender*: Dict[str, str]
_ord_cases_3 = {
    True: {
        'nomn': {'male': 'ий', 'femn': 'ья', 'mid': 'ье'},
        'gent': {'male': 'ьего', 'femn': 'ьей', 'mid': 'ьего'},
        "datv": {'male': 'ьему', 'femn': 'ьей', 'mid': 'ьему'}
    },
    False: {
        'nomn': 'ьи',
        'gent': 'ьих',
        "datv": "ьим",
        "accs": "ьи",
        "ablt": "ьими",
        "loct": "ьих"
    }
}


def _get_ord_cases(x: int,
                   gen: str = 'male',
                   case: str = 'nomn',
                   is_sing: bool = True):
    if x < 0 or x > len(_ORD_ROOTS):
        raise ValueError("Unsupported ordinal {}.".format(x))
    root = _ORD_ROOTS[x]
    end = '![Warning]'
    if x == 3:
        if is_sing:
            end = _ord_cases_3[is_sing][case][gen]
        else:
            end = _ord_cases_3[is_sing][case]
    elif is_sing:
        if gen == 'male':
            if case == 'nomn':
                if x in (0, 2, 6, 7, 8):
                    end = 'ой'
                else:
                    end = 'ый'
            if case == 'datv':
                end = 'ому'
        elif gen == 'femn':
            if case == 'nomn':
                end = 'ая'
            elif case in ['datv', 'gent']:
                end = 'ой'
        else:
            if case == 'nomn':
                end = 'ое'
            elif case == 'datv':
                end = 'ому'
    else:
        if case in ['nomn', 'accs']:
            end = 'ые'
        elif case in ['gent', 'loct']:
            end = 'ых'
        elif case == 'datv':
            end = 'ым'
        else:
            end = 'ыми'

    return root + end


def _get_numer(x: int):
    '''Get numeral string for rational'''
    l = x % 100
    n = x - l
    words = [num2words(n, lang='ru')] if n > 0 else []
    l0 = l % 10
    if not 10 < l < 20 and l0 in [1, 2]:
        l1 = l - l0
        if l1 > 0:
            words.append(num2words(l1, lang='ru'))
        if l0 == 1:
            words.append('одна')
        else:
            words.append('две')
    else:
        words.append(num2words(l, lang='ru'))

    return ' '.join(words)


class Decimal(object):
    """Display a value as a decimal."""

    def __init__(self, value):
        """Initializes a `Decimal`.

        Args:
          value: (Sympy) value to display as a decimal.

        Raises:
          ValueError: If `value` cannot be represented as a non-terminating decimal.
        """
        self._value = sympy.Rational(value)

        numer = int(sympy.numer(self._value))
        denom = int(sympy.denom(self._value))

        denom_factors = list(sympy.factorint(denom).keys())
        for factor in denom_factors:
            if factor not in [2, 5]:
                raise ValueError(
                    "Cannot represent {} as a non-recurring decimal.".format(value)
                )
        self._decimal = decimal.Decimal(numer) / decimal.Decimal(denom)

    @property
    def value(self):
        """Returns the value as a `sympy.Rational` object."""
        return self._value

    def _sympy_(self):
        return self._value

    def decimal_places(self):
        """Returns the number of decimal places, e.g., 32 has 0 and 1.43 has 2."""
        if isinstance(self._decimal, int):
            return 0
        elif isinstance(self._decimal, decimal.Decimal):
            return -self._decimal.as_tuple().exponent

    def __str__(self):
        sign, digits, exponent = self._decimal.as_tuple()
        sign = "" if sign == 0 else "-"

        num_left_digits = len(digits) + exponent  # number digits "before" point

        if num_left_digits > 0:
            int_part = "".join(str(digit) for digit in digits[:num_left_digits])
        else:
            int_part = "0"

        if exponent < 0:
            frac_part = "."
            if num_left_digits < 0:
                frac_part += "0" * -num_left_digits
            frac_part += "".join(str(digit) for digit in digits[exponent:])
        else:
            frac_part = ""

        return sign + int_part + frac_part

    def __add__(self, other):
        if not isinstance(other, Decimal):
            raise ValueError("Arithmetic support limited to other `Decimal`s.")
        return Decimal(self.value + other.value)

    def __sub__(self, other):
        if not isinstance(other, Decimal):
            raise ValueError("Arithmetic support limited to other `Decimal`s.")
        return Decimal(self.value - other.value)

    def __mul__(self, other):
        if not isinstance(other, Decimal):
            raise ValueError("Arithmetic support limited to other `Decimal`s.")
        return Decimal(self.value * other.value)

    def __neg__(self):
        return Decimal(-self.value)

    def round(self, ndigits=0):
        """Returns a new `Decimal` rounded to this many decimal places."""
        scale = sympy.Integer(10**ndigits)
        numer = sympy.numer(self.value) * scale
        denom = sympy.denom(self.value)
        return Decimal(int(round(numer / denom)) / scale)

    def __round__(self, ndigits):
        return self.round(ndigits)

    def __int__(self):
        """Returns conversion to integer if possible; TypeError if non-integer."""
        if self.decimal_places() == 0:
            return int(self._decimal)
        else:
            raise TypeError("Cannot represent {} as an integer.".format(str(self)))

    # NOTE: this is implemented in addition to `__cmp__` because SymPy does not
    # support inequality comparison between sympy objects and objects that are not
    # convertible to sympy objects (such as strings).
    def __eq__(self, other):
        return self.value == other

    # Python 2 comparison
    def __cmp__(self, other):
        if self.value == other:
            return 0
        if self.value < other:
            return -1
        return 1

    # Python 3 comparison:
    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other


class Percentage(object):
    """Container for a percentage."""

    def __init__(self, value):
        """Initializes a `Percentage`.

        Args:
          value: Percentage as a fractional value. E.g., pass in
              `sympy.Rational(2, 5)` to create the percentage "40%".
        """
        self._value = value

    def _sympy_(self):
        return self._value

    def __str__(self):
        # Display percentages as decimals (not fractions).
        value = Decimal(self._value * 100)
        return str(value) + "%"


class NonSimpleRational(object):
    """Container for rational a / b where allow gcd(a, b) > 1."""

    def __init__(self, numer, denom):
        self._numer = numer
        self._denom = denom

    @property
    def numer(self):
        return self._numer

    @property
    def denom(self):
        return self._denom

    def __str__(self):
        return "{}/{}".format(self._numer, self._denom)


class StringNumber(object):
    """A string representing a number, that can also be sympified."""

    def __init__(self, value, join_number_words_with_hyphens=True):
        """Initializes a `StringNumber`.

        Args:
          value: An integer or rational.
          join_number_words_with_hyphens: Whether to join the words in integers with
              hyphens when describing as a string.
        """
        self._join_number_words_with_hyphens = join_number_words_with_hyphens
        self._sympy_value = sympy.sympify(value)
        self._string = self._to_string(value)

    def _integer_to_words(self, integer):
        """Converts an integer to a list of words."""
        if integer < 0:
            raise ValueError("Cannot handle negative numbers.")

        if integer < 20:
            return [_INTEGER_LOW[integer]]

        words = None

        if integer < 100:
            tens, ones = divmod(integer, 10)
            if ones > 0:
                return [_INTEGER_MID[tens], _INTEGER_LOW[ones]]
            else:
                return [_INTEGER_MID[tens]]

        for value, word in _INTEGER_HIGH:
            if integer >= value:
                den, rem = divmod(integer, value)
                words = self._integer_to_words(den) + [word]
                if rem > 0:
                    if rem < 100:
                        words.append("and")
                    words += self._integer_to_words(rem)
                return words

    def _rational_to_string(self, rational):
        """Converts a rational to words, e.g., "two thirds"."""
        numer = sympy.numer(rational)
        denom = sympy.denom(rational)

        numer_words = _get_numer(numer)

        if denom == 1:
            return numer_words

        # Знаменатель дроби в женском роде.
        # В случае если числитель > 1 множе ственное число + род. падеж.
        if numer % 10 == 1:
            denom_word = StringOrdinal(denom).str_by_form('femn', 'nomn')
        else:
            denom_word = StringOrdinal(denom).str_by_form('femn', 'gent', False)

        return "{} {}".format(numer_words, denom_word)

    def _to_string(self, number):
        """Converts an integer or rational to words."""
        if isinstance(number, sympy.Integer) or isinstance(number, int):
            return num2words(number, lang=_LANG)
        elif isinstance(number, sympy.Rational):
            return self._rational_to_string(number)
        else:
            raise ValueError(
                "Unable to handle number {} with type {}.".format(number, type(number))
            )

    def _sympy_(self):
        return self._sympy_value

    def __str__(self):
        return self._string


class StringOrdinal(object):
    """A string representation of an ordinal, e.g., "first"."""

    def __init__(self, position):
        """Initializes a `StringOrdinal`.

        Args:
          position: An integer >= 0.

        Raises:
          ValueError: If `position` is non-positive or out of range.
        """
        if position < 0 or position >= 1000:
            raise ValueError("Unsupported ordinal {}.".format(position))
        self.pos = position
        self._string = self.str_by_form('male', 'nomn')

    def str_by_form(self, gen='male', case='nomn', is_sing=True):
        l = self.pos % 100
        n = self.pos - l
        words = [num2words(n, lang='ru')] if n > 0 else []
        if l >= 20:
            nl = l % 10
            words.append(num2words(l - nl, lang='ru'))
            l = nl
        words.append(_get_ord_cases(l, gen, case, is_sing))
        return ' '.join(words)

    def __str__(self):
        return self._string


class NumberList(object):
    """Contains a list of numbers, intended for display."""

    def __init__(self, numbers):
        self._numbers = numbers

    def __str__(self):
        """Converts the list to a string.

        Returns:
          Human readable string.

        Raises:
          ValueError: if any of the strings contain a comma and thus would lead to
              an ambigious representation.
        """
        strings = []
        for number in self._numbers:
            string = str(number)
            if "," in string:
                raise ValueError(
                    "String representation of the list will be ambigious, "
                    'since term "{}" contains a comma.'.format(string)
                )
            strings.append(string)
        return ", ".join(strings)


class NumberInBase(object):
    """Contains value, represented in a given base."""

    def __init__(self, value, base):
        """Initializes a `NumberInBase`.

        Args:
          value: Positive or negative integer.
          base: Integer in the range [2, 36].

        Raises:
          ValueError: If base is not in the range [2, 36] (since this is the limit
              that can be represented by 10 numbers plus 26 letters).
        """
        if not 2 <= base <= 36:
            raise ValueError("base={} must be in the range [2, 36]".format(base))
        self._value = value
        self._base = base

        chars = []
        remainder = abs(value)
        while True:
            digit = remainder % base
            char = str(digit) if digit <= 9 else chr(ord("a") + digit - 10)
            chars.append(char)
            remainder = int(remainder / base)
            if remainder == 0:
                break
        if value < 0:
            chars.append("-")

        self._str = "".join(reversed(chars))

    def __str__(self):
        return self._str

    def _sympy_(self):
        return self._value
