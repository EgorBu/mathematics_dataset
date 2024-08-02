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

"""Measurement questions, e.g., "How many hours are there in a day?"."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import random

# Dependency imports
from mathematics_dataset import example
from mathematics_dataset.modules import train_test_split
from mathematics_dataset.sample import number
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
from mathematics_dataset.util.unit_cases import UNIT_CASES
from mathematics_dataset.util.display import Decimal
import six
import sympy


def _make_modules(is_train):
    """Returns modules, with split based on the boolean `is_train`."""
    return {
        "conversion": functools.partial(
            conversion, is_train=is_train, is_extrapolation=False
        ),
        "time": functools.partial(time, is_train=is_train),
    }


def train(entropy_fn):
    """Returns dict of training modules."""
    del entropy_fn  # unused
    return _make_modules(is_train=True)


def test():
    """Returns dict of testing modules."""
    return _make_modules(is_train=False)


def test_extra():
    """Returns dict of extrapolation testing modules."""
    return {
        "conversion": functools.partial(
            conversion, is_train=False, is_extrapolation=True
        ),
    }


Unit = collections.namedtuple("Unit", ("name", "symbol"))


MICRO_SYMBOL = "u"


LENGTH = {
    Unit("метр", "m"): 1,
    Unit("километр", "km"): 1000,
    Unit("сантиметр", "cm"): sympy.Rational(1, 100),
    Unit("миллиметр", "mm"): sympy.Rational(1, 1000),
    Unit("микрометр", "um"): sympy.Rational(1, 1e6),
    Unit("нанометр", "nm"): sympy.Rational(1, 1e9),
}

TIME = {
    Unit("секунда", "s"): 1,
    Unit("минута", None): 60,
    Unit("час", None): 60 * 60,
    Unit("день", None): 24 * 60 * 60,
    Unit("неделя", None): 7 * 24 * 60 * 60,
    Unit("миллисекунда", "ms"): sympy.Rational(1, 1e3),
    Unit("микросекунда", MICRO_SYMBOL + "s"): sympy.Rational(1, 1e6),
    Unit("наносекунда", "ns"): sympy.Rational(1, 1e9),
}

TIME_YEARLY = {
    Unit("год", None): 1,
    Unit("десятилетие", None): 10,
    Unit("век", None): 100,
    Unit("тысячелетие", None): 1000,
    Unit("месяц", None): sympy.Rational(1, 12),
}

MASS = {
    Unit("килограмм", "kg"): 1,  # Yes, the *kilo*gram is the SI base unit.
    Unit("тонна", "t"): 1000,
    Unit("грамм", "g"): sympy.Rational(1, 1e3),
    Unit("миллиграмм", "mg"): sympy.Rational(1, 1e6),
    Unit("микрограмм", MICRO_SYMBOL + "g"): sympy.Rational(1, 1e9),
    Unit("нанограмм", "ng"): sympy.Rational(1, 1e12),
}

VOLUME = {
    Unit("литр", "l"): 1,
    Unit("миллилитр", "ml"): sympy.Rational(1, 1000),
}


DIMENSIONS = [
    LENGTH,
    TIME,
    TIME_YEARLY,
    MASS,
    VOLUME
]


def set_form(root, number, case):
    return UNIT_CASES[root][number][case]


def base_form(root, value, case):
    res = None
    if isinstance(value, Decimal):
        value = value._value
    if isinstance(value, int) or value.is_integer:
        if value % 100 > 20:
            d = int(value) % 10
        else:
            d = int(value) % 20

        if d == 1:
            res = set_form(root, 'sing', case)
        elif case in ['gent', 'datv', 'ablt', 'loct']:
            res = set_form(root, 'plur', case)
        else:
            if d in (2, 3, 4):
                res = set_form(root, 'sing', 'gent')
            else:
                res = set_form(root, 'plur', 'gent')
    else:
        res = set_form(root, 'sing', 'gent')

    return res


def pluralize(name):
    if name == "century":
        return "centuries"
    if name == "millennium":
        return "millennia"
    return name + "s"


def _factor_non_decimal(value):
    """Extras x dividing value such that x is coprime to 2 and 5."""
    result = 1
    factors = sympy.factorint(value)
    for factor, power in six.iteritems(factors):
        if factor not in [2, 5]:
            result *= factor**power
    return result


def _sample_conversion_decimal(dimension, is_extrapolation):
    """Samples to and from units and values."""
    base_unit, target_unit = random.sample(list(dimension.keys()), 2)
    scale = sympy.Rational(dimension[base_unit]) / dimension[target_unit]
    scale_non_decimal = _factor_non_decimal(sympy.denom(scale))
    entropy = 9 if is_extrapolation else 7
    base_value = number.non_integer_decimal(entropy, signed=False)
    base_value = display.Decimal(base_value.value * scale_non_decimal)
    target_value = display.Decimal(base_value.value * scale)
    return base_value, base_unit, target_value, target_unit


def _conversion_decimal(context, is_train, is_extrapolation):
    """E.g., "How many grams are in 5kg?"."""
    dimension = random.choice(DIMENSIONS)
    while True:
        base_value, base_unit, target_value, target_unit = _sample_conversion_decimal(
            dimension, is_extrapolation
        )
        if train_test_split.is_train(base_value) == is_train:
            break

    # print(base_value, base_unit, target_value, target_unit)
    templates = [
        (
            "Сколько {target_name} содержат {base_value} {base_name}?",
            ('plur', 'gent'),
            'loct'
        ),
        (
            "Чему равно {base_value} {base_name} в {target_name}?",
            ('plur', 'loct'),
            'gent'
        ),
        (
            "Переведи {base_value} {base_name} в {target_name}.",
            ('plur', 'accs'),
            'gent'
        ),
    ]
    if base_unit.symbol is not None:
        templates += [
            (
                "Сколько {target_name} содержится в {base_value}{base_symbol}?",
                ('plur', 'gent'),
                'loct'
            ),
            (
                "Чему равно {base_value}{base_symbol} в {target_name}?",
                ('plur', 'loct'),
                'gent'
            ),
            (
                "Переведи {base_value}{base_symbol} в {target_name}.",
                ('plur', 'accs'),
                'gent'
            ),
        ]
    template, form_target, case_base = random.choice(templates)

    # base_name = pluralize(base_unit.name)
    base_name = base_form(base_unit.name, base_value, case_base)
    # target_name = pluralize(target_unit.name)
    target_name = set_form(target_unit.name, *form_target)

    question = example.question(
        context,
        template,
        base_name=base_name,
        base_symbol=base_unit.symbol,
        base_value=base_value,
        target_name=target_name,
    )
    return example.Problem(question=question, answer=target_value)


def _conversion_fraction(context, is_train):
    """E.g., "How many grams are in three quarters of a kg?"."""
    dimension = random.choice(DIMENSIONS)

    # Limit probability of giving zero answer.
    allow_zero = random.random() < 0.2

    # Repeat until we find a pair with an integral answer. (Avoids ambiguity with
    # decimals.)
    while True:
        base_unit, target_unit = random.sample(list(dimension.keys()), 2)
        base_value = number.non_integer_rational(2, signed=False)
        if train_test_split.is_train(base_value) != is_train:
            continue
        answer = (
            base_value
            * sympy.Rational(dimension[base_unit])
            / sympy.Rational(dimension[target_unit])
        )
        if (
            abs(answer) <= 100000
            and sympy.denom(answer) == 1
            and (allow_zero or answer != 0)
        ):
            break

    template, case = random.choice(
        [
            ("Сколько {target_name} содержат {base_value} {base_name}?", 'gent'),
            ("Чему равно {base_value} {base_name} в {target_name}?", 'loct'),
        ]
    )

    if sympy.denom(base_value) > 20 or random.choice([False, True]):
        base_value_string = base_value  # Will be represented as e.g., 2/3.
    else:
        base_value_string = display.StringNumber(base_value)  # e.g., two thirds

    # base_name = set_form(base_unit.name, 'sing', 'gent')
    base_name = base_form(base_unit.name, base_value, 'gent')
    target_name = set_form(target_unit.name, 'plur', case)

    question = example.question(
        context,
        template,
        base_name=base_name,
        base_value=base_value_string,
        target_name=target_name,
    )
    return example.Problem(question=question, answer=answer)


def conversion(is_train, is_extrapolation):
    """Conversion question, in decimal or fraction."""
    context = composition.Context()
    # TODO(b/124038528): implement extrapolation for fraction conversions too
    if is_extrapolation or random.choice([False, True]):
        return _conversion_decimal(
            context, is_train=is_train, is_extrapolation=is_extrapolation
        )
    else:
        return _conversion_fraction(context, is_train=is_train)


def time(is_train):
    """Questions for calculating start, end, or time differences."""
    context = composition.Context()
    start_minutes = random.randint(1, 24 * 60 - 1)
    while True:
        duration_minutes = random.randint(1, 12 * 60 - 1)
        if train_test_split.is_train(duration_minutes) == is_train:
            break
    end_minutes = start_minutes + duration_minutes

    def format_12hr(minutes):
        """Format minutes from midnight in 12 hr format."""
        hours = (minutes // 60) % 24
        minutes %= 60
        am_pm = "AM" if hours < 12 else "PM"
        hours = (hours - 1) % 12 + 1
        return "{}:{:02} {}".format(hours, minutes, am_pm)

    def format_24hr(minutes):
        hours = (minutes // 60) % 24
        minutes %= 60
        return f"{hours}:{minutes:02d}"

    start = format_24hr(start_minutes)
    end = format_24hr(end_minutes)

    which_question = random.randint(0, 3)
    if which_question == 0:
        # Question: What is start = end - duration?
        template = random.choice(
            [
                "Сколько времени было за {duration} {unit} до {end}?",
            ]
        )
        unit = base_form('минута', duration_minutes, 'accs')
        return example.Problem(
            question=example.question(
                context,
                template,
                duration=duration_minutes,
                end=end,
                unit=unit
            ),
            answer=start,
        )
    elif which_question == 1:
        # Question: What is end = start + duration?
        template = random.choice(
            [
                "Сколько времени будет через {duration} {unit} после {start}?",
            ]
        )
        unit = base_form('минута', duration_minutes, 'accs')
        return example.Problem(
            question=example.question(
                context, template,
                duration=duration_minutes,
                start=start,
                unit=unit
            ),
            answer=end,
        )
    else:
        # Question: What is duration = end - start?
        template = random.choice(
            [
                "Сколько минут между {start} и {end}?",
            ]
        )
        return example.Problem(
            question=example.question(context, template, start=start, end=end),
            answer=duration_minutes,
        )
