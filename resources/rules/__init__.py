import re
from pathlib import Path
from typing import Iterable
from resources.rules.utils import AGGREGATE_SYMBOLS
from resources.rules.splice_junction import PATH as SPLICE_JUNCTION_PATH
from resources.rules.promoters import PATH as PROMOTERS_PATH
from resources.rules.utils import AGGREGATE_SYMBOLS, AGGREGATE_DATA_SYMBOLS, ALPHABET, VARIABLE_BASE_NAME, AND_SYMBOL, \
    OR_SYMBOL, NOT_SYMBOL

PATH = Path(__file__).parents[0]


def get_splice_junction_rules(filename: str) -> list[str]:
    return get_rules(str(SPLICE_JUNCTION_PATH / filename) + '.txt')


def get_promoters_rules(filename: str) -> list[str]:
    return get_rules(str(PROMOTERS_PATH / filename) + '.txt')


def get_rules(file: str) -> list[str]:
    rules = []
    with open(file) as file:
        for raw in file:
            raw = re.sub('\n', '', raw)
            if len(raw) > 0:
                rules.append(raw)
    return rules


def get_binary_datalog_rules(rules: Iterable[str]) -> Iterable[str]:
    results = []
    term_regex = '[a-z]+'
    variable_regex = VARIABLE_BASE_NAME + '[_]?[0-9]+'
    regex = variable_regex + '[ ]?=[ ]?' + term_regex
    for rule in rules:
        tmp_rule = rule
        partial_result = ''
        while re.search(regex, tmp_rule) is not None:
            match = re.search(regex, tmp_rule)
            start, end = match.regs[0]
            matched_string = tmp_rule[start:end]
            ante = tmp_rule[:start]
            medio = matched_string[:re.search(variable_regex, matched_string).regs[0][1]] + \
                    matched_string[re.search(term_regex, matched_string).regs[0][0]:] + ' = 1'
            partial_result += ante + medio
            tmp_rule = tmp_rule[end:]
        partial_result += tmp_rule
        results.append(partial_result)
    return results
