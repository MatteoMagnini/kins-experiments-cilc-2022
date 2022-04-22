import re
from pathlib import Path
from typing import Iterable

from resources.data import get_indices

PATH = Path(__file__).parents[0]
ALPHABET = ('A', 'C', 'G', 'T')
RULE_DEFINITION_SYMBOLS = ('::-', ':-')
RULE_DEFINITION_SYMBOLS_REGEX = '(' + '|'.join(RULE_DEFINITION_SYMBOLS) + ')'
INDEX_IDENTIFIER = '@'
AGGREGATE_SYMBOLS = {'Y': ('C', 'T'),
                     'M': ('A', 'C'),
                     'R': ('A', 'G')}
AGGREGATE_DATA_SYMBOLS = {'A': ('A', 'D', 'N', 'R'),
                          'C': ('C', 'N', 'S'),
                          'G': ('G', 'D', 'N', 'S', 'R'),
                          'T': ('T', 'D', 'N')}
NOT_IDENTIFIER = 'not'
VARIABLE_BASE_NAME = 'X'
AND_SYMBOL = ' ∧ '
OR_SYMBOL = ' ∨ '
NOT_SYMBOL = '¬'
LESS_EQUAL_SYMBOL = ' ≤ '
PLUS_SYMBOL = ' + '
IMPLICATION_SYMBOL = ' ← '


def get_rules(filename: str) -> list[str]:
    rules = []
    with open(str(PATH / filename) + '.txt') as file:
        for raw in file:
            raw = re.sub('\n', '', raw)
            if len(raw) > 0:
                rules.append(raw)
    return rules


def get_datalog_rules(rules: Iterable[str], class_labels: set[str] = ('ei', 'ie', 'n')) -> Iterable[str]:
    results = []

    for rule in rules:
        rule = re.sub(r' |\.', '', rule)
        name, _, rest = re.split(RULE_DEFINITION_SYMBOLS_REGEX, rule)
        name = re.sub('-', '_', name.lower())
        if name in class_labels:
            name = 'class(' + name + ')'
        else:
            name = name + '(' + ')'
        lhs = name
        rhs = _parse_clause(rest)
        results.append(lhs + IMPLICATION_SYMBOL + rhs)

    return results


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


# TODO: to be removed
def get_rules_with_explicit_variables(rules: Iterable[str]) -> Iterable[str]:
    result = []
    regex = r'X[_]?[0-9]* = '

    for rule in rules:
        for var in AGGREGATE_DATA_SYMBOLS:
            tmp_rule = rule
            partial_result = ''
            while re.search(regex + var.lower(), tmp_rule) is not None:
                match = re.search(regex + var.lower(), tmp_rule)
                ante = tmp_rule[:match.regs[0][0]]
                medio = _replace(var, tmp_rule[match.regs[0][0]:match.regs[0][1]])
                partial_result += ante + medio
                tmp_rule = tmp_rule[match.regs[0][1]:]
            partial_result += tmp_rule
            rule = partial_result
        result.append(rule)

    return result


# TODO: to be removed
def get_rules_with_int(rules: Iterable[str], mapping: dict[str: int]) -> Iterable[str]:
    result = []
    for rule in rules:
        for key in mapping.keys():
            rule = re.sub(' ' + key, ' ' + str(mapping[key]), rule)
        result.append(rule)
    return result


def _parse_clause(rest: str, rhs: str = '', aggregation: str = AND_SYMBOL) -> str:
    for j, clause in enumerate(rest.split(',')):
        index = re.match(INDEX_IDENTIFIER + '[-]?[0-9]*', clause)
        negation = re.match(NOT_IDENTIFIER, clause)
        n = re.match('[0-9]*of', clause)
        if index is not None:
            index = clause[index.regs[0][0]:index.regs[0][1]]
            clause = clause[len(index):]
            clause = re.sub('\'', '', clause)
            index = index[1:]
            # TODO: compute next index with check on the original variables
            rhs += aggregation.join(_explicit_variables(
                VARIABLE_BASE_NAME + ('_' if _next_index(index, i) < 0 else '') + str(abs(_next_index(index, i))) +
                ' = ' + value.lower()) for i, value in enumerate(clause))
        elif negation is not None:
            new_clause = re.sub(NOT_IDENTIFIER, NOT_SYMBOL, clause)
            new_clause = re.sub('-', '_', new_clause.lower())
            new_clause = re.sub('\)', '())', new_clause)
            rhs += new_clause
        elif n is not None:
            new_clause = clause[n.regs[0][1]:]
            new_clause = re.sub('\(|\)', '', new_clause)
            inner_clause = _parse_clause(new_clause, rhs, PLUS_SYMBOL)
            inner_clause = '(' + (')' + PLUS_SYMBOL + '(').join(e for e in inner_clause.split(PLUS_SYMBOL)) + ')'
            n = clause[n.regs[0][0]:n.regs[0][1] - 2]
            rhs += n + LESS_EQUAL_SYMBOL + '(' + inner_clause + ')'
        else:
            rhs += re.sub('-', '_', clause.lower()) + '()'
        if j < len(rest.split(',')) - 1:
            rhs += AND_SYMBOL
    return rhs


def _next_index(index: str, offset: int) -> int:
    indices = get_indices()
    new_index: int = int(index) + offset
    modified: bool = False
    while new_index not in indices:
        new_index += 1
        modified = True
    return new_index + _previous_holes(indices, indices.index(new_index)) if not modified else new_index


def _previous_holes(l: list[int], i: int) -> int:
    j = 0
    for k in list(range(0, i)):
        if l[k] + 1 != l[k + 1]:
            j += 1
    return j


def _explicit_variables(e: str) -> str:
    result = ''
    for key in AGGREGATE_SYMBOLS.keys():
        if key.lower() in e:
            values = [v for v in ALPHABET if v not in AGGREGATE_SYMBOLS[key]]
            result += AND_SYMBOL.join(
                NOT_SYMBOL + '(' + re.sub(key.lower(), value.lower(), e) + ')' for value in values)
    return '(' + result + ')' if result != '' else e


def _replace(s: str, e: str) -> str:
    values = [v for v in AGGREGATE_DATA_SYMBOLS[s]]
    return '(' + OR_SYMBOL.join('(' + re.sub(s.lower(), value.lower(), e) + ')' for value in values) + ')'
