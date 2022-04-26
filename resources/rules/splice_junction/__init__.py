import re
from pathlib import Path
from typing import Iterable
from resources.data.splice_junction import get_indices
from resources.rules.utils import *

PATH = Path(__file__).parents[0]
INDEX_IDENTIFIER = '@'
NOT_IDENTIFIER = 'not'


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
            rhs += aggregation.join(explicit_variables(
                VARIABLE_BASE_NAME + ('_' if next_index(index, get_indices(), i) < 0 else '') +
                str(abs(next_index(index, get_indices(), i))) +
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
