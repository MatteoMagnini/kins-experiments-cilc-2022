import re
from pathlib import Path
from typing import Iterable

PATH = Path(__file__).parents[0]
ALPHABET = ('A', 'C', 'G', 'T')
RULE_DEFINITION_SYMBOLS = ('::-', ':-')
RULE_DEFINITION_SYMBOLS_REGEX = '(' + '|'.join(RULE_DEFINITION_SYMBOLS) + ')'
INDEX_IDENTIFIER = '@'
AGGREGATE_SYMBOLS = {'Y':('C', 'T')}
NOT_IDENTIFIER = 'not'
VARIABLE_BASE_NAME = 'X'
AND_SYMBOL = ' ∧ '
NOT_SYMBOL = '¬'
LESS_SYMBOL = ' < '
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


def get_datalog_rules(rules: Iterable[str], labels: set[str] = ('ei', 'ie')) -> Iterable[str]:
    results = []

    for rule in rules:
        rule = re.sub(r' |\.', '', rule)
        name, _, rest = re.split(RULE_DEFINITION_SYMBOLS_REGEX, rule)
        name = re.sub('-', '_', name.lower())
        if name in labels:
            name = 'class(' + name + ')'
        else:
            name = name + '(' + ')'
        # lhs = name + '(' + ','.join(VARIABLE_BASE_NAME + ('_' if i < 0 else '') + str(abs(i)) for i in indices) + ')'
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
            rhs += aggregation.join(_explicit_variables(VARIABLE_BASE_NAME + ('_' if int(index) + i < 0 else '') +
                                    str(abs(int(index)) + i) + ' = ' + value.lower()) for i, value in enumerate(clause))
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
            rhs += n + LESS_SYMBOL + '(' + inner_clause + ')'
        else:
            rhs += re.sub('-', '_', clause.lower()) + '()'
        if j < len(rest.split(',')) - 1:
            rhs += AND_SYMBOL
    return rhs


def _explicit_variables(e: str) -> str:
    result = ''
    for key in AGGREGATE_SYMBOLS.keys():
        if key.lower() in e:
            values = [v for v in ALPHABET if v not in AGGREGATE_SYMBOLS[key]]
            result += AND_SYMBOL.join(NOT_SYMBOL + '(' + re.sub(key.lower(), value.lower(), e) + ')' for value in values)
    return '(' + result + ')' if result != '' else e
