from pathlib import Path
from resources.data.promoters import get_indices
from resources.rules import *
from resources.rules.utils import *

PATH = Path(__file__).parents[0]
INDEX_IDENTIFIER = 'p'


def parse_clause(rest: str, rhs: str = '', aggregation: str = AND_SYMBOL) -> str:
    for j, clause in enumerate(rest.split(',')):
        index = re.match(INDEX_IDENTIFIER + '[-]?[0-9]*', clause)
        if index is not None:
            index = clause[index.regs[0][0]:index.regs[0][1]]
            clause = clause[len(index):]
            clause = re.sub('=', '', clause)
            index = index[1:]
            rhs += aggregation.join(explicit_variables(
                VARIABLE_BASE_NAME + ('_' if next_index(index, get_indices(), i) < 0 else '') +
                str(abs(next_index(index, get_indices(), i))) +
                ' = ' + value.lower()) for i, value in enumerate(clause))
        else:
            rhs += re.sub('-', '_', clause.lower()) + '()'
        if j < len(rest.split(',')) - 1:
            rhs += AND_SYMBOL
    return rhs