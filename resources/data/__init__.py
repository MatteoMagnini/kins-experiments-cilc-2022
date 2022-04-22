import re
from itertools import chain
from pathlib import Path
from typing import Iterable
import pandas as pd

PATH = Path(__file__).parents[0]


def get_data(filename: str) -> pd.DataFrame:
    x = []
    with open(str(PATH / filename) + '.txt') as file:
        for row in file:
            row = re.sub('\n', '', row)
            label, _, features = row.split(',')
            features = list(f for f in features.lower())
            features.append(label.lower())
            x.append(features)
    return pd.DataFrame(x)


def get_binary_data(data: pd.DataFrame, mapping: dict[str: set[str]]) -> pd.DataFrame:
    sub_features = sorted(_get_values(mapping))
    results = []
    for _, row in data.iterrows():
        row_result = []
        for value in row:
            positive_features = mapping[value]
            for feature in sub_features:
                row_result.append(1 if feature in positive_features else 0)
        results.append(row_result)
    return pd.DataFrame(results, dtype=int)


def _get_values(mapping: dict[str: set[str]]) -> Iterable[str]:
    result = set()
    for values_set in mapping.values():
        for value in values_set:
            result.add(value)
    return result


def data_to_int(data: pd.DataFrame, mapping: dict[str: int]) -> pd.DataFrame:
    return data.applymap(lambda x: mapping[x] if x in mapping.keys() else x)


def get_indices() -> list[int]:
    return list(range(-30, 0)) + list(range(1, 31))


def get_feature_mapping(variable_indices: list[int] = get_indices()) -> dict[str: int]:
    return {'X' + ('_' if j < 0 else '') + str(abs(j)): i for i, j in enumerate(variable_indices)}


def get_extended_feature_mapping(features: list[str], variable_indices: list[int] = get_indices()) -> dict[str: int]:
    return {'X' + ('_' if j < 0 else '') + str(abs(j)) + f: k + i * len(features) for i, j in enumerate(variable_indices) for k, f in enumerate(features)}


def get_vocabulary(data: pd.DataFrame) -> list[str]:
    result = set()
    for _, row in data.iterrows():
        for value in row:
            result.add(value)
    return sorted(result)

