import re
from pathlib import Path
import pandas as pd

PATH = Path(__file__).parents[0]


def get_data(filename: str) -> pd.DataFrame:
    x = []
    with open(str(PATH / filename) + '.txt') as file:
        for raw in file:
            raw = re.sub('\n', '', raw)
            label, _, features = raw.split(',')
            features = list(f for f in features.lower())
            features.append(label)
            x.append(features)
    return pd.DataFrame(x)
