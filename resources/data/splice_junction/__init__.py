from pathlib import Path

PATH = Path(__file__).parents[0]


def get_indices() -> list[int]:
    return list(range(-30, 0)) + list(range(1, 31))
