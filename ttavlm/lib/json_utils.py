from typing import Any
import json


def save_json(obj: Any, path: str) -> None:  # noqa ANN401
    with open(path, "w") as fp:
        json.dump(obj, fp)


def load_json(path: str) -> Any:  # noqa ANN401
    with open(path) as fp:
        db = json.load(fp)
    return db
