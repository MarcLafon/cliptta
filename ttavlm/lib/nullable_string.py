from typing import Union


def nullable_string(val: Union[None, str]) -> Union[None, str]:
    if not val:
        return None
    return val
