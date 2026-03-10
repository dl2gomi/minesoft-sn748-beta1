from enum import Enum


class SubdivisionMode(str, Enum):
    EDGE: str = "egde"
    FACE: str = "face"