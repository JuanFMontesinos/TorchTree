import keyword

RESERVED = keyword.kwlist
RESERVED.extend(list(object.__dict__.keys()))
RESERVED.extend(['__pycache__'])

from . import core, trees
from .core import Tree
from .trees import Directory_Tree

