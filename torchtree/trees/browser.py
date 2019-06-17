from ..core.core import Tree
from .. import RESERVED

from os import scandir
import os

__all__ = ['Directory_Tree']


def scantree(path, tree):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.name[0] != '.' and os.path.splitext(entry.name)[0] not in RESERVED:
            if entry.is_dir(follow_symlinks=False):
                tree.add_module(entry.name, Directory_Tree())
                yield from scantree(entry.path, getattr(tree, entry.name))
            else:
                tree.register_parameter(os.path.splitext(entry.name)[0], os.path.splitext(entry.name)[1])
                yield entry


class Directory_Tree(Tree):
    def __init__(self, path=None):
        super(Directory_Tree, self).__init__()
        if path is not None:
            list(scantree(path, self))

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                memo.add(v)
                name = module_prefix + ('/' if module_prefix else '') + k
                yield name, v

    def named_modules(self, memo=None, prefix=''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('/' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def clone_tree(self, path):
        r"""
        Clones the tree directory into given path.

        :param path: Relative root in which tree directory will be cloned
        :return: None
        """
        for module, _ in self.named_modules():
            _path = os.path.join(path, module)
            if module != '' and not os.path.exists(_path):
                os.mkdir(_path)

    def paths(self, root='', recurse=True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            root (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=root, recurse=recurse)
        for elem in gen:
            yield elem[0] + elem[1]
