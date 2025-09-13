# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:56:27 2025

@author: nbhogapurapu
"""
#%%
from pathlib import Path
from itertools import islice

space =  '    '
branch = '│   '
tee =    '├── '
last =   '└── '



def tree_old(dir_path: Path, level: int=-1, limit_to_directories: bool=False,
         length_limit: int=3000):
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level: 
            return # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else: 
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space 
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1
    print(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))
            

def tree(dir_path: Path, level: int=-1, limit_to_directories: bool=False,
         length_limit: int=3000, ignore: set={'__pycache__', '.git', '.venv'}):
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path)
    files = 0
    directories = 0

    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level:
            return
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir() and d.name not in ignore]
        else:
            contents = [d for d in dir_path.iterdir() if d.name not in ignore]
        contents.sort()  # Optional: sort alphabetically
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    print(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))

# tree(Path('../'), level=4)
# tree(Path('../docs/'), level=4)
tree(Path('../'), level=4, ignore={'.git','.github', '.gitignore','__pycache__', '.pytest_cache','.vscode',
                                   'wheelhouse','condapackages',
                                   '.env', 'node_modules',
                                   'dist', 'build',
                                   })
