import os
import sys


def path_not_found_exit(path):
    if path is str:
        path = [path]
    for p in path:
        if not os.path.isfile(p) and not os.path.isdir(p):
            print(f'Der Pfad \"{p}\" existiert nicht...')
            sys.exit(0)
