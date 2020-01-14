import os
import sys

def path_not_found_exit(path):
    if not os.path.isfile(path):
        print(f'Der Pfad \"{path}\" existiert nicht...')
        sys.exit(0)