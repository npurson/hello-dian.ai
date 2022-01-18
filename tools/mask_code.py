import os
import re
import argparse


def mask_code(path):
    print(path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        pattern = re.compile(r'^( *)(# TODO.*)((\s.*?#.*$)*)([\s\S]*?)(# End of todo)$', re.M)
        text = re.sub(pattern, r'\1\2\3\n\n\1...\n\n\1\6', text)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def walk_files(dirs: list):
    for dir in dirs:
        if os.path.isfile(dir) and dir.endswith('.py'):
            mask_code(dir)
        elif os.path.isdir(dir):
            for root, dir, files in os.walk(dir):
                walk_files([os.path.join(root, f) for f in files])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', default=['lab0', 'lab1', 'lab2'], nargs='+')
    args = parser.parse_args()
    walk_files(args.dirs)
