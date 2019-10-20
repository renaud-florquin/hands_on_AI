import os
from collections import defaultdict
import hashlib

def file_as_bytes(file):
    with file:
        return file.read()

def scan_and_remove_duplicate(from_dir, debug=False):
    """ Remove all duplicate files of a directories based on SHA1. 
    :param from_dir: the directory with files
    :param debug: if debug=True the duplicated files are no removed
    """
    all_files = defaultdict(list)
    for root, dirs, files in os.walk(from_dir):
        for file in files:
            full_path = os.path.join(root, file)
            md5 = hashlib.md5(file_as_bytes(open(full_path, 'rb'))).hexdigest()
            all_files[md5].append(full_path)
    for v in all_files.values():
        duplicated_files = v[1:]
        if duplicated_files:
            for f in duplicated_files:
                print('Remove diplicated file: {} (original: {})'.format(f, v[0]))
                if not debug:
                    os.remove(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Remove all duplicated files from a directory and sub-directories.')
    parser.add_argument('from_dir', type=str, help='a existing directory with files (sub-directories)')

    args = parser.parse_args()
    
    scan_and_remove_duplicate(args.from_dir)
