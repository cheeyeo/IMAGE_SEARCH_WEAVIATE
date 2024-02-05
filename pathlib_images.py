from pathlib import Path


# Using pathlib to scan for images in given directory
# Scans and locates sub-dirs for images only


if __name__ == '__main__':
    path = Path('dataset')
    for x in path.rglob('**/*.jpg'):
        if x.is_file():
            print(f'FILE FOUND: {x}')