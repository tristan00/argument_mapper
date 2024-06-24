import os

from common import base_dir


#TODO: download other utilities like gecko, llama...

def create_directory_structure():
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    directories = [
        'webpages', 'raw_arguments', 'lda_model', 'dictionary', 'posts', 'llm'
    ]

    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")


if __name__ == '__main__':
    create_directory_structure()