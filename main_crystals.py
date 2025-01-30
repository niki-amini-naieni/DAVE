import os

def list_files_recursive(path='.'):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            print(full_path)

# Specify the directory path you want to start from
directory_path = './'
list_files_recursive(directory_path)
