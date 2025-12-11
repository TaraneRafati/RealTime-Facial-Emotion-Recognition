import os
import random
import shutil

for i in range(7):
    input_dir = f'./dataset/{i}'
    output_dir_test = f'./dataset/test/{i}'
    output_dir_validation = f'./dataset/validation/{i}'
    output_dir_train = f'./dataset/train/{i}'
    
    os.makedirs(output_dir_test, exist_ok=True)
    os.makedirs(output_dir_validation, exist_ok=True)
    os.makedirs(output_dir_train, exist_ok=True)

    file_list = os.listdir(input_dir)
    num_files = len(file_list)
    num_files_test = int(0.15 * num_files)
    num_files_validation = int(0.15 * num_files)

    # randomly select files for test and validation sets
    files_test = random.sample(file_list, num_files_test)
    remaining_files = list(set(file_list) - set(files_test))
    files_validation = random.sample(remaining_files, num_files_validation)
    files_train = list(set(remaining_files) - set(files_validation))

    # copy files to respective directories
    for filename in files_test:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir_test, filename)
        shutil.copy(src_path, dst_path)

    for filename in files_validation:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir_validation, filename)
        shutil.copy(src_path, dst_path)

    for filename in files_train:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir_train, filename)
        shutil.copy(src_path, dst_path)
