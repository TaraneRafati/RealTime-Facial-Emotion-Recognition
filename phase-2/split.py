import os
import random
import shutil

for i in range(7):
    input_dir = f'./images/{i}'
    output_dir_20 = f'./images/test/{i}'
    output_dir_80 = f'./images/train/{i}'
    os.makedirs(output_dir_20, exist_ok=True)
    os.makedirs(output_dir_80, exist_ok=True)

    file_list = os.listdir(input_dir)
    num_files = len(file_list)
    num_files_20 = int(0.2 * num_files)
    files_20_percent = random.sample(file_list, num_files_20)

    for filename in file_list:
        src_path = os.path.join(input_dir, filename)
        if filename in files_20_percent:
            dst_path = os.path.join(output_dir_20, filename)
        else:
            dst_path = os.path.join(output_dir_80, filename)
        shutil.copy(src_path, dst_path)