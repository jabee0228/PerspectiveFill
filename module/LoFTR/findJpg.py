import os

def pair_jpg_files(directory):
    jpg_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]
    paired_files = []
    for i in range(len(jpg_files)):
        for j in range(len(jpg_files)):
            if i != j:
                paired_files.append((jpg_files[i], jpg_files[j]))
    return paired_files

def returnPair(path):

    paired_files = pair_jpg_files(path) 
    return paired_files

def checkDirectory(input_path):
    if (os.path.exists(input_path)):
        return True
    else:
        return False

def combine_filenames(filename_a, filename_b):
    # 提取文件名和扩展名
    name_a, ext_a = os.path.splitext(filename_a)
    name_b, ext_b = os.path.splitext(filename_b)
    

    
    # 合并前缀
    new_prefix = name_a + "-" + name_b

    # 合并后缀
    new_extension = ".jpg"
    
    # get new file name
    new_filename = new_prefix + new_extension

    # print(new_filename, new_prefix)
    
    return new_filename

def getDirs(input_path):
    dirs = []
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isdir(item_path):
            dirs.append(item_path)
    return dirs
