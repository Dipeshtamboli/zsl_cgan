import os
import shutil


source_file = "/home/SharedData/fabio/data/UCF-101/split/train"
target_file = "/home/SharedData/fabio/zsl_cgan/ucf_split1"

train_test_split = 0.8
d = []

train_path = target_file + "/train"
test_path = target_file + "/test"

if os.path.exists(train_path):
    shutil.rmtree(train_path)

if os.path.exists(test_path):
    shutil.rmtree(test_path)
    
os.mkdir(train_path)
os.mkdir(test_path)

for (dir) in os.listdir(source_file):
    dir_name = source_file + '/' + dir
    train_path_name = train_path + '/' + dir
    test_path_name = test_path + '/' + dir
    os.mkdir(train_path_name)
    os.mkdir(test_path_name)
    #print(dir)

    for dir1 in os.listdir(dir_name):
        dir_name1 = dir_name + '/' + dir1
        #print(dir1)
        train_path_name = train_path + '/' + dir + '/' + dir1
        test_path_name = test_path + '/' + dir + '/' + dir1
        os.mkdir(train_path_name)
        os.mkdir(test_path_name)
        
        for (a, b, files) in os.walk(dir_name1):
            length = len(files)
            train_length = train_test_split*length
            
            for (i, file) in enumerate(files):
                file_name = dir_name1 + '/' + file
                if (i < train_length):
                    shutil.copy(file_name, train_path_name)
                else:
                    shutil.copy(file_name, test_path_name)
            
print(len(d)) 