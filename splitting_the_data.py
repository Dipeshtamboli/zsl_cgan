import pdb
import numpy as np
import csv
import json
import os 
import scipy.io as sio
import shutil
# 5372+6606+1342
def create_dictionary_of_labels():
    # labels_id = open("new_split16_label.txt","r")

    # dict_id_to_label ={}
    # for i in labels_id:
    #   # print((i.split(' ')[0]).split('\n')[0])
    #   # print((i.split(' ')[1]).split('\n')[0])
    #   dict_id_to_label[(i.split(' ')[1]).split('\n')[0]] = int((i.split(' ')[0]).split('\n')[0])
    # # print(labels_id)
    dict_id_to_label = json.load( open( "dict_id_to_label.json" ) )
    # print(dict_id_to_label)
    # json.dump( dict_id_to_label, open( "dict_id_to_label.json", 'w' ) )
    # exit()
    return dict_id_to_label

def check_split():

    split_1 = sio.loadmat("/home/SharedData/fabio/cgan/hmdb_i3d/split_1/att_splits.mat")
    for i in split_1:
        print(i)
    print("----------")
    print(split_1.keys())
    print("----------")
    # print(split_1["allclasses_names"].shape)
    # print("----------")
    # print(att)
    print()
    att = split_1["att"]
    print("allclasses_names",split_1["allclasses_names"].shape)
    print("train_loc",split_1["train_loc"].shape)
    # print("train_loc",split_1["train_loc"])
    print("test_unseen_loc",split_1["test_unseen_loc"].shape)
    print("trainval_loc",split_1["trainval_loc"].shape)
    print("test_seen_loc",split_1["test_seen_loc"].shape)
    print("val_loc",split_1["val_loc"].shape)
    print("original_att",split_1["original_att"].shape)
    print("att.shape:",att.shape)
    print("_________________________________________________________")
    pdb.set_trace()
    return split_1["trainval_loc"],split_1["test_seen_loc"],split_1["test_unseen_loc"] 


def create_all_vid_list_from_split(label_dict):

    all_vid = []
    count = 1

    with open("splitted.csv",'w') as f:
        for (dirs,a,b) in os.walk('/home/dipesh/ritesh/data/UCF-101/16_frames_split/train'): 
            # print("count:",count)
            print (dirs )
            print('splits:',len(dirs.split("/")))
            # print (a )
            a.sort() #----------------Important line
            if len(dirs.split("/")) ==10:
                all_vid.append(dirs)
                writer = csv.writer(f, dialect='excel')
                writer.writerow([str(count)+","+ dirs+","+ dirs.split('/')[-2]+','+str(label_dict[dirs.split('/')[-2]])])
                print(str(count)+','+ dirs+","+ dirs.split('/')[-2]+','+str(label_dict[dirs.split('/')[-2]]))
                count +=1
            if len(all_vid) > 100:
              break
            print ('--------------------------------')

    print("_______________________________")
    print("len of vid:",len(all_vid))
    for i in all_vid:
      print(i)
    # print(all_vid)

def create_train_test_split_folder(csv_path):
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for r in spamreader:
            dir_add=r[0].split(',')[1]
            label = r[0].split(',')[2]
            index=int(r[0].split(',')[0])
            # print("index: ",index)
            print("dir_add: ",dir_add)
            # print("label: ",label)
            # print("--------------------------------------")

            if index in trainval_loc[:,0]:
                print(index)
                print("trainval_loc")
                dest = "/home/dipesh/ritesh/data/UCF-101/split1/train/"+label+'/'+dir_add.split('/')[-1]
                destination = shutil.copytree(dir_add, dest)
            elif index in test_seen_loc[:,0]:
                print(index)
                print("test_seen_loc")
                dest = "/home/dipesh/ritesh/data/UCF-101/split1/test_seen/"+label+'/'+dir_add.split('/')[-1]
                destination = shutil.copytree(dir_add, dest)
            elif index in test_unseen_loc[:,0]:
                print(index)
                print("test_unseen_loc")
                dest = "/home/dipesh/ritesh/data/UCF-101/split1/test_unseen/"+label+'/'+dir_add.split('/')[-1]
                destination = shutil.copytree(dir_add, dest)            
        # print(spamreader)

if __name__ == "__main__":
    # dict_id_to_label=create_dictionary_of_labels()
    trainval_loc,test_seen_loc,test_unseen_loc = check_split()
    # print(type(trainval_loc))
    # create_all_vid_list_from_split(dict_id_to_label)
    # create_train_test_split_folder("splitted.csv")
    # create_train_test_split_folder("splt1_trainval_test.csv")

