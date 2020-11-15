import pdb
import numpy as np
import csv
import json
import os 
import scipy.io as sio
import shutil
import pprint as p

def cout(anything):
	p.pprint(anything)

def load_labels_jason_file():
    dict_id_to_label = json.load( open( "semantic_data/dict_id_to_label.json" ) )
    return dict_id_to_label

def create_dictionary_of_labels_from_ucf_textfile():
    labels_id = open("ucf_labels.txt","r")
    dict_id_to_label ={}
    for i in labels_id:
      dict_id_to_label[(i.split(' ')[1]).split('\n')[0]] = int((i.split(' ')[0]).split('\n')[0])
    print(dict_id_to_label)
    json.dump( dict_id_to_label, open( "dict_id_to_label.json", 'w' ) )
    return dict_id_to_label

def check_split():
    split_1 = sio.loadmat("/home/SharedData/fabio/zsl_cgan/semantic_data/ucf101_i3d/split_1/att_splits.mat")
    for i in split_1:
        print(i)
    print("----------")
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
    # pdb.set_trace()
    return split_1["trainval_loc"],split_1["test_seen_loc"],split_1["test_unseen_loc"]

def create_all_vid_list_from_split(label_dict):
    all_vid = []
    count = 1
    with open("splitted.csv",'w') as f:
        for (dirs,a,b) in os.walk('/home/SharedData/fabio/data/UCF-101/16_frames_split/train'): 
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
            # if len(all_vid) > 100:
            #   break
            print ('--------------------------------')

    print("_______________________________")
    print("len of vid:",len(all_vid))
    for i in all_vid:
      print(i)
    # print(all_vid)

if __name__ == '__main__':
	dict_id_to_label = load_labels_jason_file()
	trainval_loc,test_seen_loc,test_unseen_loc = check_split()
	create_all_vid_list_from_split(dict_id_to_label)