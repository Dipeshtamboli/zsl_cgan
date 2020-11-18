import numpy as np
import pdb
import json
import scipy.io as sio
import torch

label_to_id = json.load(open("all_seen_unseen_labs.json"))
split_1 = sio.loadmat("att_splits.mat")
att = split_1["att"]
# att = torch.tensor(att)
# att = torch.transpose(att,1,0)

att = np.transpose(att)
# att = att.numpy()

label_to_sem_vec = {}

seen_semantic_51 = np.zeros((51,300))
unseen_semantic_50 = np.zeros((50,300))
# for i in range(len(att)):

for label in label_to_id.keys():
	label_to_sem_vec[label] = att[label_to_id[label][0]-1].tolist()
	if label_to_id[label][1] != -1:
		seen_semantic_51[label_to_id[label][1]-1] = att[label_to_id[label][0]-1]
	if label_to_id[label][2] != -1:
		unseen_semantic_50[label_to_id[label][2]-1] = att[label_to_id[label][0]-1]

np.save(f"unseen_semantic_50.npy", unseen_semantic_50)
np.save(f"seen_semantic_51.npy", seen_semantic_51)
json.dump(label_to_sem_vec, open( "label_to_sem_vec.json", 'w' ))
pdb.set_trace()