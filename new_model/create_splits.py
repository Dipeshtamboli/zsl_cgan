import shutil
import os 
import pdb
# sftp://bbanerjee@10.107.35.16
for (dirs,a,b) in os.walk('/home/SharedData/fabio/zsl_cgan/ucf_split1/train'): 
	a.sort()
	count = 0
	for folder in a[10:20]:
		for mode in ["train", "test_seen"]:
			dir_add = f"/home/SharedData/fabio/zsl_cgan/ucf_split1/{mode}/" + folder
			# pdb.set_trace()
			dest = f"/home/SharedData/fabio/zsl_cgan/ucf_split1/data_2_{mode.split('_')[0]}/" + folder
			print(f"\ndir_add: {dir_add}\ndest: {dest} \n")
			destination = shutil.copytree(dir_add, dest)
		count += 1

	break
print(f"folders copied: {count}")