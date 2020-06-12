# Document of crowd density label 

This only shows how to make crowd density label in the training set. The test set is similar.
## 1 Step1 (Make labels of density map and point map)

Density map label: map_mcnn.m
Point map label:   map_count.m

## 2 Step2 (Convert mat files to npy files)

python mat2npy.py