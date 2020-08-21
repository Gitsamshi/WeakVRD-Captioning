### Data

1. coco_cmb_vrg_final.zip:  unzip it, used for training. (convert from coco_cmb_vrg by merging dictionary)
2. cocotalk_final.json:  split and dictionary infos, used for training
3. cocotalk_label.h5: captions and tags, used for training
4. coco-train-idxs.p and coco-train-words.p are used in RL training
5. Please refer to [SGAE](https://github.com/yangxuntu/SGAE) for cocobu_att, cocobu_fc, cocobu_box directories.



Other files are intermediate/resource files in training.

1. coco_cmb_vrg and coco_dict.json: weakly supervised vrg along with its dictionary (offset -1 to its index for use)
2. coco_class_names: category to super-category mapping








