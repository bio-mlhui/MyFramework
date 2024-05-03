import os
scene_ids = []
with open(os.path.join(ov_root, 'kiuisobj_v1_merged_80K.csv'), 'r') as f:
    for line in f.readlines():
        scene_ids.append(line.strip().split(',')[-1])