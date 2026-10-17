import glob
import os
import random

mpdd_classes = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

MPDD_DIR = '../../PromptAD/datasets/anomaly_detection/MPDD'


def load_mpdd(category, k_shot):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '_mask.png') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in mpdd_classes

    test_img_path = os.path.join(MPDD_DIR, category, 'test')
    train_img_path = os.path.join(MPDD_DIR, category, 'train')
    ground_truth_path = os.path.join(MPDD_DIR, category, 'ground_truth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    # MPDD 数据集的 train 目录只有 good 样本，所以需要从 good 样本中选择
    # 种子文件中的索引是针对 good 样本的编号（如 000, 001, 108 等）
    good_train_paths = [p for p, t in zip(train_img_tot_paths, train_tot_types) if t == 'good']
    good_train_paths.sort()  # 确保按文件名排序
    
    seed_file = os.path.join('./datasets/seeds_mpdd', category, 'selected_samples_per_run.txt')
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    training_indx = []
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]
            index = strip_line.split(' ')
            training_indx = [int(item) for item in index]

    # 根据种子文件中的索引选择 good 样本
    selected_train_img_tot_paths = [good_train_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [0] * len(selected_train_img_tot_paths)
    selected_train_tot_labels = [0] * len(selected_train_img_tot_paths)
    selected_train_tot_types = ['good'] * len(selected_train_img_tot_paths)

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)


