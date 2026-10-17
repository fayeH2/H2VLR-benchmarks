import glob
import os
import random

# belt皮带异常文件夹里面的 检测物体类别
belt_classes = ['belt']

# VISA数据集地址，数据集格式为MVTEC格式！！！
BELT_DIR = "../../PromptAD/datasets/anomaly_detection/belt_anomaly_detection"

def load_belt(category,k_shot):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")  # 我们的格式为jpg
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg") # 我们的格式为jpg
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '.png') for s in
                            img_paths]       # 对应的异常类型的mask标签，png格式
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in belt_classes

    test_img_path = os.path.join(BELT_DIR, category, 'test')
    train_img_path = os.path.join(BELT_DIR, category, 'train')
    ground_truth_path = os.path.join(BELT_DIR, category, 'ground_truth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels,\
    train_tot_types = load_phase(train_img_path, ground_truth_path)
    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    # 随机种子地址，暂时不用修改
    seed_file = os.path.join('./datasets/seeds_belt', category, 'selected_samples_per_run.txt')
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    training_indx = []
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]
            index = strip_line.split(' ')
            training_indx = [int(item) for item in index]
    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
    selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
    selected_train_tot_types = [train_tot_types[k] for k in training_indx]
    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types),\
              (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)  
    