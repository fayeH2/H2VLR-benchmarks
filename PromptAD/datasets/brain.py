import glob
import os
import random

brain_classes = ['brain']

BRAIN_DIR = '/path/to/PromptAD/datasets/anomaly_detection/BrainAD/brain'


def load_brain(category, k_shot):
    """
    加载Brain数据集
    
    Args:
        category: 类别名称 ('brain')
        k_shot: Few-shot样本数 (1, 2, 4)
    
    Returns:
        (train_data, test_data) 元组
    """
    def load_phase(root_path, gt_path):
        """
        加载单个phase（train或test）的数据
        
        Args:
            root_path: 图像根路径（包含good和ungood文件夹）
            gt_path: ground truth根路径
        
        Returns:
            img_tot_paths, gt_tot_paths, tot_labels, tot_types
        """
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        # Brain使用'good'表示正常样本
        if os.path.exists(os.path.join(root_path, 'good')):
            # 正常样本
            defect_type = 'good'
            img_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']:
                img_paths.extend(glob.glob(os.path.join(root_path, defect_type, ext)))
            
            img_paths.sort()
            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend([0] * len(img_paths))  # 正常样本没有GT mask
            tot_labels.extend([0] * len(img_paths))
            tot_types.extend(['good'] * len(img_paths))
        
        # 加载异常样本（ungood）
        defect_type = 'ungood'
        if os.path.exists(os.path.join(root_path, defect_type)):
            img_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']:
                img_paths.extend(glob.glob(os.path.join(root_path, defect_type, ext)))
            
            img_paths.sort()
            
            # 获取对应的ground truth mask
            gt_paths = []
            if os.path.exists(gt_path) and os.path.exists(os.path.join(gt_path, defect_type)):
                for img_path in img_paths:
                    img_name = os.path.basename(img_path)
                    # brain的mask文件名与图像文件名相同
                    mask_path = os.path.join(gt_path, defect_type, img_name)
                    if os.path.exists(mask_path):
                        gt_paths.append(mask_path)
                    else:
                        gt_paths.append(0)
            else:
                gt_paths = [0] * len(img_paths)
            
            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend(gt_paths)
            tot_labels.extend([1] * len(img_paths))
            tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in brain_classes, f"Unknown category: {category}. Must be one of {brain_classes}"

    # brain数据集路径结构
    test_img_path = os.path.join(BRAIN_DIR, 'test')
    train_img_path = os.path.join(BRAIN_DIR, 'train')
    ground_truth_path = os.path.join(BRAIN_DIR, 'ground_truth')

    # 加载train和test数据
    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    # 从train中筛选出正常样本（good类型）
    good_train_paths = []
    good_train_gt_paths = []
    good_train_labels = []
    good_train_types = []
    
    for path, gt, label, typ in zip(train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types):
        if typ == 'good':  # 只保留正常样本
            good_train_paths.append(path)
            good_train_gt_paths.append(gt)
            good_train_labels.append(label)
            good_train_types.append(typ)
    
    # 按文件名排序以确保一致性
    sorted_pairs = sorted(zip(good_train_paths, good_train_gt_paths, good_train_labels, good_train_types))
    good_train_paths, good_train_gt_paths, good_train_labels, good_train_types = zip(*sorted_pairs)
    good_train_paths = list(good_train_paths)
    good_train_gt_paths = list(good_train_gt_paths)
    good_train_labels = list(good_train_labels)
    good_train_types = list(good_train_types)

    # 读取种子文件（在datasets目录下）
    seed_file = os.path.join('./datasets/seeds_BrainAD', 'brain', 'selected_samples_per_run.txt')
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Seed file not found: {seed_file}. Please run generate_brain_seeds.py first.")
    
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    training_indx = []
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]  # 移除换行符
            index = strip_line.split(' ')
            # 种子文件中的索引需要转换为int
            training_indx = [int(item) for item in index if item.strip()]

    # 检查索引是否在有效范围内
    num_good_samples = len(good_train_paths)
    for idx in training_indx:
        if idx >= num_good_samples:
            raise ValueError(f"Seed index {idx} is out of range [0, {num_good_samples-1}] for category {category}")

    # 根据种子文件选择训练样本
    selected_train_img_tot_paths = [good_train_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [good_train_gt_paths[k] for k in training_indx]
    selected_train_tot_labels = [good_train_labels[k] for k in training_indx]
    selected_train_tot_types = [good_train_types[k] for k in training_indx]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)

