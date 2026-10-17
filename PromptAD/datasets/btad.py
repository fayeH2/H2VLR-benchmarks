import glob
import os
import random

btad_classes = ['01', '02', '03']

BTAD_DIR = '/path/to/PromptAD/datasets/anomaly_detection/BTAD'


def load_btad(category, k_shot):
    """
    加载BTAD数据集
    
    Args:
        category: 类别名称 ('01', '02', '03')
        k_shot: Few-shot样本数 (1, 2, 4)
    
    Returns:
        (train_data, test_data) 元组
    """
    def load_phase(root_path, gt_path):
        """
        加载单个phase（train或test）的数据
        
        Args:
            root_path: 图像根路径（包含ok和ko文件夹）
            gt_path: ground truth根路径
        
        Returns:
            img_tot_paths, gt_tot_paths, tot_labels, tot_types
        """
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        # BTAD使用'ok'表示正常样本，'ko'表示异常样本
        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'ok':  # 正常样本
                # 获取所有图像文件（支持多种格式）
                img_paths = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']:
                    img_paths.extend(glob.glob(os.path.join(root_path, defect_type, ext)))
                
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))  # 正常样本没有GT mask
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['ok'] * len(img_paths))
            elif defect_type == 'ko':  # 异常样本
                # 获取所有图像文件
                img_paths = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']:
                    img_paths.extend(glob.glob(os.path.join(root_path, defect_type, ext)))
                
                img_paths.sort()
                
                # 获取对应的ground truth mask
                # BTAD的ground truth在ground_truth/ko/目录下，文件名与图像文件名相同但扩展名是.png
                gt_paths = []
                if os.path.exists(gt_path) and os.path.exists(os.path.join(gt_path, defect_type)):
                    for img_path in img_paths:
                        img_name = os.path.basename(img_path)
                        base_name = os.path.splitext(img_name)[0]
                        # BTAD的mask文件名：图像名.png（如0000.bmp -> 0000.png）
                        valid_extensions = ['.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff']
                        mask_path = None

                        # 遍历后缀查找真实存在的文件
                        for ext in valid_extensions:
                            candidate_path = os.path.join(gt_path, defect_type, base_name + ext)
                            if os.path.exists(candidate_path):
                                mask_path = candidate_path
                                break

                        # 如果没找到，做一个兜底处理（可选）
                        if mask_path is None:
                            # 如果找不到对应的掩膜，可能意味着该样本是正常的，或者路径有问题
                            # 这里可以根据你的逻辑决定是报错还是设为 None
                            # print(f"Warning: No mask found for {base_name}, trying default .png")
                            mask_path = os.path.join(gt_path, defect_type, base_name + '.png')




                        # mask_path = os.path.join(gt_path, defect_type, base_name + '.png')
                        if os.path.exists(mask_path):
                            gt_paths.append(mask_path)
                        else:
                            gt_paths.append(0)
                else:
                    gt_paths = [0] * len(img_paths)
                
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend(['ko'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in btad_classes, f"Unknown category: {category}. Must be one of {btad_classes}"

    # BTAD数据集路径结构
    test_img_path = os.path.join(BTAD_DIR, category, 'test')
    train_img_path = os.path.join(BTAD_DIR, category, 'train')
    ground_truth_path = os.path.join(BTAD_DIR, category, 'ground_truth')

    # 加载train和test数据
    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    # 从train中筛选出正常样本（ok类型）
    good_train_paths = []
    good_train_gt_paths = []
    good_train_labels = []
    good_train_types = []
    
    for path, gt, label, typ in zip(train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types):
        if typ == 'ok':  # 只保留正常样本
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
    seed_file = os.path.join('./datasets/seeds_btad', category, 'selected_samples_per_run.txt')
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"Seed file not found: {seed_file}. Please run generate_btad_seeds.py first.")
    
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    training_indx = []
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]  # 移除换行符
            index = strip_line.split(' ')
            # 种子文件中的索引是4位数字字符串，需要转换为int
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

