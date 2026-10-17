import os

beltad_classes = ['belt']
btad_classes = ['01', '02', '03']
mpdd_classes = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector',
                'metal_plate', 'tubes']


def _load_industrial(dataset, category, k_shot):
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from _shared.industrial_data import (
        collect_samples as _collect_samples,
        dataset_dir_name,
        normal_train_images as _normal_train_images,
        support_manifest as _support_manifest,
    )

    ind_root = os.environ.get(
        'INDUSTRIAL_DATA_ROOT', '/path/to/datasets/industrial')
    dataset = dataset_dir_name(dataset)
    root = os.path.join(ind_root, dataset)

    candidates = _normal_train_images(root, category)
    test_samples = _collect_samples(root, category)
    test_img_tot_paths = [s[0] for s in test_samples]
    test_gt_tot_paths = [s[1] if s[1] is not None else 0 for s in test_samples]
    test_tot_labels = [s[2] for s in test_samples]
    test_tot_types = []
    for image_path, _mask, _label in test_samples:
        defect_dir = os.path.basename(os.path.dirname(image_path))
        test_tot_types.append(defect_dir)

    train_img_path = os.path.join(root, category, 'train')
    train_gt_tot_paths = [0] * len(candidates)
    train_tot_types = []
    train_dir = os.path.basename(os.path.dirname(candidates[0])) if candidates else 'good'
    train_tot_types = [train_dir] * len(candidates)
    train_tot_labels = [0] * len(candidates)
    train_img_tot_paths = [str(p) for p in candidates]

    manifest = _support_manifest(root, dataset, category, k_shot, rounds=1, seed=668)[0]
    chosen_stems = [os.path.splitext(os.path.basename(rel))[0] for rel in manifest]
    training_indx = [
        i for i, p in enumerate(train_img_tot_paths)
        if os.path.splitext(os.path.basename(p))[0] in chosen_stems
    ]

    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
    selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
    selected_train_tot_types = [train_tot_types[k] for k in training_indx]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths,
            selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)


def load_btad(category, k_shot):
    assert category in btad_classes
    return _load_industrial('btad', category, k_shot)


def load_mpdd(category, k_shot):
    assert category in mpdd_classes
    return _load_industrial('mpdd', category, k_shot)


def load_beltad(category, k_shot):
    assert category in beltad_classes
    return _load_industrial('beltad', category, k_shot)