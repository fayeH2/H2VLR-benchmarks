import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes

from .belt import load_belt,belt_classes
from .mpdd import load_mpdd, mpdd_classes
from .btad import load_btad, btad_classes
from .busi import load_busi, busi_classes
from .liver import load_liver, liver_classes
from .brain import load_brain, brain_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
    'belt':load_belt,
    'mpdd': load_mpdd,
    'btad': load_btad,
    'busi': load_busi,
    'liver': load_liver,
    'brain': load_brain,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
    'belt':belt_classes,
    'mpdd': mpdd_classes,
    'btad': btad_classes,
    'busi': busi_classes,
    'liver': liver_classes,
    'brain': brain_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot']
    )

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    # debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    # # logger.info(debug_str)

    return data_loader, dataset_inst