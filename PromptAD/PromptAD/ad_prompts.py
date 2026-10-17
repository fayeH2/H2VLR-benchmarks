

class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut",
    "01": "machined metal ring",
    "02": "wood surface",
    "03": "rubber sealing ring cross-section",
}


state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

#
class_state_abnormal = {
    'bottle': ['{} with large breakage', '{} with small breakage', '{} with contamination'],
    'toothbrush': ['{} with defect', '{} with anomaly'],
    'carpet': ['{} with hole', '{} with color stain', '{} with metal contamination', '{} with thread residue', '{} with thread', '{} with cut'],
    'hazelnut': ['{} with crack', '{} with cut', '{} with hole', '{} with print'],
    'leather': ['{} with color stain', '{} with cut', '{} with fold', '{} with glue', '{} with poke'],
    'cable': ['{} with bent wire', '{} with missing part', '{} with missing wire', '{} with cut', '{} with poke'],
    'capsule': ['{} with crack', '{} with faulty imprint', '{} with poke', '{} with scratch', '{} squeezed with compression'],
    'grid': ['{} with breakage',  '{} with thread residue', '{} with thread', '{} with metal contamination', '{} with glue', '{} with a bent shape'],
    'pill': ['{} with color stain', '{} with contamination', '{} with crack', '{} with faulty imprint', '{} with scratch', '{} with abnormal type'],
    'transistor': ['{} with bent lead', '{} with cut lead', '{} with damage', '{} with misplaced transistor'],
    'metal_nut': ['{} with a bent shape ', '{} with color stain', '{} with a flipped orientation', '{} with scratch'],
    'screw': ['{} with manipulated front',  '{} with scratch neck', '{} with scratch head'],
    'zipper': ['{} with broken teeth', '{} with fabric border', '{} with defect fabric', '{} with broken fabric', '{} with split teeth', '{} with squeezed teeth'],
    'tile': ['{} with crack', '{} with glue strip', '{} with gray stroke', '{} with oil', '{} with rough surface'],
    'wood': ['{} with color stain', '{} with hole', '{} with scratch', '{} with liquid'],

    'candle': ['{} with melded wax', '{} with foreign particals', '{} with extra wax', '{} with chunk of wax missing', '{} with weird candle wick', '{} with damaged corner of packaging', '{} with different colour spot'],
    'capsules': ['{} with scratch', '{} with discolor', '{} with misshape', '{} with leak', '{} with bubble'],
    # 'capsules': [],
    'cashew': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with spot'],
    'chewinggum': ['{} with corner missing', '{} with scratches', '{} with chunk of gum missing', '{} with colour spot', '{} with cracks'],
    'fryum': ['{} with breakage', '{} with scratches', '{} with burnt', '{} with colour spot', '{} with fryum stuck together', '{} with colour spot'],
    'macaroni1': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'macaroni2': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'pcb1': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb2': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb3': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb4': ['{} with scratch', '{} with extra', '{} with missing', '{} with wrong place', '{} with damage', '{} with burnt', '{} with dirt'],
    'pipe_fryum': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with colour spot', '{} with cracks'],
    # "MPDD"
    'bracket_black': ['{} with bent arm', '{} with twisted arm', '{} with dented metal', '{} with missing screw', '{} with incomplete screws', '{} with damaged edge'],
    'bracket_brown': ['{} with bent arm', '{} with twisted arm', '{} with dented metal', '{} with missing screw', '{} with incomplete screws', '{} with damaged edge'],
    'bracket_white': ['{} with scratch', '{} with dent', '{} with surface crack', '{} with coating defect', '{} with stain', '{} with dark spot', '{} with foreign particle'],
    'connector': ['{} with missing knob', '{} with missing metal shell', '{} with incomplete part', '{} with unmatched components', '{} with broken connector'],
    'metal_plate': ['{} with rust', '{} with discoloration', '{} with scratch', '{} with scraped surface', '{} with abrasion', '{} with damaged coating'],
    'tubes': ['{} with dent', '{} with flattened shape', '{} with bent tube', '{} with damaged edge', '{} with stain', '{} with discoloration'],
    # "BTAD"    
    '01': ['{} with distorted circular pattern', '{} with missing groove texture', '{} with blurred surface', '{} with scratch', '{} with dent', '{} with chipped edge'],
    '02': ['{} with crack', '{} with split', '{} with chipped surface', '{} with missing material', '{} with abnormal texture', '{} with discoloration'],
    '03': ['{} with deformed shape', '{} with uneven thickness', '{} with irregular cavity', '{} with internal void', '{} with bubble defect', '{} with edge crack'],
    # "BeltAD"
    'belt': ['{} with crack', '{} with worn surface', '{} with perforation', '{} with missing section', '{} with hole', '{} with scratch', '{} with wrinkle', '{} with discoloration', '{} with contamination', '{} with deformation'],

    # 医学数据集
    # "BUSI"
    'breast': ['{} with hypoechoic mass', '{} with irregular boundary', '{} with heterogeneous echo', '{} with posterior shadowing', '{} with acoustic enhancement'],
    # "LiverAD"
    'liver': ['{} with abnormal attenuation', '{} with focal lesion', '{} with irregular margin', '{} with heterogeneous texture', '{} with abnormal contour', '{} with shape deformation'],
    # "BrainAD"
    'brain': ['{} with abnormal signal intensity', '{} with focal lesion', '{} with irregular boundary', '{} with heterogeneous texture', '{} with structural asymmetry', '{} with shape distortion']

    }
