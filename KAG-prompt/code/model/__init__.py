from .openllama import OpenLLAMAPEFTModel
# from .openllama_CLIP import OpenLLAMAPEFTModel_CLIP
from .ImageBind import models

def load_model(args):
    # DeepSpeed is only required for training. Import it lazily so the
    # authors' standalone inference scripts do not require the training stack.
    from .agent import DeepSpeedAgent

    agent_name = args['models'][args['model']]['agent_name']
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    agent = globals()[agent_name](model, args)
    return agent
