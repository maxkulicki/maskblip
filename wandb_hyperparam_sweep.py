# Import the W&B Python Library and log into W&B
import wandb
from maskblip_diff import MaskBLIP
from maskpblip_evaluate_miou import get_ade20k_label_mapping, evaluate_captioning_model
import torch
from lavis.models import load_model_and_preprocess
wandb.login()

# 1: Define objective/training function
def objective(config, blip_model, vis_processors, device):
    model = MaskBLIP(blip_model, device, n_clusters=config.n_clusters, n_iter=config.n_iter, compactness=config.compactness,
                     merging_threshold=config.merging_threshold)
    n_samples = 3
    _, recall, precision, cluster_number = evaluate_captioning_model(model, n_samples, device, vis_processors)

    return recall, precision, cluster_number

def main():
    wandb.init(project='maskBLIP')

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                         device=device)

    recall, precision, cluster_number = objective(wandb.config, blip_model, vis_processors, device)

    wandb.log({
        'recall': recall,
        'precision': precision,
        'cluster_number': cluster_number
    })

# 2: Define the search space
sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'recall'},
    'parameters':
    {
        'n_iter': {'max': 12, 'min': 3},
        'merging_threshold': {'max': 1.0, 'min': 0.99},
        'n_clusters': {'values': [9, 16, 25]},
        'compactness': {'max': 0.1, 'min': 0.005}
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='maskBLIP')
wandb.agent(sweep_id, function=main, count=5)