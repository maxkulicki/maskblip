from evaluate_miou import evaluate_mIoU
import wandb

def main():
    wandb.init(project='maskblip')
    score = evaluate_mIoU(device='cuda', batch_size=1, dataset='pascal_context', wandb.config)
    wandb.log({'score': score})

sweep_configuration = {
    'method': 'grid',
    'metric':
    {
        'goal': 'maximize',
        'name': 'score'
        },
    'parameters':
    {
        'kmeans_range': {'values': [3, 4, 5, 6]},
        'pos_emb_dim': {'values': [256, 512, 768, 1024]},
        'smoothness_weight': {'values': [1.0, 2.0, 3.0, 4.0, 5.0]},
        'smoothness_theta': {'values': [0.5, 1.0, 1.5, 2.0]},
        'nr_of_scales': {'values': [2, 3, 4, 5]},
        'scale_step': {'values': [32, 64, 128]},
        'use_nucleus': {'values': [False]},
        'repetition_penalty': {'values': [1.0, 2.0, 3.0]},
        'num_beams': {'values': [1, 3, 5]},                             # beam search (caption generation)
        'top_p': {'values': [0.8, 0.9, 0.95]},                          # nucleus     (caption generation)
        'local_global': {'values': ['local', 'global', 'concat']},
        'background': {'values': [True, False]},
     }
}

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_configuration, project='maskblip')
    wandb.agent(sweep_id, function=main)
