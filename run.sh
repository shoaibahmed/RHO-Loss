#!/bin/bash

python3 run_irreducible.py datamodule.data_dir=/mnt/sas/Datasets/clothing/ +experiment=c1m_resnet18_irred.yaml
python3 run.py datamodule.data_dir=/mnt/sas/Datasets/clothing/ +experiment=c1m_resnet50_main.yaml irreducible_loss_generator=irreducible_loss_model irreducible_loss_generator.checkpoint_path="/mnt/sas/Repositories/RHO-Loss/logs/runs/2022-07-07/22-23-58/checkpoints/epoch_004.ckpt" selection_method=reducible_loss_selection

# python3 run.py datamodule.data_dir=/mnt/sas/Datasets/clothing/ +experiment=c1m_resnet50_main.yaml irreducible_loss_generator=precomputed_loss +precomputed_loss.f="/mnt/sas/Repositories/RHO-Loss/logs/runs/2022-07-07/22-23-58/checkpoints/irred_losses_and_checks.pt" selection_method=reducible_loss_selection