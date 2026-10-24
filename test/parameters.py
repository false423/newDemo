"""
RDAF-v2 参数配置与源域选择

用法:
    from parameters import GetParameters, ChooseSource
    args = GetParameters()
    source_names = ChooseSource(args.source)
"""

import argparse
import torch


def GetParameters():
    parser = argparse.ArgumentParser(description="RDAF-v2: Reverse Diffusion Alignment for Heterogeneous Multi-source HSI")

    # Dataset options
    parser.add_argument('--target', type=str, default='houston2013', help="Target dataset name")
    parser.add_argument('--source', type=int, default=0, help="Source combination index (see ChooseSource)")
    parser.add_argument('--use_dummy', action='store_true', help="Use dummy data for smoke test")
    parser.add_argument('--need_pca', action='store_true', help="Apply PCA to reduce bands")
    parser.add_argument('--no_normalize', action='store_true', help="Disable spectral normalization")

    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--patch_size', type=int, default=16, help="Spatial patch size")
    group_dataset.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    group_dataset.add_argument('--train_per_class', type=int, default=15, help="Target domain training samples per class")
    group_dataset.add_argument('--val_per_class', type=int, default=15, help="Validation samples per class")

    # ★ 统一的目标域采样比例（默认 1%），供 CDAN / OUR_v3_proto 使用
    group_dataset.add_argument('--ratio', type=float, default=0.01,
                               help="Target domain training sample ratio (default: 1%%)")

    group_dataset.add_argument('--source_expand_ratio', type=float, default=0.02,
                               help="Source expand_set sampling ratio (default 2%%)")
    group_dataset.add_argument('--num_workers', type=int, default=0)
    group_dataset.add_argument('--num_threads', type=int, default=4)

    # Model architecture
    group_model = parser.add_argument_group('Model')
    group_model.add_argument('--hidden_channels', type=int, default=64)
    group_model.add_argument('--feature_dim', type=int, default=128)
    group_model.add_argument('--timesteps', type=int, default=300, help="Total diffusion timesteps T")
    group_model.add_argument('--reverse_steps', type=int, default=4, help="Number of reverse denoising steps")
    group_model.add_argument('--time_emb_dim', type=int, default=64)
    group_model.add_argument('--domain_emb_dim', type=int, default=64)
    group_model.add_argument('--dropout', type=float, default=0.1)

    # Training options
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--epochs', type=int, default=2000)
    group_train.add_argument('--lr', type=float, default=1e-4)
    group_train.add_argument('--weight_decay', type=float, default=1e-4)
    group_train.add_argument('--grad_clip', type=float, default=5.0)
    # group_train.add_argument('--grad_accum', type=int, default=8,
    #                          help="Gradient accumulation steps per optimizer update "
    #                               "(larger effective batch for CORAL/MMD alignment stats)")
    # group_train.add_argument('--stat_ema', type=float, default=0.9,
    #                          help="EMA momentum for smoothing adaptive tau and source weights")
    group_train.add_argument('--seed', type=int, default=0)
    group_train.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    group_train.add_argument('--n_run', type=int, default=1, help="Number of repeated runs")

    # Loss weights
    group_loss = parser.add_argument_group('Loss')
    group_loss.add_argument('--alpha_source', type=float, default=0.5,
                            help="tau score mix: source-source vs source-target distance")
    group_loss.add_argument('--target_weight_temperature', type=float, default=1.0)
    group_loss.add_argument('--lambda_source_cls', type=float, default=1.0)
    group_loss.add_argument('--lambda_source_align', type=float, default=0.5)
    group_loss.add_argument('--lambda_source_target', type=float, default=0.2)
    group_loss.add_argument('--lambda_contrast', type=float, default=0.1)
    group_loss.add_argument('--lambda_residual', type=float, default=0.01)
    group_loss.add_argument('--lambda_target_cls', type=float, default=0.2)
    group_loss.add_argument('--lambda_target_residual', type=float, default=0.005)
    group_loss.add_argument('--lambda_class_align', type=float, default=0.0)
    group_loss.add_argument('--same_label_space', action='store_true',
                            help="Only enable if all source class IDs are semantically identical")

    # Evaluation & output
    group_eval = parser.add_argument_group('Evaluation')
    group_eval.add_argument('--eval_interval', type=int, default=50)
    group_eval.add_argument('--target_eval', choices=['knn', 'prototype', 'linear', 'cosine_knn'], default='prototype')
    group_eval.add_argument('--knn_k', type=int, default=1)
    group_eval.add_argument('--save_dir', type=str, default='./checkpoints_rdaf_v2')
    group_eval.add_argument('--save_source_features', action='store_true')
    group_eval.add_argument('--draw_figure', action='store_true', default=False)
    group_eval.add_argument('--draw_tsne', action='store_true', default=False,
                            help="Generate t-SNE visualization after testing")
    group_eval.add_argument('--draw_map', action='store_true', default=False,
                            help="Generate classification map after testing")
    group_eval.add_argument('--info', type=str, default='-', help="Experiment description tag")

    args = parser.parse_args()
    return args


def ChooseSource(index):
    """
    Source domain combination presets.

    Index 0: 4-source full (default for paper experiments)
    Index 1-4: single source ablation
    Index 5-10: dual source ablation
    Index 11-14: triple source ablation
    """
    # === 4-source full ===
    if index == 0:
        return ['PaviaU', 'Chikusei', 'IndianPines', 'WHU-Hi-HanChuan']

    # === Single source ablation ===
    if index == 1:
        return ['PaviaU']
    if index == 2:
        return ['Chikusei']
    if index == 3:
        return ['IndianPines']
    if index == 4:
        return ['WHU-Hi-HanChuan']

    # === Dual source ablation ===
    if index == 5:
        return ['PaviaU', 'Chikusei']
    if index == 6:
        return ['PaviaU', 'IndianPines']
    if index == 7:
        return ['PaviaU', 'WHU-Hi-HanChuan']
    if index == 8:
        return ['Chikusei', 'IndianPines']
    if index == 9:
        return ['Chikusei', 'WHU-Hi-HanChuan']
    if index == 10:
        return ['IndianPines', 'WHU-Hi-HanChuan']

    # === Triple source ablation ===
    if index == 11:
        return ['PaviaU', 'Chikusei', 'IndianPines']
    if index == 12:
        return ['PaviaU', 'Chikusei', 'WHU-Hi-HanChuan']
    if index == 13:
        return ['PaviaU', 'IndianPines', 'WHU-Hi-HanChuan']
    if index == 14:
        return ['Chikusei', 'IndianPines', 'WHU-Hi-HanChuan']

    raise ValueError(f"Unknown source index: {index}. Valid range: 0-14")