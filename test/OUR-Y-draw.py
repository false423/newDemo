import argparse
import csv
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import GetParameters, ChooseSource

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_device(value) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))


def stable_name_seed(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 100000


class InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def load_hsi_dataset(name: str, need_pca: bool = False):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from dataset import get_dataset
    try:
        out = get_dataset(name, need_pca)
    except TypeError:
        out = get_dataset(name)
    data, gt = out[0], out[2]
    return data.astype(np.float32), gt.astype(np.int64)


def normalize_hsi(data: np.ndarray):
    flat = data.reshape(-1, data.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return ((data - mean) / (std + 1e-6)).astype(np.float32)


def build_label_mapping(gt: np.ndarray):
    old_labels = sorted(int(v) for v in np.unique(gt) if int(v) != 0)
    mapping = {old: i for i, old in enumerate(old_labels)}
    return mapping, len(old_labels)


def get_coords_by_class(gt: np.ndarray, label_to_index: Dict[int, int]):
    by_class = {idx: [] for idx in label_to_index.values()}
    h, w = gt.shape
    for r in range(h):
        for c in range(w):
            old = int(gt[r, c])
            if old in label_to_index:
                by_class[label_to_index[old]].append((r, c))
    return by_class


def pad_data(data: np.ndarray, patch_size: int):
    before = patch_size // 2
    after = patch_size - 1 - before
    return np.pad(data, ((before, after), (before, after), (0, 0)), mode="reflect")


class LazyHSIPatchDataset(Dataset):
    def __init__(self, padded_data, coords, labels, patch_size):
        self.padded_data = padded_data
        self.coords = list(coords)
        self.labels = torch.LongTensor(labels)
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        ps = self.patch_size
        patch = self.padded_data[r:r + ps, c:c + ps].copy()
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        return patch, self.labels[idx]


def make_loader(ds, batch_size, shuffle, seed, drop_last=False):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=0, generator=gen)


def load_source_data(name: str, args, seed: int):
    data, gt = load_hsi_dataset(name, getattr(args, "need_pca", False))
    if not getattr(args, "no_normalize", False):
        data = normalize_hsi(data)
    label_to_index, n_classes = build_label_mapping(gt)
    by_class = get_coords_by_class(gt, label_to_index)
    rng = np.random.RandomState(seed + stable_name_seed(name))
    all_coords, all_labels = [], []
    for cls, coords in by_class.items():
        coords = list(coords)
        if not coords:
            continue
        order = rng.permutation(len(coords))
        coords = [coords[i] for i in order]
        all_coords.extend(coords)
        all_labels.extend([cls] * len(coords))
    ds = LazyHSIPatchDataset(pad_data(data, args.patch_size), all_coords, all_labels,
                             args.patch_size)
    loader = make_loader(ds, args.batch_size, True,
                         seed + 1000 + stable_name_seed(name),
                         drop_last=(len(ds) >= args.batch_size))
    return loader, data.shape[2], n_classes


@dataclass
class TargetDataBundle:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    channels: int
    n_classes: int
    train_counts: List[int]


def load_target_data_fixed_shot(name: str, args, seed: int, shots: int = 15):
    data, gt = load_hsi_dataset(name, getattr(args, "need_pca", False))
    if not getattr(args, "no_normalize", False):
        data = normalize_hsi(data)
    label_to_index, n_classes = build_label_mapping(gt)
    by_class = get_coords_by_class(gt, label_to_index)
    val_per_class = int(getattr(args, "val_per_class", shots))
    rng = np.random.RandomState(seed + 900000)
    tr_c, tr_y, va_c, va_y, te_c, te_y = [], [], [], [], [], []
    train_counts = []
    for cls, coords in by_class.items():
        coords = list(coords)
        n = len(coords)
        if n <= shots:
            raise RuntimeError(
                f"{name} class {cls} has only {n} pixels; cannot draw {shots} train samples."
            )
        order = rng.permutation(n)
        coords = [coords[i] for i in order]
        n_val = min(val_per_class, max(0, n - shots - 1))
        tr = coords[:shots]
        va = coords[shots:shots + n_val]
        te = coords[shots + n_val:]
        train_counts.append(len(tr))
        tr_c.extend(tr); tr_y.extend([cls] * len(tr))
        va_c.extend(va); va_y.extend([cls] * len(va))
        te_c.extend(te); te_y.extend([cls] * len(te))
    padded = pad_data(data, args.patch_size)
    train_ds = LazyHSIPatchDataset(padded, tr_c, tr_y, args.patch_size)
    val_ds = LazyHSIPatchDataset(padded, va_c, va_y, args.patch_size)
    test_ds = LazyHSIPatchDataset(padded, te_c, te_y, args.patch_size)
    return TargetDataBundle(
        train_loader=make_loader(train_ds, args.batch_size, True, seed + 910000),
        train_eval_loader=make_loader(train_ds, args.batch_size, False, seed + 920000),
        val_loader=make_loader(val_ds, args.batch_size, False, seed + 930000),
        test_loader=make_loader(test_ds, args.batch_size, False, seed + 940000),
        channels=data.shape[2], n_classes=n_classes, train_counts=train_counts,
    )


def _group_count(channels: int):
    for g in (16, 8, 4, 2):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualGNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm2(self.conv2(self.conv1(x))))


class SpectralAdapter(nn.Module):
    """Purely spectral target/source entrance: 1x1 convolutions only."""
    def __init__(self, in_channels: int, adapter_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.adapter_channels = int(adapter_channels)
        self.conv_in = nn.Conv2d(in_channels, adapter_channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(adapter_channels), adapter_channels)
        self.conv_mix = nn.Conv2d(adapter_channels, adapter_channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(adapter_channels), adapter_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv_in(x)))
        x = self.act(self.norm2(self.conv_mix(x)))
        return x


class SharedSemanticEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_dim, dropout):
        super().__init__()
        h = hidden_channels
        self.net = nn.Sequential(
            ConvGNAct(in_channels, h, 3, 1, 1), ResidualGNBlock(h),
            ConvGNAct(h, h * 2, 3, 2, 1), ResidualGNBlock(h * 2),
            ConvGNAct(h * 2, h * 4, 3, 2, 1), ResidualGNBlock(h * 4),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(h * 4, feature_dim), nn.LayerNorm(feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class SharedSemanticDictionary(nn.Module):
    """
    Shared basis B={b_k}. Different samples use different coefficients alpha.
    Sharing is the representation basis, not the sample distribution.
    """
    def __init__(self, feature_dim, num_atoms, temperature=0.10, min_gate=0.35):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_atoms = int(num_atoms)
        self.temperature = float(temperature)
        self.min_gate = float(min_gate)
        self.atoms = nn.Parameter(F.normalize(torch.randn(num_atoms, feature_dim), dim=1))
        self.dict_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(feature_dim)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def effective_gate(self):
        return self.min_gate + (1.0 - self.min_gate) * torch.sigmoid(self.gate_logit)

    def forward(self, h, enabled=True):
        h = F.normalize(h, dim=1)
        if not enabled:
            return {"feature": h, "alpha": None, "dictionary_feature": None,
                    "gate": torch.tensor(0.0, device=h.device)}
        B = F.normalize(self.atoms, dim=1)
        alpha = F.softmax(h @ B.t() / self.temperature, dim=1)
        d = alpha @ B
        gate = self.effective_gate()
        z = self.fusion_norm(h + gate * self.dict_proj(d))
        z = F.normalize(z, dim=1)
        return {"feature": z, "alpha": alpha, "dictionary_feature": d, "gate": gate}


def dictionary_regularization(dictionary, alpha_all, target_sample_entropy):
    B = F.normalize(dictionary.atoms, dim=1)
    gram = B @ B.t()
    eye = torch.eye(dictionary.num_atoms, device=B.device)
    off = gram * (1.0 - eye)
    loss_div = off.pow(2).sum() / max(dictionary.num_atoms * (dictionary.num_atoms - 1), 1)
    eps = 1e-8
    sample_h = -(alpha_all * torch.log(alpha_all + eps)).sum(1) / math.log(dictionary.num_atoms)
    loss_sel = (sample_h.mean() - target_sample_entropy).pow(2)
    usage = alpha_all.mean(0)
    loss_bal = (usage * torch.log(usage * dictionary.num_atoms + eps)).sum()
    usage_h = -(usage * torch.log(usage + eps)).sum() / math.log(dictionary.num_atoms)
    return {"diversity": loss_div, "selectivity": loss_sel, "balance": loss_bal,
            "sample_entropy": sample_h.mean(), "usage_entropy": usage_h,
            "max_usage": usage.max()}


class SemanticManifoldModel(nn.Module):
    def __init__(self, source_channels, source_classes, target_channels,
                 adapter_channels, hidden_channels, feature_dim, num_atoms,
                 dictionary_temperature, dropout):
        super().__init__()
        self.source_names = list(source_channels.keys())
        self.source_channels = dict(source_channels)
        self.feature_dim = feature_dim
        self.source_adapters = nn.ModuleDict({
            n: SpectralAdapter(source_channels[n], adapter_channels) for n in self.source_names
        })
        self.target_adapter = SpectralAdapter(target_channels, adapter_channels)
        self.shared_encoder = SharedSemanticEncoder(adapter_channels, hidden_channels,
                                                    feature_dim, dropout)
        self.semantic_dictionary = SharedSemanticDictionary(feature_dim, num_atoms,
                                                             dictionary_temperature)
        self.source_heads = nn.ModuleDict({
            n: nn.Linear(feature_dim, source_classes[n]) for n in self.source_names
        })

    def _encode(self, adapted, use_dictionary):
        h = self.shared_encoder(adapted)
        sem = self.semantic_dictionary(h, enabled=use_dictionary)
        return {"base_feature": F.normalize(h, dim=1), "feature": sem["feature"],
                "alpha": sem["alpha"], "dictionary_feature": sem["dictionary_feature"],
                "dictionary_gate": sem["gate"]}

    def encode_source(self, x, name, use_dictionary=True):
        return self._encode(self.source_adapters[name](x), use_dictionary)

    def encode_target(self, x, use_dictionary=True):
        return self._encode(self.target_adapter(x), use_dictionary)


def supervised_contrastive_loss(features, labels, temperature=0.10):
    if features.size(0) <= 1:
        return torch.tensor(0.0, device=features.device)
    z = F.normalize(features, dim=1)
    sim = z @ z.t() / temperature
    labels_col = labels.view(-1, 1)
    pos = (labels_col == labels_col.t()).float()
    eye = torch.eye(features.size(0), device=features.device)
    pos = pos * (1.0 - eye)
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim) * (1.0 - eye)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    loss = -(pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return loss[valid].mean()


@torch.no_grad()
def resize_spectral_conv_weight(weight, target_in_channels):
    out_channels, source_in, _, _ = weight.shape
    if source_in == target_in_channels:
        return weight.clone()
    w = weight[:, :, 0, 0].unsqueeze(0)
    wr = F.interpolate(w, size=target_in_channels, mode="linear", align_corners=True).squeeze(0)
    old_norm = weight[:, :, 0, 0].norm(dim=1, keepdim=True)
    new_norm = wr.norm(dim=1, keepdim=True).clamp_min(1e-8)
    wr = wr * old_norm / new_norm
    return wr[:, :, None, None]


@torch.no_grad()
def initialize_target_adapter_from_sources(model):
    target = model.target_adapter
    ct = target.in_channels
    names = list(model.source_adapters.keys())
    raw = []
    for n in names:
        cs = model.source_adapters[n].in_channels
        raw.append(math.exp(-abs(math.log(max(cs, 1) / max(ct, 1)))))
    s = sum(raw) + 1e-12
    mixing = [v / s for v in raw]
    first = None
    for a, n in zip(mixing, names):
        w = resize_spectral_conv_weight(model.source_adapters[n].conv_in.weight.data, ct)
        first = a * w if first is None else first + a * w
    target.conv_in.weight.data.copy_(first)

    def avg(getter):
        value = None
        for a, n in zip(mixing, names):
            t = getter(model.source_adapters[n]).detach()
            value = a * t if value is None else value + a * t
        return value

    target.norm1.weight.data.copy_(avg(lambda x: x.norm1.weight.data))
    target.norm1.bias.data.copy_(avg(lambda x: x.norm1.bias.data))
    target.conv_mix.weight.data.copy_(avg(lambda x: x.conv_mix.weight.data))
    target.norm2.weight.data.copy_(avg(lambda x: x.norm2.weight.data))
    target.norm2.bias.data.copy_(avg(lambda x: x.norm2.bias.data))
    return {n: mixing[i] for i, n in enumerate(names)}


def train_source_stage(model, source_loaders, args, cfg, device):
    print("\n" + "=" * 96)
    print("STAGE I - MULTI-SOURCE TRANSFERABLE SEMANTIC MANIFOLD LEARNING")
    print("Target data are NOT used.")
    print("=" * 96)

    names = list(source_loaders.keys())
    iters = {n: InfiniteLoader(source_loaders[n]) for n in names}
    use_dictionary = cfg.ablation != "no_dictionary"
    use_supcon = cfg.ablation != "no_supcon"

    trainable = (
        list(model.source_adapters.parameters())
        + list(model.shared_encoder.parameters())
        + list(model.semantic_dictionary.parameters())
        + list(model.source_heads.parameters())
    )

    opt = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(cfg.source_steps, 1),
        eta_min=args.lr * 0.01,
    )

    history = {
        "step": [],
        "total": [],
        "cls": [],
        "supcon": [],
        "dict_div": [],
        "dict_sel": [],
        "dict_bal": [],
        "gate": [],
        "sample_entropy": [],
        "usage_entropy": [],
        "max_atom_usage": [],
    }

    for step in range(1, cfg.source_steps + 1):
        model.train()
        outputs, labels = {}, {}

        for n in names:
            x, y = next(iters[n])
            x, y = x.to(device), y.to(device)
            outputs[n] = model.encode_source(x, n, use_dictionary)
            labels[n] = y

        loss_cls = torch.stack([
            F.cross_entropy(
                model.source_heads[n](outputs[n]["feature"]),
                labels[n],
            )
            for n in names
        ]).mean()

        if use_supcon:
            loss_sup = torch.stack([
                supervised_contrastive_loss(
                    outputs[n]["feature"],
                    labels[n],
                    cfg.contrast_temperature,
                )
                for n in names
            ]).mean()
        else:
            loss_sup = torch.tensor(0.0, device=device)

        if use_dictionary:
            alpha_all = torch.cat([
                outputs[n]["alpha"]
                for n in names
            ], dim=0)

            reg = dictionary_regularization(
                model.semantic_dictionary,
                alpha_all,
                cfg.dictionary_target_entropy,
            )

            ldiv = reg["diversity"]
            lsel = reg["selectivity"]
            lbal = reg["balance"]
            gate = float(
                model.semantic_dictionary.effective_gate().detach().item()
            )
        else:
            zero = torch.tensor(0.0, device=device)
            reg = {
                "sample_entropy": zero,
                "usage_entropy": zero,
                "max_usage": zero,
            }
            ldiv = lsel = lbal = zero
            gate = 0.0

        total = (
            cfg.lambda_source_cls * loss_cls
            + cfg.lambda_supcon * loss_sup
            + cfg.lambda_dict_diversity * ldiv
            + cfg.lambda_dict_selectivity * lsel
            + cfg.lambda_dict_balance * lbal
        )

        opt.zero_grad()
        total.backward()

        if getattr(args, "grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable,
                args.grad_clip,
            )

        opt.step()
        sched.step()

        history["step"].append(step)
        history["total"].append(float(total.item()))
        history["cls"].append(float(loss_cls.item()))
        history["supcon"].append(float(loss_sup.item()))
        history["dict_div"].append(float(ldiv.item()))
        history["dict_sel"].append(float(lsel.item()))
        history["dict_bal"].append(float(lbal.item()))
        history["gate"].append(gate)
        history["sample_entropy"].append(float(reg["sample_entropy"]))
        history["usage_entropy"].append(float(reg["usage_entropy"]))
        history["max_atom_usage"].append(float(reg["max_usage"]))

        if step == 1 or step % cfg.print_interval == 0 or step == cfg.source_steps:
            print(
                f"[Source] {step:5d}/{cfg.source_steps} | "
                f"total={total.item():.4f} | "
                f"cls={loss_cls.item():.4f} | "
                f"supcon={loss_sup.item():.4f} | "
                f"dict_div={ldiv.item():.4f} | "
                f"dict_sel={lsel.item():.4f} | "
                f"dict_bal={lbal.item():.4f}"
            )
            print(
                f"         gate={gate:.4f} | "
                f"sample_H={float(reg['sample_entropy']):.4f} | "
                f"usage_H={float(reg['usage_entropy']):.4f} | "
                f"max_atom={float(reg['max_usage']):.4f}"
            )

    return history


@torch.no_grad()
def collect_small_dataset(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.cpu()); ys.append(y.cpu())
    return torch.cat(xs, 0), torch.cat(ys, 0)


def sample_fewshot_episode(all_x, all_y, n_classes, support_num, query_num, generator):
    s_idx, q_idx = [], []
    for cls in range(n_classes):
        idx = torch.where(all_y == cls)[0]
        need = support_num + query_num
        if len(idx) < need:
            raise RuntimeError(f"Class {cls}: {len(idx)} samples, episode requires {need}.")
        perm = torch.randperm(len(idx), generator=generator)
        chosen = idx[perm[:need]]
        s_idx.append(chosen[:support_num]); q_idx.append(chosen[support_num:])
    s_idx, q_idx = torch.cat(s_idx), torch.cat(q_idx)
    return all_x[s_idx], all_y[s_idx], all_x[q_idx], all_y[q_idx]


def freeze_for_target_tuning(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.target_adapter.parameters():
        p.requires_grad = True


def snapshot_module_parameters(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def module_anchor_loss(module, anchor):
    return torch.stack([(p - anchor[n].to(p.device)).pow(2).mean()
                        for n, p in module.named_parameters()]).mean()


def cosine_prototype_logits(query_feature, support_feature, support_labels,
                            n_classes, temperature):
    protos = []
    for cls in range(n_classes):
        mask = support_labels == cls
        if mask.sum() == 0:
            raise RuntimeError(f"No support sample for target class {cls}.")
        protos.append(support_feature[mask].mean(0))
    protos = F.normalize(torch.stack(protos), dim=1)
    query_feature = F.normalize(query_feature, dim=1)
    return query_feature @ protos.t() / temperature


def adapter_relative_change(module, anchor):
    """
    Relative L2 movement of the target adapter from its source-informed initialization.

        ||theta - theta_0||_2 / (||theta_0||_2 + eps)

    This is much easier to interpret than the raw mean-squared anchor loss.
    """
    diff_sq = torch.tensor(0.0, device=next(module.parameters()).device)
    base_sq = torch.tensor(0.0, device=next(module.parameters()).device)

    for name, p in module.named_parameters():
        p0 = anchor[name].to(p.device)
        diff_sq = diff_sq + (p - p0).pow(2).sum()
        base_sq = base_sq + p0.pow(2).sum()

    return float(
        torch.sqrt(diff_sq).div(torch.sqrt(base_sq) + 1e-12).detach().item()
    )


@torch.no_grad()
def leave_one_out_1nn_accuracy(features, labels):
    """
    Honest training-set geometry diagnostic.

    Self-neighbor is explicitly removed, so this is NOT the trivial 100% training
    accuracy produced by fitting 1-NN and querying each sample with itself.
    """
    z = F.normalize(features, dim=1)
    sim = z @ z.t()
    sim.fill_diagonal_(-1e9)
    nn_index = sim.argmax(dim=1)
    pred = labels[nn_index]
    return float((pred == labels).float().mean().item())


@torch.no_grad()
def target_overfit_diagnostic(model, target, cfg, device, anchor):
    use_dictionary = cfg.ablation != "no_dictionary"

    train_f, train_y = extract_target_features(
        model,
        target.train_eval_loader,
        device,
        use_dictionary,
    )

    loo_acc = leave_one_out_1nn_accuracy(
        train_f,
        train_y,
    )

    val = evaluate_target_streaming(
        model,
        target.train_eval_loader,
        target.val_loader,
        target.n_classes,
        device,
        use_dictionary,
        cfg.classifier,
    )

    delta = adapter_relative_change(
        model.target_adapter,
        anchor,
    )

    return {
        "loo_train_acc": loo_acc,
        "val_oa": float(val["oa"]),
        "val_aa": float(val["aa"]),
        "val_kappa": float(val["kappa"]),
        "generalization_gap": loo_acc - float(val["oa"]),
        "adapter_relative_change": delta,
    }


def _append_target_history(history, step, total_loss, proto_loss, anchor_loss,
                           episode_acc, diag):
    history["step"].append(int(step))
    history["total_loss"].append(float(total_loss))
    history["proto_loss"].append(float(proto_loss))
    history["anchor_loss"].append(float(anchor_loss))
    history["episode_acc"].append(float(episode_acc))
    history["loo_train_acc"].append(float(diag["loo_train_acc"]))
    history["val_oa"].append(float(diag["val_oa"]))
    history["val_aa"].append(float(diag["val_aa"]))
    history["val_kappa"].append(float(diag["val_kappa"]))
    history["generalization_gap"].append(float(diag["generalization_gap"]))
    history["adapter_relative_change"].append(float(diag["adapter_relative_change"]))


def train_target_adapter_stage(model, target, args, cfg, device):
    print("\n" + "=" * 96)
    print("STAGE II - 15-SHOT TARGET SPECTRAL ADAPTER TUNING")
    print("Target labels are used, but ONLY the shallow Target Adapter is trainable.")
    print("Shared Encoder + Semantic Dictionary are frozen.")
    print("Validation is MONITORED only for overfitting visualization; it never updates the model")
    print("and is NOT used for early stopping or checkpoint selection.")
    print("=" * 96)

    use_dictionary = cfg.ablation != "no_dictionary"

    if cfg.ablation == "random_target_init":
        init_weights = {n: 0.0 for n in model.source_names}
        print("[Target Init] RANDOM (ablation)")
    else:
        init_weights = initialize_target_adapter_from_sources(model)
        print(
            "[Target Init] source-informed: "
            + ", ".join(f"{n}:{w:.3f}" for n, w in init_weights.items())
        )

    # Anchor is defined AFTER target initialization.
    anchor = snapshot_module_parameters(model.target_adapter)

    history = {
        "step": [],
        "total_loss": [],
        "proto_loss": [],
        "anchor_loss": [],
        "episode_acc": [],
        "loo_train_acc": [],
        "val_oa": [],
        "val_aa": [],
        "val_kappa": [],
        "generalization_gap": [],
        "adapter_relative_change": [],
    }

    # Step-0 diagnostic: source-informed target entrance BEFORE target label tuning.
    # It is especially important for quantifying how much the source manifold alone transfers.
    diag0 = target_overfit_diagnostic(
        model,
        target,
        cfg,
        device,
        anchor,
    )

    _append_target_history(
        history,
        step=0,
        total_loss=float("nan"),
        proto_loss=float("nan"),
        anchor_loss=0.0,
        episode_acc=float("nan"),
        diag=diag0,
    )

    print(
        f"[TargetAdapter] step=0 | LOO-train={diag0['loo_train_acc']*100:.2f}% | "
        f"Val-OA={diag0['val_oa']*100:.2f}% | "
        f"Gap={diag0['generalization_gap']*100:.2f}pp | "
        f"Adapter-delta={diag0['adapter_relative_change']*100:.3f}%"
    )

    if cfg.ablation == "no_target_tune":
        print("[Target Tune] skipped by ablation.")
        return {
            "init_weights": init_weights,
            "final_proto_loss": float("nan"),
            "final_anchor_loss": float("nan"),
            "history": history,
        }

    freeze_for_target_tuning(model)

    # Keep dropout and all frozen modules deterministic.
    model.eval()
    model.target_adapter.train()

    trainable = [
        p
        for p in model.target_adapter.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        trainable,
        lr=cfg.target_lr,
        weight_decay=args.weight_decay,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(cfg.target_steps, 1),
        eta_min=cfg.target_lr * 0.01,
    )

    all_x, all_y = collect_small_dataset(
        target.train_eval_loader
    )

    expected = cfg.target_shots * target.n_classes
    if len(all_y) != expected:
        raise RuntimeError(
            f"Expected target pool {expected}, got {len(all_y)}"
        )

    gen = torch.Generator()
    gen.manual_seed(args.seed + 555555)

    query_num = cfg.target_shots - cfg.support_num
    if query_num <= 0:
        raise ValueError("target_shots must be > support_num")

    stats = {}

    for step in range(1, cfg.target_steps + 1):
        model.eval()
        model.target_adapter.train()

        sx, sy, qx, qy = sample_fewshot_episode(
            all_x,
            all_y,
            target.n_classes,
            cfg.support_num,
            query_num,
            gen,
        )

        sx = sx.to(device)
        sy = sy.to(device)
        qx = qx.to(device)
        qy = qy.to(device)

        s_out = model.encode_target(
            sx,
            use_dictionary,
        )

        q_out = model.encode_target(
            qx,
            use_dictionary,
        )

        logits = cosine_prototype_logits(
            q_out["feature"],
            s_out["feature"],
            sy,
            target.n_classes,
            cfg.target_proto_temperature,
        )

        loss_proto = F.cross_entropy(
            logits,
            qy,
            label_smoothing=cfg.target_label_smoothing,
        )

        loss_anchor = module_anchor_loss(
            model.target_adapter,
            anchor,
        )

        total = (
            loss_proto
            + cfg.lambda_target_anchor * loss_anchor
        )

        opt.zero_grad()
        total.backward()

        if getattr(args, "grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable,
                args.grad_clip,
            )

        opt.step()
        sched.step()

        with torch.no_grad():
            episode_acc = float(
                (logits.argmax(1) == qy).float().mean().item()
            )

        stats = {
            "final_proto_loss": float(loss_proto.item()),
            "final_anchor_loss": float(loss_anchor.item()),
        }

        need_diagnostic = (
            step == 1
            or step % cfg.visual_interval == 0
            or step == cfg.target_steps
        )

        diag = None
        if need_diagnostic:
            # No gradients and no optimizer decisions depend on this result.
            diag = target_overfit_diagnostic(
                model,
                target,
                cfg,
                device,
                anchor,
            )

            _append_target_history(
                history,
                step=step,
                total_loss=float(total.item()),
                proto_loss=float(loss_proto.item()),
                anchor_loss=float(loss_anchor.item()),
                episode_acc=episode_acc,
                diag=diag,
            )

        if step == 1 or step % cfg.print_interval == 0 or step == cfg.target_steps:
            message = (
                f"[TargetAdapter] {step:5d}/{cfg.target_steps} | "
                f"total={total.item():.4f} | "
                f"proto={loss_proto.item():.4f} | "
                f"anchor={loss_anchor.item():.6f} | "
                f"episode_acc={episode_acc*100:.2f}%"
            )

            if diag is not None:
                message += (
                    f" | LOO-train={diag['loo_train_acc']*100:.2f}%"
                    f" | Val-OA={diag['val_oa']*100:.2f}%"
                    f" | Gap={diag['generalization_gap']*100:.2f}pp"
                    f" | Adapter-delta={diag['adapter_relative_change']*100:.3f}%"
                )

            print(message)

    stats["init_weights"] = init_weights
    stats["history"] = history
    return stats


@torch.no_grad()
def extract_target_features(model, loader, device, use_dictionary):
    model.eval(); feats, labels = [], []
    for x, y in loader:
        out = model.encode_target(x.to(device), use_dictionary)
        feats.append(out["feature"].cpu()); labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


def build_target_prototypes(train_feature, train_y, n_classes):
    protos = []
    for cls in range(n_classes):
        mask = train_y == cls
        if mask.sum() == 0:
            raise RuntimeError(f"No target reference for class {cls}.")
        protos.append(train_feature[mask].mean(0))
    return F.normalize(torch.stack(protos), dim=1)


@torch.no_grad()
def evaluate_target_streaming(model, train_loader, eval_loader, n_classes,
                              device, use_dictionary, classifier):
    train_f, train_y = extract_target_features(model, train_loader, device, use_dictionary)
    if classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        clf.fit(train_f.numpy(), train_y.numpy())
        protos = None
    elif classifier == "proto":
        protos = build_target_prototypes(train_f, train_y, n_classes)
        clf = None
    else:
        raise ValueError(classifier)
    true_parts, pred_parts = [], []
    model.eval()
    for x, y in eval_loader:
        f = model.encode_target(x.to(device), use_dictionary)["feature"].cpu()
        if classifier == "knn":
            pred = clf.predict(f.numpy())
        else:
            pred = torch.argmax(F.normalize(f, dim=1) @ protos.t(), dim=1).numpy()
        true_parts.append(y.numpy()); pred_parts.append(pred)
    y_true = np.concatenate(true_parts); pred = np.concatenate(pred_parts)
    oa = accuracy_score(y_true, pred)
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    each = np.diag(cm) / (cm.sum(axis=1) + 1e-12)
    return {"oa": oa, "aa": float(np.mean(each)),
            "kappa": cohen_kappa_score(y_true, pred), "class_acc": each}


# =============================================================================
# Visualization and overfitting diagnostics
# =============================================================================

def history_dict_to_rows(history):
    if not history or not history.get("step"):
        return []

    keys = list(history.keys())
    n = len(history["step"])
    return [
        {k: history[k][i] for k in keys}
        for i in range(n)
    ]


def save_history_csv(history, path):
    rows = history_dict_to_rows(history)
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finish_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_source_history(history, out_dir):
    if not history or not history.get("step"):
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    step = np.asarray(history["step"])

    plt.figure(figsize=(8.0, 5.2))
    plt.plot(step, history["total"], label="Total")
    plt.plot(step, history["cls"], label="Source CE")
    plt.plot(step, history["supcon"], label="Source SupCon")
    plt.xlabel("Source training step")
    plt.ylabel("Loss")
    plt.title("Source-stage loss curves")
    plt.grid(alpha=0.25)
    plt.legend()
    path = os.path.join(out_dir, "source_loss_curves.png")
    _finish_figure(path)
    saved.append(path)

    plt.figure(figsize=(8.0, 5.2))
    plt.plot(step, history["gate"], label="Dictionary gate")
    plt.plot(step, history["sample_entropy"], label="Sample entropy")
    plt.plot(step, history["usage_entropy"], label="Global usage entropy")
    plt.plot(step, history["max_atom_usage"], label="Max atom usage")
    plt.xlabel("Source training step")
    plt.ylabel("Normalized value")
    plt.ylim(-0.02, 1.02)
    plt.title("Semantic dictionary diagnostics")
    plt.grid(alpha=0.25)
    plt.legend()
    path = os.path.join(out_dir, "dictionary_diagnostics.png")
    _finish_figure(path)
    saved.append(path)

    save_history_csv(
        history,
        os.path.join(out_dir, "source_history.csv"),
    )

    return saved


def plot_target_history(history, out_dir):
    if not history or not history.get("step"):
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    step = np.asarray(history["step"], dtype=float)
    loo = 100.0 * np.asarray(history["loo_train_acc"], dtype=float)
    val = 100.0 * np.asarray(history["val_oa"], dtype=float)
    gap = 100.0 * np.asarray(history["generalization_gap"], dtype=float)
    delta = 100.0 * np.asarray(history["adapter_relative_change"], dtype=float)
    proto = np.asarray(history["proto_loss"], dtype=float)
    episode = 100.0 * np.asarray(history["episode_acc"], dtype=float)

    # 1) Core overfitting chart: honest LOO train geometry vs independent validation OA.
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(step, loo, marker="o", label="15-shot train LOO-1NN")
    plt.plot(step, val, marker="s", label="Validation OA")
    valid_episode = np.isfinite(episode)
    if valid_episode.any():
        plt.plot(
            step[valid_episode],
            episode[valid_episode],
            linestyle="--",
            alpha=0.7,
            label="Episode query accuracy",
        )
    plt.xlabel("Target adapter training step")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 101)
    plt.title("Overfitting diagnostic: train vs validation")
    plt.grid(alpha=0.25)
    plt.legend()
    path = os.path.join(out_dir, "target_train_vs_val_accuracy.png")
    _finish_figure(path)
    saved.append(path)

    # 2) Generalization gap.
    plt.figure(figsize=(8.0, 5.2))
    plt.axhline(0.0, linewidth=1.0)
    plt.plot(step, gap, marker="o")
    plt.xlabel("Target adapter training step")
    plt.ylabel("LOO train accuracy - Validation OA (pp)")
    plt.title("Target generalization gap")
    plt.grid(alpha=0.25)
    path = os.path.join(out_dir, "target_generalization_gap.png")
    _finish_figure(path)
    saved.append(path)

    # 3) Prototype loss.
    finite = np.isfinite(proto)
    if finite.any():
        plt.figure(figsize=(8.0, 5.2))
        plt.plot(step[finite], proto[finite], marker="o")
        plt.xlabel("Target adapter training step")
        plt.ylabel("Prototype cross-entropy")
        plt.title("Target episodic prototype loss")
        plt.grid(alpha=0.25)
        path = os.path.join(out_dir, "target_proto_loss.png")
        _finish_figure(path)
        saved.append(path)

    # 4) Target adapter movement from source-informed initialization.
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(step, delta, marker="o")
    plt.xlabel("Target adapter training step")
    plt.ylabel("Relative parameter change (%)")
    plt.title("Target adapter movement from initialization")
    plt.grid(alpha=0.25)
    path = os.path.join(out_dir, "target_adapter_relative_change.png")
    _finish_figure(path)
    saved.append(path)

    save_history_csv(
        history,
        os.path.join(out_dir, "target_diagnostics.csv"),
    )

    return saved


def get_dataset_labels(dataset):
    if hasattr(dataset, "labels"):
        labels = dataset.labels
        if torch.is_tensor(labels):
            return labels.cpu().long()
        return torch.as_tensor(labels).long()

    if isinstance(dataset, Subset):
        parent = get_dataset_labels(dataset.dataset)
        return parent[torch.as_tensor(dataset.indices).long()]

    # Safe fallback for a small dataset. Avoid this for huge datasets when labels
    # are already stored as a dataset attribute.
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item[1]))
    return torch.LongTensor(labels)


def make_stratified_subset_loader(dataset, n_classes, per_class, batch_size, seed):
    labels = get_dataset_labels(dataset)
    rng = np.random.RandomState(seed)
    selected = []

    labels_np = labels.numpy()

    for cls in range(n_classes):
        idx = np.where(labels_np == cls)[0]
        if len(idx) == 0:
            continue

        if len(idx) > per_class:
            idx = rng.choice(
                idx,
                size=per_class,
                replace=False,
            )

        selected.extend(idx.tolist())

    subset = Subset(dataset, selected)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


@torch.no_grad()
def extract_target_features_and_alpha(model, loader, device, use_dictionary):
    model.eval()
    feats, labels, alphas = [], [], []

    for x, y in loader:
        out = model.encode_target(
            x.to(device),
            use_dictionary,
        )

        feats.append(out["feature"].cpu())
        labels.append(y.cpu())

        if out["alpha"] is not None:
            alphas.append(out["alpha"].cpu())

    return (
        torch.cat(feats, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(alphas, dim=0) if alphas else None,
    )


@torch.no_grad()
def collect_domain_atom_usage(model, loader, device, domain_name=None,
                              is_target=False, max_samples=2048):
    model.eval()
    alpha_sum = None
    count = 0

    for x, _y in loader:
        x = x.to(device)

        if is_target:
            out = model.encode_target(x, use_dictionary=True)
        else:
            out = model.encode_source(x, domain_name, use_dictionary=True)

        alpha = out["alpha"]
        if alpha is None:
            return None

        cur = alpha.sum(dim=0).cpu()
        alpha_sum = cur if alpha_sum is None else alpha_sum + cur
        count += alpha.size(0)

        if count >= max_samples:
            break

    if count == 0:
        return None

    return alpha_sum / float(count)


def _fit_tsne(features, seed, max_iter=1000):
    x = np.asarray(features, dtype=np.float32)

    if x.shape[1] > 50:
        x = PCA(
            n_components=50,
            random_state=seed,
        ).fit_transform(x)

    n = len(x)
    perplexity = min(
        30.0,
        max(5.0, (n - 1) / 3.0),
    )

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )

    try:
        tsne = TSNE(max_iter=max_iter, **kwargs)
    except TypeError:
        # Compatibility with older scikit-learn versions.
        tsne = TSNE(n_iter=max_iter, **kwargs)

    return tsne.fit_transform(x)


def plot_target_tsne(model, target, cfg, args, device, out_dir):
    use_dictionary = cfg.ablation != "no_dictionary"

    train_f, train_y, _ = extract_target_features_and_alpha(
        model,
        target.train_eval_loader,
        device,
        use_dictionary,
    )

    test_subset_loader = make_stratified_subset_loader(
        target.test_loader.dataset,
        target.n_classes,
        cfg.tsne_samples_per_class,
        args.batch_size,
        args.seed + 880001,
    )

    test_f, test_y, _ = extract_target_features_and_alpha(
        model,
        test_subset_loader,
        device,
        use_dictionary,
    )

    all_f = torch.cat([train_f, test_f], dim=0).numpy()
    embedding = _fit_tsne(
        all_f,
        seed=args.seed,
        max_iter=cfg.tsne_max_iter,
    )

    n_train = len(train_f)
    emb_train = embedding[:n_train]
    emb_test = embedding[n_train:]

    plt.figure(figsize=(9.0, 7.0))
    cmap = plt.cm.get_cmap("tab20", target.n_classes)

    for cls in range(target.n_classes):
        tr_mask = train_y.numpy() == cls
        te_mask = test_y.numpy() == cls

        plt.scatter(
            emb_train[tr_mask, 0],
            emb_train[tr_mask, 1],
            s=28,
            marker="o",
            alpha=0.90,
            color=cmap(cls),
            label=f"Class {cls}",
        )

        plt.scatter(
            emb_test[te_mask, 0],
            emb_test[te_mask, 1],
            s=18,
            marker="x",
            alpha=0.55,
            color=cmap(cls),
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Target semantic manifold: circles=train, x=test")
    plt.legend(
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    path = os.path.join(
        out_dir,
        "target_tsne_train_test.png",
    )

    _finish_figure(path)
    return path


def plot_semantic_atom_heatmaps(model, source_loaders, target, cfg, args, device, out_dir):
    if cfg.ablation == "no_dictionary":
        return []

    saved = []
    rows = []
    names = []

    # Source-domain profiles.
    for name in model.source_names:
        usage = collect_domain_atom_usage(
            model,
            source_loaders[name],
            device,
            domain_name=name,
            is_target=False,
            max_samples=cfg.atom_samples_per_domain,
        )

        if usage is not None:
            rows.append(usage.numpy())
            names.append(name)

    # Target-train profile.
    usage = collect_domain_atom_usage(
        model,
        target.train_eval_loader,
        device,
        is_target=True,
        max_samples=10**9,
    )

    if usage is not None:
        rows.append(usage.numpy())
        names.append("Target-train")

    # Target-test profile on a stratified sample.
    target_test_subset = make_stratified_subset_loader(
        target.test_loader.dataset,
        target.n_classes,
        cfg.tsne_samples_per_class,
        args.batch_size,
        args.seed + 880002,
    )

    usage = collect_domain_atom_usage(
        model,
        target_test_subset,
        device,
        is_target=True,
        max_samples=10**9,
    )

    if usage is not None:
        rows.append(usage.numpy())
        names.append("Target-test")

    if rows:
        matrix = np.stack(rows, axis=0)

        plt.figure(figsize=(12.0, max(4.0, 0.65 * len(names) + 2.0)))
        image = plt.imshow(matrix, aspect="auto")
        plt.colorbar(image, label="Mean attention weight")
        plt.yticks(np.arange(len(names)), names)

        tick_step = max(1, cfg.num_atoms // 16)
        ticks = np.arange(0, cfg.num_atoms, tick_step)
        plt.xticks(ticks, [str(i) for i in ticks])
        plt.xlabel("Shared semantic atom index")
        plt.ylabel("Domain")
        plt.title("Shared semantic atom usage across domains")

        path = os.path.join(out_dir, "semantic_atom_usage_domains.png")
        _finish_figure(path)
        saved.append(path)

        np.savetxt(
            os.path.join(out_dir, "semantic_atom_usage_domains.csv"),
            matrix,
            delimiter=",",
        )

    # Target class-wise profiles: train vs sampled unseen test.
    train_f, train_y, train_alpha = extract_target_features_and_alpha(
        model,
        target.train_eval_loader,
        device,
        use_dictionary=True,
    )

    test_f, test_y, test_alpha = extract_target_features_and_alpha(
        model,
        target_test_subset,
        device,
        use_dictionary=True,
    )

    if train_alpha is not None and test_alpha is not None:
        train_profiles = []
        test_profiles = []

        for cls in range(target.n_classes):
            tr = train_alpha[train_y == cls].mean(dim=0)
            te = test_alpha[test_y == cls].mean(dim=0)
            train_profiles.append(tr.numpy())
            test_profiles.append(te.numpy())

        train_profiles = np.stack(train_profiles)
        test_profiles = np.stack(test_profiles)

        plt.figure(figsize=(12.0, max(6.0, 0.36 * target.n_classes + 2.0)))
        image = plt.imshow(train_profiles, aspect="auto")
        plt.colorbar(image, label="Mean attention weight")
        plt.yticks(
            np.arange(target.n_classes),
            [f"Class {i}" for i in range(target.n_classes)],
        )
        tick_step = max(1, cfg.num_atoms // 16)
        ticks = np.arange(0, cfg.num_atoms, tick_step)
        plt.xticks(ticks, [str(i) for i in ticks])
        plt.xlabel("Shared semantic atom index")
        plt.ylabel("Target class")
        plt.title("Target 15-shot class-wise semantic atom usage")
        path = os.path.join(out_dir, "target_train_class_atom_usage.png")
        _finish_figure(path)
        saved.append(path)

        # Per-class train/test atom-profile cosine similarity.
        tr_norm = train_profiles / (
            np.linalg.norm(train_profiles, axis=1, keepdims=True) + 1e-12
        )
        te_norm = test_profiles / (
            np.linalg.norm(test_profiles, axis=1, keepdims=True) + 1e-12
        )
        similarity = (tr_norm * te_norm).sum(axis=1)

        plt.figure(figsize=(9.0, 5.2))
        plt.bar(np.arange(target.n_classes), similarity)
        plt.ylim(0.0, 1.02)
        plt.xlabel("Target class")
        plt.ylabel("Cosine similarity")
        plt.title("Train-test semantic atom profile consistency")
        plt.grid(axis="y", alpha=0.25)
        path = os.path.join(out_dir, "target_atom_profile_train_test_similarity.png")
        _finish_figure(path)
        saved.append(path)

        with open(
            os.path.join(out_dir, "target_atom_profile_similarity.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["class", "cosine_similarity"])
            for cls, value in enumerate(similarity):
                writer.writerow([cls, float(value)])

    return saved


def write_overfitting_report(history, path):
    if not history or not history.get("step"):
        return

    steps = np.asarray(history["step"], dtype=int)
    val = np.asarray(history["val_oa"], dtype=float)
    loo = np.asarray(history["loo_train_acc"], dtype=float)
    gap = np.asarray(history["generalization_gap"], dtype=float)
    delta = np.asarray(history["adapter_relative_change"], dtype=float)

    best_idx = int(np.nanargmax(val))
    best_step = int(steps[best_idx])
    best_val = float(val[best_idx])
    final_val = float(val[-1])
    post_peak_drop = best_val - final_val

    with open(path, "w", encoding="utf-8") as f:
        f.write("TARGET ADAPTER OVERFITTING DIAGNOSTIC\n")
        f.write("=" * 72 + "\n")
        f.write("Validation was monitored for diagnosis only.\n")
        f.write("No early stopping/checkpoint selection used these values.\n\n")
        f.write(f"Best diagnostic Val OA: {best_val*100:.2f}% at step {best_step}\n")
        f.write(f"Final diagnostic Val OA: {final_val*100:.2f}%\n")
        f.write(f"Post-peak Val OA drop: {post_peak_drop*100:.2f} pp\n")
        f.write(f"Final LOO-1NN train accuracy: {loo[-1]*100:.2f}%\n")
        f.write(f"Final LOO-train minus Val gap: {gap[-1]*100:.2f} pp\n")
        f.write(f"Final target-adapter relative parameter change: {delta[-1]*100:.3f}%\n")


def generate_visualizations(model, source_loaders, target, source_history,
                            target_history, args, cfg, run_dir, device,
                            heavy_visuals=True):
    out_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    saved.extend(plot_source_history(source_history, out_dir))
    saved.extend(plot_target_history(target_history, out_dir))

    write_overfitting_report(
        target_history,
        os.path.join(out_dir, "overfitting_diagnostic.txt"),
    )

    if heavy_visuals:
        try:
            saved.append(
                plot_target_tsne(
                    model,
                    target,
                    cfg,
                    args,
                    device,
                    out_dir,
                )
            )
        except Exception as exc:
            print(f"[Visualization warning] t-SNE failed: {exc}")

        try:
            saved.extend(
                plot_semantic_atom_heatmaps(
                    model,
                    source_loaders,
                    target,
                    cfg,
                    args,
                    device,
                    out_dir,
                )
            )
        except Exception as exc:
            print(f"[Visualization warning] atom heatmap failed: {exc}")

    print(f"Visualization directory: {out_dir}")
    for path in saved:
        print(f"  [FIG] {path}")

    return out_dir



@dataclass
class SemanticManifoldConfig:
    ablation: str = "full"
    target_shots: int = 15
    support_num: int = 5
    adapter_channels: int = 64
    num_atoms: int = 64
    dictionary_temperature: float = 0.10
    dictionary_target_entropy: float = 0.45
    contrast_temperature: float = 0.10
    lambda_source_cls: float = 1.0
    lambda_supcon: float = 0.10
    lambda_dict_diversity: float = 0.02
    lambda_dict_selectivity: float = 0.02
    lambda_dict_balance: float = 0.01
    target_proto_temperature: float = 0.10
    target_label_smoothing: float = 0.05
    lambda_target_anchor: float = 1.0
    target_lr: float = 5e-4
    source_steps: int = 1500
    target_steps: int = 500
    classifier: str = "knn"
    print_interval: int = 50

    # Visualization / overfitting monitoring
    visualize: bool = True
    visualize_all_runs: bool = False
    visual_interval: int = 25
    tsne_samples_per_class: int = 100
    tsne_max_iter: int = 1000
    atom_samples_per_domain: int = 2048


ABLATION_CHOICES = ["full", "no_dictionary", "no_supcon",
                    "random_target_init", "no_target_tune"]


def train_single_run(args, cfg, run_index=1):
    set_random_seed(args.seed)
    device = as_device(args.device)
    print("\n" + "#" * 100)
    print(f"# OUR-V9 SEMANTIC MANIFOLD | ablation={cfg.ablation}")
    print(f"# Sources={args.sources}")
    print(f"# Target={args.target} | seed={args.seed} | shots={cfg.target_shots}")
    print("# Shared object = semantic representation basis/manifold, NOT a common distribution.")
    print("# Target-private learning = shallow spectral adapter ONLY.")
    print("#" * 100)

    source_loaders, source_channels, source_classes = {}, {}, {}
    for n in args.sources:
        loader, ch, nc = load_source_data(n, args, args.seed)
        source_loaders[n] = loader; source_channels[n] = ch; source_classes[n] = nc
        print(f"Source {n}: channels={ch}, classes={nc}, samples={len(loader.dataset)}")

    target = load_target_data_fixed_shot(args.target, args, args.seed, cfg.target_shots)
    if any(c != cfg.target_shots for c in target.train_counts):
        raise RuntimeError(f"Strict shot check failed: {target.train_counts}")
    print(f"Target {args.target}: channels={target.channels}, classes={target.n_classes}, "
          f"train={len(target.train_loader.dataset)}, val={len(target.val_loader.dataset)}, "
          f"test={len(target.test_loader.dataset)}")

    model = SemanticManifoldModel(
        source_channels, source_classes, target.channels, cfg.adapter_channels,
        args.hidden_channels, args.feature_dim, cfg.num_atoms,
        cfg.dictionary_temperature, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    target_params = sum(p.numel() for p in model.target_adapter.parameters())
    shared_params = sum(p.numel() for p in list(model.shared_encoder.parameters()) +
                        list(model.semantic_dictionary.parameters()))
    print(f"Parameters: total={total_params:,}, shared={shared_params:,}, "
          f"target_adapter={target_params:,} ({100*target_params/max(total_params,1):.3f}%)")

    run_dir = os.path.join(
        "./checkpoints_v9_semantic_manifold",
        str(args.target),
        cfg.ablation,
        f"seed_{args.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)

    start = time.time()
    source_history = train_source_stage(model, source_loaders, args, cfg, device)
    target_stats = train_target_adapter_stage(model, target, args, cfg, device)
    train_time = time.time() - start

    use_dictionary = cfg.ablation != "no_dictionary"
    val = evaluate_target_streaming(model, target.train_eval_loader, target.val_loader,
                                    target.n_classes, device, use_dictionary, cfg.classifier)
    ts = time.time()
    test = evaluate_target_streaming(model, target.train_eval_loader, target.test_loader,
                                     target.n_classes, device, use_dictionary, cfg.classifier)
    test_time = time.time() - ts
    print(f"Diagnostic Val | OA={val['oa']*100:.2f}% | AA={val['aa']*100:.2f}% | Kappa={val['kappa']:.4f}")
    print(f"Final Test | OA={test['oa']*100:.2f}% | AA={test['aa']*100:.2f}% | Kappa={test['kappa']:.4f}")

    if cfg.visualize:
        heavy = (run_index == 1) or cfg.visualize_all_runs
        generate_visualizations(
            model=model,
            source_loaders=source_loaders,
            target=target,
            source_history=source_history,
            target_history=target_stats.get("history", {}),
            args=args,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            heavy_visuals=heavy,
        )

    return test, val, target_stats, train_time, test_time, {
        "total_params": total_params, "shared_params": shared_params,
        "target_adapter_params": target_params}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sm_ablation", type=str, default="full", choices=ABLATION_CHOICES)
    pre.add_argument("--sm_target_shots", type=int, default=15)
    pre.add_argument("--sm_support_num", type=int, default=5)
    pre.add_argument("--sm_adapter_channels", type=int, default=64)
    pre.add_argument("--sm_num_atoms", type=int, default=64)
    pre.add_argument("--sm_dictionary_temperature", type=float, default=0.10)
    pre.add_argument("--sm_dictionary_target_entropy", type=float, default=0.45)
    pre.add_argument("--sm_contrast_temperature", type=float, default=0.10)
    pre.add_argument("--sm_lambda_supcon", type=float, default=0.10)
    pre.add_argument("--sm_lambda_dict_diversity", type=float, default=0.02)
    pre.add_argument("--sm_lambda_dict_selectivity", type=float, default=0.02)
    pre.add_argument("--sm_lambda_dict_balance", type=float, default=0.01)
    pre.add_argument("--sm_target_proto_temperature", type=float, default=0.10)
    pre.add_argument("--sm_target_label_smoothing", type=float, default=0.05)
    pre.add_argument("--sm_lambda_target_anchor", type=float, default=1.0)
    pre.add_argument("--sm_target_lr", type=float, default=-1.0)
    pre.add_argument("--sm_source_steps", type=int, default=-1)
    pre.add_argument("--sm_target_steps", type=int, default=-1)
    pre.add_argument("--sm_classifier", type=str, default="knn", choices=["knn", "proto"])
    pre.add_argument("--sm_print_interval", type=int, default=50)

    # Visualization / overfitting diagnostics
    pre.add_argument("--sm_no_visualization", action="store_true")
    pre.add_argument("--sm_visualize_all_runs", action="store_true")
    pre.add_argument("--sm_visual_interval", type=int, default=25)
    pre.add_argument("--sm_tsne_samples_per_class", type=int, default=100)
    pre.add_argument("--sm_tsne_max_iter", type=int, default=1000)
    pre.add_argument("--sm_atom_samples_per_domain", type=int, default=2048)
    extra, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = GetParameters()
    if getattr(args, "source", None) is None:
        raise ValueError("Please provide --source")
    if getattr(args, "target", None) is None:
        raise ValueError("Please provide --target")
    args.sources = ChooseSource(args.source)
    defaults = {
        "hidden_channels": 64, "feature_dim": 128, "dropout": 0.1,
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
        "patch_size": 9, "grad_clip": 5.0, "n_run": 5, "seed": 0,
        "epochs": 2000, "need_pca": False, "no_normalize": False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    if not hasattr(args, "val_per_class"):
        args.val_per_class = extra.sm_target_shots

    total = max(int(args.epochs), 2)
    if extra.sm_source_steps > 0 and extra.sm_target_steps > 0:
        source_steps, target_steps = extra.sm_source_steps, extra.sm_target_steps
    elif extra.sm_source_steps > 0:
        source_steps = extra.sm_source_steps; target_steps = max(1, total - source_steps)
    elif extra.sm_target_steps > 0:
        target_steps = extra.sm_target_steps; source_steps = max(1, total - target_steps)
    else:
        source_steps = int(round(total * 0.75)); target_steps = max(1, total - source_steps)
    target_lr = extra.sm_target_lr if extra.sm_target_lr > 0 else min(args.lr, 5e-4)

    cfg = SemanticManifoldConfig(
        ablation=extra.sm_ablation, target_shots=extra.sm_target_shots,
        support_num=extra.sm_support_num, adapter_channels=extra.sm_adapter_channels,
        num_atoms=extra.sm_num_atoms, dictionary_temperature=extra.sm_dictionary_temperature,
        dictionary_target_entropy=extra.sm_dictionary_target_entropy,
        contrast_temperature=extra.sm_contrast_temperature,
        lambda_source_cls=getattr(args, "lambda_source_cls", 1.0),
        lambda_supcon=extra.sm_lambda_supcon,
        lambda_dict_diversity=extra.sm_lambda_dict_diversity,
        lambda_dict_selectivity=extra.sm_lambda_dict_selectivity,
        lambda_dict_balance=extra.sm_lambda_dict_balance,
        target_proto_temperature=extra.sm_target_proto_temperature,
        target_label_smoothing=extra.sm_target_label_smoothing,
        lambda_target_anchor=extra.sm_lambda_target_anchor,
        target_lr=target_lr, source_steps=source_steps, target_steps=target_steps,
        classifier=extra.sm_classifier, print_interval=max(1, extra.sm_print_interval),
        visualize=not extra.sm_no_visualization,
        visualize_all_runs=extra.sm_visualize_all_runs,
        visual_interval=max(1, extra.sm_visual_interval),
        tsne_samples_per_class=max(1, extra.sm_tsne_samples_per_class),
        tsne_max_iter=max(250, extra.sm_tsne_max_iter),
        atom_samples_per_domain=max(1, extra.sm_atom_samples_per_domain))
    return args, cfg


def save_results(args, cfg, rows, class_acc_rows):
    out_dir = os.path.join("./checkpoints_v9_semantic_manifold", str(args.target), cfg.ablation)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_run_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    oa = np.array([r["oa"] for r in rows]); aa = np.array([r["aa"] for r in rows])
    kp = np.array([r["kappa"] for r in rows]); ca = np.stack(class_acc_rows)
    summary = os.path.join(out_dir, "summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\nOUR-V9 TRANSFERABLE SEMANTIC MANIFOLD\n" + "=" * 100 + "\n")
        f.write(f"Target: {args.target}\nSources: {args.sources}\nAblation: {cfg.ablation}\n")
        f.write(f"Runs: {len(rows)}\nTarget shots/class: {cfg.target_shots}\n")
        f.write(f"Target episode: support={cfg.support_num}, query={cfg.target_shots-cfg.support_num}\n")
        f.write("Shared object: transferable semantic manifold/basis; no common-distribution assumption.\n")
        f.write("Target representation tuning: shallow target spectral adapter ONLY.\n")
        f.write("Shared encoder/dictionary during target tuning: FROZEN.\n")
        f.write(f"Source steps: {cfg.source_steps}\nTarget steps: {cfg.target_steps}\n")
        f.write(f"Classifier: {cfg.classifier}\n\n")
        f.write(f"OA: {oa.mean()*100:.2f} +/- {oa.std()*100:.4f}\n")
        f.write(f"AA: {aa.mean()*100:.2f} +/- {aa.std()*100:.4f}\n")
        f.write(f"Kappa: {kp.mean():.4f} +/- {kp.std():.6f}\n")
        f.write("\nPer-Class Accuracy (Mean +/- Std) [%]\n")
        for i, (m, s) in enumerate(zip(ca.mean(0)*100, ca.std(0)*100)):
            f.write(f"Class {i}: {m:.2f} +/- {s:.4f}\n")
    print(f"Saved summary: {summary}")


def main():
    args, cfg = parse_args()
    base_seed = int(args.seed)
    rows, class_acc = [], []
    for run in range(1, args.n_run + 1):
        args.seed = base_seed + run - 1
        test, val, target_stats, train_time, test_time, ps = train_single_run(args, cfg, run_index=run)
        rows.append({
            "run": run, "seed": args.seed, "oa": test["oa"], "aa": test["aa"],
            "kappa": test["kappa"], "val_oa_diagnostic": val["oa"],
            "target_final_proto_loss": target_stats.get("final_proto_loss", float("nan")),
            "target_final_anchor_loss": target_stats.get("final_anchor_loss", float("nan")),
            "total_params": ps["total_params"], "shared_params": ps["shared_params"],
            "target_adapter_params": ps["target_adapter_params"],
            "train_time": train_time, "test_time": test_time,
        })
        class_acc.append(test["class_acc"])
        print(f"Run {run}/{args.n_run}: OA={test['oa']*100:.2f}%, AA={test['aa']*100:.2f}%, "
              f"Kappa={test['kappa']:.4f}")
    save_results(args, cfg, rows, class_acc)
    oa = np.array([r["oa"] for r in rows]); aa = np.array([r["aa"] for r in rows]); kp = np.array([r["kappa"] for r in rows])
    print("\n" + "=" * 100)
    print(f"OUR-V9 FINAL | target={args.target} | ablation={cfg.ablation}")
    print(f"OA: {oa.mean()*100:.2f} +/- {oa.std()*100:.4f}")
    print(f"AA: {aa.mean()*100:.2f} +/- {aa.std()*100:.4f}")
    print(f"Kappa: {kp.mean():.4f} +/- {kp.std():.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()