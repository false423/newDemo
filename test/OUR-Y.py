
import csv
import math
import os
import random
import argparse
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
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import GetParameters, ChooseSource

# Plotting helpers used by the second code.  They are optional here so that
# training still works even when Visdom is not installed/running.
try:
    import visdom
except Exception:
    visdom = None

try:
    from utils.display import display_predictions, convert_to_color
except Exception:
    display_predictions = None
    convert_to_color = None

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
    height: int
    width: int
    gt: np.ndarray
    test_coords: List[Tuple[int, int]]


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
        height=int(gt.shape[0]), width=int(gt.shape[1]), gt=gt.copy(),
        test_coords=list(te_c),
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
    trainable = (list(model.source_adapters.parameters()) +
                 list(model.shared_encoder.parameters()) +
                 list(model.semantic_dictionary.parameters()) +
                 list(model.source_heads.parameters()))
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.source_steps, 1), eta_min=args.lr * 0.01)

    for step in range(1, cfg.source_steps + 1):
        model.train()
        outputs, labels = {}, {}
        for n in names:
            x, y = next(iters[n])
            x, y = x.to(device), y.to(device)
            outputs[n] = model.encode_source(x, n, use_dictionary)
            labels[n] = y
        loss_cls = torch.stack([
            F.cross_entropy(model.source_heads[n](outputs[n]["feature"]), labels[n])
            for n in names
        ]).mean()
        if use_supcon:
            loss_sup = torch.stack([
                supervised_contrastive_loss(outputs[n]["feature"], labels[n], cfg.contrast_temperature)
                for n in names
            ]).mean()
        else:
            loss_sup = torch.tensor(0.0, device=device)

        if use_dictionary:
            alpha_all = torch.cat([outputs[n]["alpha"] for n in names], dim=0)
            reg = dictionary_regularization(model.semantic_dictionary, alpha_all,
                                            cfg.dictionary_target_entropy)
            ldiv, lsel, lbal = reg["diversity"], reg["selectivity"], reg["balance"]
        else:
            zero = torch.tensor(0.0, device=device)
            reg = {"sample_entropy": zero, "usage_entropy": zero, "max_usage": zero}
            ldiv = lsel = lbal = zero

        total = (cfg.lambda_source_cls * loss_cls + cfg.lambda_supcon * loss_sup +
                 cfg.lambda_dict_diversity * ldiv +
                 cfg.lambda_dict_selectivity * lsel + cfg.lambda_dict_balance * lbal)
        opt.zero_grad(); total.backward()
        if getattr(args, "grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step(); sched.step()

        if step == 1 or step % cfg.print_interval == 0 or step == cfg.source_steps:
            gate = float(model.semantic_dictionary.effective_gate().detach()) if use_dictionary else 0.0
            print(f"[Source] {step:5d}/{cfg.source_steps} | total={total.item():.4f} | "
                  f"cls={loss_cls.item():.4f} | supcon={loss_sup.item():.4f} | "
                  f"dict_div={ldiv.item():.4f} | dict_sel={lsel.item():.4f} | dict_bal={lbal.item():.4f}")
            print(f"         gate={gate:.4f} | sample_H={float(reg['sample_entropy']):.4f} | "
                  f"usage_H={float(reg['usage_entropy']):.4f} | max_atom={float(reg['max_usage']):.4f}")


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


def train_target_adapter_stage(model, target, args, cfg, device):
    print("\n" + "=" * 96)
    print("STAGE II - 15-SHOT TARGET SPECTRAL ADAPTER TUNING")
    print("Target labels are used, but ONLY the shallow Target Adapter is trainable.")
    print("Shared Encoder + Semantic Dictionary are frozen.")
    print("=" * 96)
    use_dictionary = cfg.ablation != "no_dictionary"

    if cfg.ablation == "random_target_init":
        init_weights = {n: 0.0 for n in model.source_names}
        print("[Target Init] RANDOM (ablation)")
    else:
        init_weights = initialize_target_adapter_from_sources(model)
        print("[Target Init] source-informed: " + ", ".join(
            f"{n}:{w:.3f}" for n, w in init_weights.items()))

    if cfg.ablation == "no_target_tune":
        print("[Target Tune] skipped by ablation.")
        return {"init_weights": init_weights, "final_proto_loss": float("nan"),
                "final_anchor_loss": float("nan")}

    freeze_for_target_tuning(model)
    model.eval(); model.target_adapter.train()
    trainable = [p for p in model.target_adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.target_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.target_steps, 1), eta_min=cfg.target_lr * 0.01)
    anchor = snapshot_module_parameters(model.target_adapter)
    all_x, all_y = collect_small_dataset(target.train_eval_loader)
    expected = cfg.target_shots * target.n_classes
    if len(all_y) != expected:
        raise RuntimeError(f"Expected target pool {expected}, got {len(all_y)}")
    gen = torch.Generator(); gen.manual_seed(args.seed + 555555)
    query_num = cfg.target_shots - cfg.support_num
    if query_num <= 0:
        raise ValueError("target_shots must be > support_num")
    stats = {}

    for step in range(1, cfg.target_steps + 1):
        model.eval(); model.target_adapter.train()
        sx, sy, qx, qy = sample_fewshot_episode(
            all_x, all_y, target.n_classes, cfg.support_num, query_num, gen)
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
        s_out = model.encode_target(sx, use_dictionary)
        q_out = model.encode_target(qx, use_dictionary)
        logits = cosine_prototype_logits(q_out["feature"], s_out["feature"], sy,
                                         target.n_classes, cfg.target_proto_temperature)
        loss_proto = F.cross_entropy(logits, qy, label_smoothing=cfg.target_label_smoothing)
        loss_anchor = module_anchor_loss(model.target_adapter, anchor)
        total = loss_proto + cfg.lambda_target_anchor * loss_anchor
        opt.zero_grad(); total.backward()
        if getattr(args, "grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step(); sched.step()
        stats = {"final_proto_loss": float(loss_proto.item()),
                 "final_anchor_loss": float(loss_anchor.item())}
        if step == 1 or step % cfg.print_interval == 0 or step == cfg.target_steps:
            with torch.no_grad():
                acc = (logits.argmax(1) == qy).float().mean().item()
            print(f"[TargetAdapter] {step:5d}/{cfg.target_steps} | total={total.item():.4f} | "
                  f"proto={loss_proto.item():.4f} | anchor={loss_anchor.item():.6f} | "
                  f"episode_acc={acc*100:.2f}%")
    stats["init_weights"] = init_weights
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


def get_output_dir(args, cfg):
    return os.path.join("./checkpoints_v9_semantic_manifold", str(args.target), cfg.ablation)


def _label_map_to_color(label_map: np.ndarray, n_classes: int):
    """Match the second code's convert_to_color behavior, with a local fallback."""
    if convert_to_color is not None:
        try:
            return convert_to_color(label_map.astype(np.float64), n_classes + 1, None)
        except Exception as exc:
            print(f"[Plot] convert_to_color failed ({exc}); using matplotlib fallback palette.")

    label_map = label_map.astype(np.int64)
    rgba = np.zeros(label_map.shape + (4,), dtype=np.float32)
    rgba[..., 3] = 1.0
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 1))
    for cls in range(1, n_classes + 1):
        rgba[label_map == cls] = cmap((cls - 1) % max(n_classes, 1))
    return rgba


def _save_map_image(color_img: np.ndarray, path: str, width: int, height: int, dpi: int = 300):
    plt.figure(figsize=(max(width, 1) / 100, max(height, 1) / 100), dpi=dpi)
    plt.imshow(color_img)
    plt.axis("off")
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def _display_with_visdom(color_img: np.ndarray, vis, caption: str):
    if vis is None or display_predictions is None:
        return
    try:
        display_predictions(color_img, vis, caption=caption)
    except Exception as exc:
        print(f"[Plot] Visdom display skipped: {exc}")


@torch.no_grad()
def predict_target_streaming(model, train_loader, eval_loader, n_classes,
                             device, use_dictionary, classifier):
    """Return predictions in eval_loader order, using the same classifier as evaluation."""
    train_f, train_y = extract_target_features(model, train_loader, device, use_dictionary)
    eval_f, eval_y = extract_target_features(model, eval_loader, device, use_dictionary)

    if classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        clf.fit(train_f.numpy(), train_y.numpy())
        pred = clf.predict(eval_f.numpy())
    elif classifier == "proto":
        protos = build_target_prototypes(train_f, train_y, n_classes)
        pred = torch.argmax(F.normalize(eval_f, dim=1) @ protos.t(), dim=1).numpy()
    else:
        raise ValueError(classifier)
    return pred, eval_y.numpy(), eval_f


def _safe_tsne(features: np.ndarray, seed: int):
    n = int(len(features))
    if n < 2:
        return None
    perplexity = min(30.0, max(1.0, float(n - 1)))
    return TSNE(
        n_components=2,
        init="pca",
        random_state=int(seed),
        perplexity=perplexity,
    ).fit_transform(features)


def _plot_tsne(points: np.ndarray, labels: np.ndarray, path: str, n_classes: int):
    if points is None or len(points) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 1))
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() == 0:
            continue
        color = np.array(cmap(i)).reshape(1, -1)
        ax.scatter(points[mask, 0], points[mask, 1], marker="o", s=4,
                   c=color, label=f"{i + 1}", alpha=0.8, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.legend(loc="lower right", fontsize=7, markerscale=2.5,
              frameon=True, framealpha=0.9, edgecolor="gray",
              handletextpad=0.3, borderpad=0.4, labelspacing=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def generate_target_visualizations(model, target, args, cfg, device, use_dictionary, vis=None):
    """
    Plotting logic transferred from the second code:
      1) Ground-truth map
      2) Classification map reconstructed at original pixel coordinates
      3) Raw patch t-SNE
      4) Learned semantic-feature t-SNE

    Unlike the second code, PNG saving does not depend on a live Visdom server.
    """
    draw_map = bool(getattr(args, "draw_map", False))
    draw_tsne = bool(getattr(args, "draw_tsne", False))
    if not (draw_map or draw_tsne):
        return

    figure_dir = os.path.join(get_output_dir(args, cfg), "figures")
    os.makedirs(figure_dir, exist_ok=True)

    if draw_map:
        print("Generating Ground Truth Map...")
        gt_mapped = np.zeros_like(target.gt, dtype=np.int64)
        label_to_index, _ = build_label_mapping(target.gt)
        for old_label, new_label in label_to_index.items():
            gt_mapped[target.gt == old_label] = new_label + 1

        gt_color = _label_map_to_color(gt_mapped, target.n_classes)
        gt_path = os.path.join(figure_dir, f"GT_{args.target}.png")
        _save_map_image(gt_color, gt_path, target.width, target.height, dpi=300)
        _display_with_visdom(gt_color, vis, f"Ground Truth of {args.target}")
        print(f"  Ground Truth saved to: {gt_path}")

        print("Generating Classification Map...")
        pred_arr, _, _ = predict_target_streaming(
            model, target.train_eval_loader, target.test_loader, target.n_classes,
            device, use_dictionary, cfg.classifier)

        if len(pred_arr) != len(target.test_coords):
            raise RuntimeError(
                f"Prediction/coordinate length mismatch: pred={len(pred_arr)}, "
                f"coords={len(target.test_coords)}")

        predict_mat = np.zeros((target.height, target.width), dtype=np.int64)
        for pred, (r, c) in zip(pred_arr, target.test_coords):
            predict_mat[int(r), int(c)] = int(pred) + 1

        color_img = _label_map_to_color(predict_mat, target.n_classes)
        pred_path = os.path.join(
            figure_dir, f"Pred_{args.target}_OUR-V9_seed{args.seed}.png")
        _save_map_image(color_img, pred_path, target.width, target.height, dpi=300)
        _display_with_visdom(
            color_img, vis, f"Classification Map of {args.target} - OUR-V9")
        print(f"  Classification Map saved to: {pred_path}")

    if draw_tsne:
        n_tsne = min(3000, len(target.test_loader.dataset))
        if n_tsne < 2:
            print("[Plot] t-SNE skipped: fewer than 2 target test samples.")
            return

        # ===== Raw patch t-SNE: same idea as the second code =====
        print("Generating Raw Feature t-SNE...")
        rng_raw = np.random.RandomState(int(args.seed) + 12345)
        indices_raw = np.sort(
            rng_raw.choice(len(target.test_loader.dataset), n_tsne, replace=False))
        index_set = set(indices_raw.tolist())
        raw_feats, raw_labels = [], []
        offset = 0
        for batch in target.test_loader:
            x, y = batch[0], batch[1]
            bs = x.size(0)
            for i in range(bs):
                if (offset + i) in index_set:
                    raw_feats.append(x[i].reshape(-1).cpu())
                    raw_labels.append(int(y[i].item()))
            offset += bs
            if len(raw_feats) >= n_tsne:
                break

        raw_np = torch.stack(raw_feats).numpy()
        raw_labels_np = np.asarray(raw_labels, dtype=np.int64)
        raw_points = _safe_tsne(raw_np, args.seed)
        raw_path = os.path.join(
            figure_dir, f"tsne_raw_{args.target}_seed{args.seed}.png")
        _plot_tsne(raw_points, raw_labels_np, raw_path, target.n_classes)
        print(f"  Raw t-SNE saved to: {raw_path}")

        # ===== Learned semantic feature t-SNE =====
        print("Generating Model Feature t-SNE...")
        test_feat, test_y = extract_target_features(
            model, target.test_loader, device, use_dictionary)
        feat_np = test_feat.numpy()
        labels_np = test_y.numpy()
        n_model = min(3000, len(feat_np))
        rng_model = np.random.RandomState(int(args.seed) + 54321)
        indices = rng_model.choice(len(feat_np), n_model, replace=False)
        feat_sub = feat_np[indices]
        labels_sub = labels_np[indices]
        model_points = _safe_tsne(feat_sub, args.seed)
        tsne_path = os.path.join(
            figure_dir, f"tsne_OUR-V9_{args.target}_seed{args.seed}.png")
        _plot_tsne(model_points, labels_sub, tsne_path, target.n_classes)
        print(f"  Model Feature t-SNE saved to: {tsne_path}")


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


ABLATION_CHOICES = ["full", "no_dictionary", "no_supcon",
                    "random_target_init", "no_target_tune"]


def train_single_run(args, cfg, vis=None):
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

    start = time.time()
    train_source_stage(model, source_loaders, args, cfg, device)
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

    generate_target_visualizations(
        model, target, args, cfg, device, use_dictionary, vis=vis)

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
    pre.add_argument("--sm_draw_map", action="store_true",
                     help="Draw/save ground-truth and classification maps.")
    pre.add_argument("--sm_draw_tsne", action="store_true",
                     help="Draw/save raw-patch and learned-feature t-SNE figures.")
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

    # Same high-level behavior as the second code: --draw_figure enables both
    # classification maps and t-SNE.  The sm_* flags can also enable them separately.
    draw_figure = bool(getattr(args, "draw_figure", False))
    args.draw_map = bool(extra.sm_draw_map or getattr(args, "draw_map", False) or draw_figure)
    args.draw_tsne = bool(extra.sm_draw_tsne or getattr(args, "draw_tsne", False) or draw_figure)

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
        classifier=extra.sm_classifier, print_interval=max(1, extra.sm_print_interval))
    return args, cfg


def save_results(args, cfg, rows, class_acc_rows):
    out_dir = get_output_dir(args, cfg)
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

    vis = None
    if bool(getattr(args, "draw_figure", False)) and visdom is not None:
        try:
            vis = visdom.Visdom(env="hyperspectral", port=8813)
            if not vis.check_connection():
                print("Visdom is not connected. PNG figures will still be saved.")
                vis = None
        except Exception as exc:
            print(f"Visdom initialization failed ({exc}). PNG figures will still be saved.")
            vis = None
    elif bool(getattr(args, "draw_figure", False)) and visdom is None:
        print("Visdom is not installed. PNG figures will still be saved.")

    base_seed = int(args.seed)
    rows, class_acc = [], []
    for run in range(1, args.n_run + 1):
        args.seed = base_seed + run - 1
        test, val, target_stats, train_time, test_time, ps = train_single_run(args, cfg, vis=vis)
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