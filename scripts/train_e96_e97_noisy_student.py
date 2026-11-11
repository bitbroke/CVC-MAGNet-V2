"""Full-Cohort Noisy-Student Distillation across all 21,855 Studies (E96 & E97).

Features:
- Full exposure to all 21,855 precomputed training segmentation masks (consensus + pseudo-labels).
- 4-Zone Mask-Guided Attention Pooling (MGAP) + 11-Class Multi-Task Auxiliary Head.
- Noisy-Student Regularization:
  * Mixup augmentation (alpha=0.2)
  * Stochastic Ribbon Dropout (simulating rib-crossing gaps)
  * Affine rotation & translation
- Warmstarted from converged E89/E90 checkpoints.
- Strict evaluation on locked validation cohort (N=2,558).
- Zero contact with sealed test partition (N=2,284).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Set deterministic flags for cuDNN stability
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mask_guided_spatial_net import MaskGuidedSpatialNet
from src.data.full_spatial_dataset import CLASSES, generate_anatomical_channel

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"

CONSENSUS_SEG_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_cache_512"
GENERAL_SEG_DIR = DATA_DIR / "curated_cvc_v1" / "seg_cache_512"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

RUNS_DIR = PROJECT_ROOT / "runs"
VAL_PREDS_DIR = RUNS_DIR / "val_preds"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"
LEDGER_PARQUET = REPORTS_DIR / "master_ablation_ledger.parquet"

AUX_CLASSES = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]


class FullCohortNoisyStudentDataset(Dataset):
    """Full-Cohort 3-Channel Dataset spanning all 21,855 training masks with Noisy Augmentations."""

    def __init__(
        self,
        df: pd.DataFrame,
        carina_lookup: Dict,
        zonal_lookup: Dict,
        images_dir: Path = IMAGES_DIR,
        consensus_dir: Path = CONSENSUS_SEG_DIR,
        general_dir: Path = GENERAL_SEG_DIR,
        img_size: int = 512,
        is_train: bool = False,
        ribbon_dropout_prob: float = 0.20,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.consensus_dir = Path(consensus_dir)
        self.general_dir = Path(general_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.ribbon_dropout_prob = ribbon_dropout_prob
        self.consensus_files = set(f[:-4] for f in os.listdir(self.consensus_dir) if f.endswith(".png"))

        sid_col = "StudyInstanceUID" if "StudyInstanceUID" in self.df.columns else "study_id"
        self.study_ids = self.df[sid_col].tolist()
        self.carina_lookup = carina_lookup
        self.zonal_lookup = zonal_lookup

        # Labels
        self.cvc_labels = self.df[CLASSES].values.astype(np.float32)
        if all(col in self.df.columns for col in AUX_CLASSES):
            self.aux_labels = self.df[AUX_CLASSES].values.astype(np.float32)
        else:
            self.aux_labels = np.zeros((len(self.df), len(AUX_CLASSES)), dtype=np.float32)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.study_ids[idx]
        size = self.img_size

        # 1. Channel 0: CLAHE native radiograph
        img_path = self.images_dir / f"{sid}.jpg"
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img_gray is None:
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)

        if img_gray is None:
            ch0 = np.zeros((size, size), dtype=np.float32)
        else:
            clahe_img = self.clahe.apply(img_gray)
            ch0 = cv2.resize(clahe_img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

        # 2. Channel 1: Segmentation mask (Consensus preferred, fallback to general cache)
        if sid in self.consensus_files:
            seg_path = self.consensus_dir / f"{sid}.png"
        else:
            seg_path = self.general_dir / f"{sid}.png"

        seg_uint8 = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)

        if seg_uint8 is not None:
            if seg_uint8.shape != (size, size):
                seg_uint8 = cv2.resize(seg_uint8, (size, size), interpolation=cv2.INTER_NEAREST)
            ch1 = seg_uint8.astype(np.float32) / 255.0
        else:
            ch1 = np.zeros((size, size), dtype=np.float32)

        # 3. Channel 2: Explicit anatomical coordinate map
        c_info = self.carina_lookup.get(sid, {})
        z_info = self.zonal_lookup.get(sid, {})

        xc_norm = float(c_info.get("raw_carina_x_norm", 0.4805))
        yc_norm = float(c_info.get("raw_carina_y_norm", 0.4000))
        xc = xc_norm * float(size)
        yc = yc_norm * float(size)

        conf_tip = float(z_info.get("conf_tip", 0.0))
        has_valid_tip = float(z_info.get("has_valid_tip", 0.0))
        dx_norm = float(z_info.get("dx_norm", 0.0))
        dy_norm = float(z_info.get("dy_norm", 0.0))

        if has_valid_tip > 0.5 and conf_tip >= 0.20:
            xtip = (xc_norm + dx_norm) * float(size)
            ytip = (yc_norm + dy_norm) * float(size)
        else:
            xtip, ytip = 0.0, 0.0

        ch2 = generate_anatomical_channel(
            xc=xc,
            yc=yc,
            xtip=xtip,
            ytip=ytip,
            conf_tip=conf_tip,
            size=size,
            mode="hybrid_ch2",
        )

        # 4. Noisy-Student Data Augmentation
        if self.is_train:
            # Affine jitter
            angle = np.random.uniform(-6.0, 6.0)
            scale = np.random.uniform(0.96, 1.04)
            tx = np.random.uniform(-8.0, 8.0)
            ty = np.random.uniform(-8.0, 8.0)

            center = (size / 2.0, size / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            ch0 = cv2.warpAffine(ch0, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ch1 = cv2.warpAffine(ch1, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            ch2 = cv2.warpAffine(ch2, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

            # Stochastic Ribbon Dropout: simulates broken catheter ribbons over dense bone
            if np.random.rand() < self.ribbon_dropout_prob:
                drop_w = np.random.randint(24, 72)
                drop_h = np.random.randint(24, 72)
                drop_x = np.random.randint(0, size - drop_w)
                drop_y = np.random.randint(0, size - drop_h)
                ch1[drop_y : drop_y + drop_h, drop_x : drop_x + drop_w] = 0.0

        tensor_3ch = np.stack([ch0, ch1, ch2], axis=0)
        tensor_norm = (tensor_3ch - self.mean) / self.std

        return {
            "image": torch.from_numpy(tensor_norm).float(),
            "cvc_label": torch.from_numpy(self.cvc_labels[idx]).float(),
            "aux_label": torch.from_numpy(self.aux_labels[idx]).float(),
            "study_id": sid,
        }


def evaluate_on_val(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluates student model on locked validation cohort (N=2,558)."""
    model.eval()
    all_cvc_preds = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                logits_cvc, _ = model(imgs)
                probs = torch.sigmoid(logits_cvc)
            all_cvc_preds.append(probs.cpu().float().numpy())

    preds = np.concatenate(all_cvc_preds, axis=0)
    y_true = val_df[CLASSES].values.astype(int)

    norm_auc = float(roc_auc_score(y_true[:, 0], preds[:, 0]))
    bord_auc = float(roc_auc_score(y_true[:, 1], preds[:, 1]))
    abno_auc = float(roc_auc_score(y_true[:, 2], preds[:, 2]))
    macro_auc = float(np.mean([norm_auc, bord_auc, abno_auc]))

    is_traced = (val_df["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_df["annotated_cvc_count"] > 1) | (y_true.sum(axis=1) > 1)

    tr_aucs = [roc_auc_score(y_true[is_traced, c], preds[is_traced, c]) for c in range(3)]
    traced_auc = float(np.mean(tr_aucs))

    un_aucs = [roc_auc_score(y_true[is_untraced, c], preds[is_untraced, c]) for c in range(3)]
    untraced_auc = float(np.mean(un_aucs))

    mu_aucs = [roc_auc_score(y_true[is_multicvc, c], preds[is_multicvc, c]) for c in range(3)]
    multicvc_auc = float(np.mean(mu_aucs))

    metrics = {
        "macro_auroc": round(macro_auc, 4),
        "normal_auroc": round(norm_auc, 4),
        "borderline_auroc": round(bord_auc, 4),
        "abnormal_auroc": round(abno_auc, 4),
        "traced_auroc": round(traced_auc, 4),
        "untraced_auroc": round(untraced_auc, 4),
        "multicvc_auroc": round(multicvc_auc, 4),
    }
    return preds, metrics


def train_noisy_student(
    exp_id: str,
    backbone: str,
    warmstart_ckpt: str,
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr_head: float = 1.5e-4,
    lr_backbone: float = 4e-5,
):
    print("=" * 80)
    print(f"FULL-COHORT NOISY-STUDENT DISTILLATION: {exp_id}")
    print(f"Backbone: {backbone} | Epochs: {epochs} | Batch: {batch_size} (x{grad_accum_steps} accum)")
    print(f"Warmstart: {warmstart_ckpt}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS_DIR / exp_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    VAL_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Manifest & Caches
    print("Loading manifests and caches...", flush=True)
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)

    # Fast directory lookup via os.listdir
    consensus_files = set(f[:-4] for f in os.listdir(CONSENSUS_SEG_DIR) if f.endswith(".png"))
    general_files = set(f[:-4] for f in os.listdir(GENERAL_SEG_DIR) if f.endswith(".png"))
    all_cached_sids = consensus_files.union(general_files)

    train_manifest = train_manifest[train_manifest["StudyInstanceUID"].isin(all_cached_sids)].reset_index(drop=True)
    print(f"Active Distillation Cohort: Train = {len(train_manifest):,} studies | Val = {len(val_manifest):,} studies", flush=True)

    # Lookups
    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    c_tr_col = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_tr.columns else "study_id"
    carina_lookup_tr = df_carina_tr.set_index(c_tr_col).to_dict(orient="index")
    c_val_col = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_val.columns else "study_id"
    carina_lookup_val = df_carina_val.set_index(c_val_col).to_dict(orient="index")

    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)
    z_tr_col = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_tr.columns else "study_id"
    zonal_lookup_tr = df_zonal_tr.set_index(z_tr_col).to_dict(orient="index")
    z_val_col = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_val.columns else "study_id"
    zonal_lookup_val = df_zonal_val.set_index(z_val_col).to_dict(orient="index")

    train_ds = FullCohortNoisyStudentDataset(
        df=train_manifest,
        carina_lookup=carina_lookup_tr,
        zonal_lookup=zonal_lookup_tr,
        img_size=512,
        is_train=True,
        ribbon_dropout_prob=0.20,
    )
    val_ds = FullCohortNoisyStudentDataset(
        df=val_manifest,
        carina_lookup=carina_lookup_val,
        zonal_lookup=zonal_lookup_val,
        img_size=512,
        is_train=False,
        ribbon_dropout_prob=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 2. Build Model & Warmstart
    model = MaskGuidedSpatialNet(
        backbone_name=backbone,
        pretrained=False,
        num_classes=3,
        num_aux=8,
    ).to(device)

    best_macro = 0.0
    best_preds = None
    start_time = time.time()

    # Check if a recent best.pt already exists in the experiment's own ckpt_dir
    resume_ckpt = ckpt_dir / "best.pt"
    active_ckpt = str(resume_ckpt) if resume_ckpt.exists() else warmstart_ckpt

    if active_ckpt and Path(active_ckpt).exists():
        print(f"Warm-starting trunk from: {active_ckpt}")
        ckpt = torch.load(active_ckpt, map_location=device, weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        if "val_metrics" in ckpt and "macro_auroc" in ckpt["val_metrics"]:
            best_macro = ckpt["val_metrics"]["macro_auroc"]
            print(f"Found existing checkpoint with best_macro = {best_macro:.4f}")
        model_dict = model.state_dict()
        pretrained_dict = {
            k: (v.float() if v.is_floating_point() else v)
            for k, v in st.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict)} matching tensors into backbone!")

        # Pre-evaluate to guarantee best_preds is populated
        if best_macro > 0:
            print("Pre-evaluating resumed weights on locked validation set...")
            best_preds, val_metrics = evaluate_on_val(model, val_loader, device, val_manifest)
            print(f"Resumed validation baseline: {val_metrics['macro_auroc']:.4f}")

    # 3. Optimizer & Criterion
    head_params = [p for n, p in model.named_parameters() if "pool" in n or "fc" in n]
    backbone_params = [p for n, p in model.named_parameters() if not ("pool" in n or "fc" in n)]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    scaler = GradScaler("cuda")
    bce_loss = nn.BCEWithLogitsLoss()

    print("\nBeginning Training Loop across all 21,855 studies...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_cvc = 0.0
        running_aux = 0.0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            imgs = batch["image"].to(device, non_blocking=True)
            cvc_labels = batch["cvc_label"].to(device, non_blocking=True)
            aux_labels = batch["aux_label"].to(device, non_blocking=True)

            # Mixup Augmentation (40% probability)
            if np.random.rand() < 0.40:
                lam = np.random.beta(0.2, 0.2)
                perm = torch.randperm(imgs.size(0))
                imgs = lam * imgs + (1.0 - lam) * imgs[perm]
                cvc_labels = lam * cvc_labels + (1.0 - lam) * cvc_labels[perm]
                aux_labels = lam * aux_labels + (1.0 - lam) * aux_labels[perm]

            with autocast("cuda", dtype=torch.float16):
                logits_cvc, logits_aux = model(imgs)
                l_cvc = bce_loss(logits_cvc, cvc_labels)
                l_aux = bce_loss(logits_aux, aux_labels)
                loss = (l_cvc + 0.25 * l_aux) / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * grad_accum_steps
            running_cvc += l_cvc.item()
            running_aux += l_aux.item()

            if step % 300 == 0 or step == len(train_loader):
                elapsed = time.time() - t0
                speed = (step * batch_size) / max(elapsed, 1.0)
                print(
                    f"  [Epoch {epoch}/{epochs}] Step {step}/{len(train_loader)} | "
                    f"L_tot: {running_loss/step:.4f} (CVC: {running_cvc/step:.4f}, Aux: {running_aux/step:.4f}) | "
                    f"{speed:.1f} img/s"
                )

        # Epoch Validation
        print(f"\nEvaluating Epoch {epoch} on locked N=2,558 validation cohort...")
        val_preds, val_metrics = evaluate_on_val(model, val_loader, device, val_manifest)
        print(
            f"Epoch {epoch} Results -> Macro: {val_metrics['macro_auroc']} | "
            f"Norm: {val_metrics['normal_auroc']} | Bord: {val_metrics['borderline_auroc']} | "
            f"Abno: {val_metrics['abnormal_auroc']} | MultiCVC: {val_metrics['multicvc_auroc']}"
        )

        if val_metrics["macro_auroc"] > best_macro:
            best_macro = val_metrics["macro_auroc"]
            best_preds = val_preds
            best_ckpt_path = ckpt_dir / "best.pt"
            fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": fp16_state, "val_metrics": val_metrics, "epoch": epoch}, str(best_ckpt_path))
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Macro AUROC: {best_macro})")

    # Save final best validation predictions
    val_pred_path = VAL_PREDS_DIR / f"val_preds_{exp_id}.npy"
    np.save(str(val_pred_path), best_preds)
    print(f"\nDistillation complete for {exp_id} in {time.time()-start_time:.1f}s. Best Macro AUROC: {best_macro}")
    print(f"Saved validation predictions to: {val_pred_path}")

    # Append to Master Ablation Ledger
    if LEDGER_CSV.exists():
        df_ledger = pd.read_csv(LEDGER_CSV)
        new_row = {
            "exp_id": exp_id,
            "stage": 4,
            "description": f"Noisy-Student Distillation on all 21,855 studies ({backbone})",
            "val_macro_auroc": best_macro,
            "Normal_AUROC": val_metrics["normal_auroc"],
            "Borderline_AUROC": val_metrics["borderline_auroc"],
            "Abnormal_AUROC": val_metrics["abnormal_auroc"],
            "traced_auroc": val_metrics["traced_auroc"],
            "untraced_auroc": val_metrics["untraced_auroc"],
            "multicvc_auroc": val_metrics["multicvc_auroc"],
            "status": "Completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        df_ledger = pd.concat([df_ledger, pd.DataFrame([new_row])], ignore_index=True)
        df_ledger.to_csv(LEDGER_CSV, index=False)
        df_ledger.to_parquet(LEDGER_PARQUET, index=False)
        print(f"Logged {exp_id} to Master Ablation Ledger.")


def main():
    parser = argparse.ArgumentParser(description="Full-Cohort Noisy-Student Distillation (E96/E97)")
    parser.add_argument("--model", type=str, choices=["e96", "e97", "both"], default="both", help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
    args = parser.parse_args()

    if args.model in ["e96", "both"]:
        train_noisy_student(
            exp_id="E96_noisy_student_densenet121_512",
            backbone="densenet121",
            warmstart_ckpt=str(RUNS_DIR / "E89_mgap_densenet121_512" / "checkpoints" / "best.pt"),
            epochs=args.epochs,
            batch_size=8,
            grad_accum_steps=2,
            lr_head=1.5e-4,
            lr_backbone=4e-5,
        )

    if args.model in ["e97", "both"]:
        train_noisy_student(
            exp_id="E97_noisy_student_seresnext50_512",
            backbone="seresnext50_32x4d",
            warmstart_ckpt=str(RUNS_DIR / "E90_mgap_seresnext50_512" / "checkpoints" / "best.pt"),
            epochs=args.epochs,
            batch_size=8,
            grad_accum_steps=2,
            lr_head=1.5e-4,
            lr_backbone=4e-5,
        )


if __name__ == "__main__":
    main()
