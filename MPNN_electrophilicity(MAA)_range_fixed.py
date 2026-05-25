#!/usr/bin/env python
# coding: utf-8
"""
Site-aware GNN / MPNN for ESNUEL electrophilicity prediction.
VERSION 4 — molecule-level split to eliminate data leakage.

=======================================================================
PROBLEM FIXED vs v3:
=======================================================================
v3 (and the Set-fold path) split at the **row** level.
Because each molecule appears once per electrophilic site, the same
SMILES could end up in train AND test simultaneously.
The model therefore "saw" the molecule during training and test was
not truly held-out → inflated / over-optimistic MAE/RMSE/R².

v4 fixes this with **molecule-level** splitting:
  1. Canonicalize every SMILES with RDKit.
  2. Collect the set of unique canonical SMILES.
  3. Shuffle and partition at the SMILES level (80/10/10 default).
  4. Assign ALL rows of a molecule to the same split.

Verification is built-in: the script asserts zero SMILES overlap
between train/val/test before building graphs, and reports it.

Optional scaffold-based split (--split_strategy scaffold):
  Groups molecules by Bemis-Murcko scaffold so chemically similar
  molecules stay together in the same fold — even stricter evaluation.

=======================================================================
Task:
    molecule graph + highlighted electrophilic atom/site -> MAA for that site

Expected columns in df_elec.parquet:
    - smiles
    - elec_sites
    - MAA_values
    - elec_names        (optional)
    - elec_GCS_3_cm5    (optional)
    - Set               (optional, used only for reference / logging)

Recommended Colab run:
    !pip install rdkit torch-geometric -q
    !python gnn_esnuel_siteaware_v4.py --data_path df_elec.parquet --epochs 50 --batch_size 128

Outputs:
    outputs_gnn_siteaware_v4/
        summary_metrics.csv
        training_history.csv
        test_predictions.csv
        worst_outliers.csv
        split_summary.csv        ← NEW: molecule/row counts per split
        parity_plot.png
        loss_curve.png
        best_gnn_model.pt
"""

import os
import math
import argparse
import random
from pathlib import Path
import urllib.request
import tarfile
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Chemistry featurization
# -----------------------------
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices]


def atom_features(atom: Chem.Atom, is_target_site: bool) -> list:
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 5.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(is_target_site),
    ] + one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)


def bond_features(bond: Chem.Bond) -> list:
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
    )


ATOM_FDIM = 7 + len(HYBRIDIZATION_TYPES)
BOND_FDIM = len(BOND_TYPES) + 2


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: str):
    """Return canonical SMILES or None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_murcko_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES, or the molecule itself if acyclic."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "__invalid__"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold, canonical=True)
        return smi if smi else "__no_scaffold__"
    except Exception:
        return "__scaffold_error__"


def parse_site(site):
    try:
        if isinstance(site, str):
            site = site.strip()
            if site == "":
                return None
            site = float(site)
        return int(site)
    except Exception:
        return None


def smiles_site_to_graph(smiles: str, site_idx: int, y_value: float, row_idx: int):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if site_idx is None or site_idx < 0 or site_idx >= n_atoms:
        return None

    x = torch.tensor(
        [atom_features(atom, atom.GetIdx() == site_idx) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.append([i, j])
        edge_attrs.append(bf)
        edge_indices.append([j, i])
        edge_attrs.append(bf)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y_value)], dtype=torch.float),
    )

    data.smiles = smiles
    data.row_idx = int(row_idx)
    data.elec_site = int(site_idx)
    return data


# -----------------------------
# Molecule-level splitting  ← KEY FIX
# -----------------------------

def make_molecule_split(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Split at the molecule (canonical SMILES) level so that no molecule
    appears in more than one of {train, val, test}.

    Strategy options:
      'random'    — shuffle unique canonical SMILES, then partition 80/10/10
      'scaffold'  — group by Bemis-Murcko scaffold, distribute scaffolds
                    across splits so each scaffold is wholly in one split.
                    This is stricter: tests generalization to new scaffolds.
      'maa_range' — split molecules by their maximum site MAA:
                    train = lower/moderate max-MAA molecules,
                    val   = high max-MAA molecules,
                    test  = strongest max-MAA molecules.
                    This is an extrapolation stress test for strong electrophiles.

    After assigning each canonical SMILES to a split, ALL rows for that
    SMILES inherit that split label.

    Returns df with a 'split' column added.
    """
    df = df.copy()

    # Canonicalize in-place; drop rows where RDKit can't parse the SMILES
    df["canon_smiles"] = df["smiles"].apply(canonicalize_smiles)
    n_invalid = df["canon_smiles"].isna().sum()
    if n_invalid:
        print(f"  Dropping {n_invalid} rows with un-parseable SMILES.")
    df = df.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    unique_mols = df["canon_smiles"].unique()
    n_mols = len(unique_mols)
    print(f"  Unique molecules (canonical SMILES): {n_mols}")
    print(f"  Total rows (sites): {len(df)}")
    print(f"  Average sites per molecule: {len(df)/n_mols:.2f}")

    rng = np.random.default_rng(args.seed)

    if args.split_strategy == "scaffold":
        return _scaffold_split(df, unique_mols, args, rng)
    elif args.split_strategy == "maa_range":
        return _maa_range_split(df, unique_mols, args, rng)
    else:
        return _random_mol_split(df, unique_mols, args, rng)


def _random_mol_split(df, unique_mols, args, rng):
    """Randomly assign molecules to splits."""
    perm = rng.permutation(len(unique_mols))
    n = len(unique_mols)
    train_end = int(args.train_frac * n)
    val_end = int((args.train_frac + args.val_frac) * n)

    train_mols = set(unique_mols[perm[:train_end]])
    val_mols   = set(unique_mols[perm[train_end:val_end]])
    test_mols  = set(unique_mols[perm[val_end:]])

    def assign(smi):
        if smi in train_mols:
            return "train"
        elif smi in val_mols:
            return "val"
        else:
            return "test"

    df["split"] = df["canon_smiles"].apply(assign)
    return df


def _scaffold_split(df, unique_mols, args, rng):
    """
    Bemis-Murcko scaffold split.

    1. Compute scaffold for each unique molecule.
    2. Group molecules by scaffold.
    3. Sort scaffolds by size descending (largest first for better balance).
    4. Greedily assign scaffolds to splits, keeping totals near target ratios.

    This ensures an entire scaffold family is in exactly one split,
    giving the strictest possible evaluation of generalization.
    """
    print("  Computing Murcko scaffolds (may take a moment)...")

    mol_to_scaffold = {smi: get_murcko_scaffold(smi) for smi in unique_mols}
    scaffold_to_mols: dict[str, list] = {}
    for smi, scaf in mol_to_scaffold.items():
        scaffold_to_mols.setdefault(scaf, []).append(smi)

    # Sort scaffolds: largest first for better balance
    scaffolds_sorted = sorted(scaffold_to_mols.keys(),
                              key=lambda s: len(scaffold_to_mols[s]),
                              reverse=True)
    rng.shuffle(scaffolds_sorted)  # break ties randomly

    n_total = len(unique_mols)
    train_target = int(args.train_frac * n_total)
    val_target   = int(args.val_frac   * n_total)

    train_mols, val_mols, test_mols = set(), set(), set()
    n_train, n_val = 0, 0

    for scaf in scaffolds_sorted:
        mols = scaffold_to_mols[scaf]
        if n_train < train_target:
            train_mols.update(mols)
            n_train += len(mols)
        elif n_val < val_target:
            val_mols.update(mols)
            n_val += len(mols)
        else:
            test_mols.update(mols)

    print(f"  Scaffold split: {len(train_mols)} train / {len(val_mols)} val / {len(test_mols)} test molecules")

    def assign(smi):
        if smi in train_mols:
            return "train"
        elif smi in val_mols:
            return "val"
        else:
            return "test"

    df["split"] = df["canon_smiles"].apply(assign)
    return df



def _maa_range_split(df, unique_mols, args, rng):
    """
    Molecule-level MAA-range split.

    Goal:
        Test whether the model can extrapolate toward strong electrophiles.

    Method:
        1. For each molecule, compute max(MAA_values) across all its sites.
        2. Sort molecules by this max site MAA.
        3. Assign:
              train = molecules below maa_val_quantile
              val   = molecules between maa_val_quantile and maa_test_quantile
              test  = molecules above maa_test_quantile

    Default:
        maa_val_quantile = 0.80
        maa_test_quantile = 0.90

    This means:
        train sees lower/moderate-electrophilicity molecules,
        validation checks high-electrophilicity molecules,
        test checks the strongest electrophile molecules.

    It is intentionally harder than random/scaffold splits.
    """
    mol_max = (
        df.groupby("canon_smiles")["MAA_values"]
        .max()
        .sort_values()
    )

    q_val = float(args.maa_val_quantile)
    q_test = float(args.maa_test_quantile)

    if not (0.0 < q_val < q_test < 1.0):
        raise ValueError("Require 0 < maa_val_quantile < maa_test_quantile < 1.")

    val_threshold = mol_max.quantile(q_val)
    test_threshold = mol_max.quantile(q_test)

    train_mols = set(mol_max[mol_max < val_threshold].index)
    val_mols = set(mol_max[(mol_max >= val_threshold) & (mol_max < test_threshold)].index)
    test_mols = set(mol_max[mol_max >= test_threshold].index)

    # In case of ties around thresholds, some bins can be slightly imbalanced.
    # This is acceptable and chemically more transparent than forcing random reassignment.
    print("  MAA-range split based on molecule-level max site MAA.")
    print(f"  Molecule max-MAA val quantile : {q_val:.2f}")
    print(f"  Molecule max-MAA test quantile: {q_test:.2f}")
    print(f"  Validation threshold: max MAA >= {val_threshold:.3f}")
    print(f"  Test threshold      : max MAA >= {test_threshold:.3f}")
    print(f"  Split: {len(train_mols)} train / {len(val_mols)} val / {len(test_mols)} test molecules")
    print("  Interpretation: train = lower/moderate molecules, val = high molecules, test = strongest molecules.")

    def assign(smi):
        if smi in train_mols:
            return "train"
        elif smi in val_mols:
            return "val"
        else:
            return "test"

    df["split"] = df["canon_smiles"].apply(assign)
    return df


def verify_no_smiles_overlap(df: pd.DataFrame) -> None:
    """
    Hard assertion: no canonical SMILES appears in more than one split.
    Raises if there is any overlap (should never happen after mol-level split).
    """
    splits = ["train", "val", "test"]
    mol_sets = {s: set(df.loc[df["split"] == s, "canon_smiles"]) for s in splits}

    train_val = mol_sets["train"] & mol_sets["val"]
    train_test = mol_sets["train"] & mol_sets["test"]
    val_test   = mol_sets["val"]   & mol_sets["test"]

    if train_val or train_test or val_test:
        raise RuntimeError(
            f"SMILES overlap detected after split!\n"
            f"  train∩val : {len(train_val)}\n"
            f"  train∩test: {len(train_test)}\n"
            f"  val∩test  : {len(val_test)}\n"
            "This should not happen — please report a bug."
        )
    print("  ✓ No SMILES overlap between train / val / test.")


# -----------------------------
# Data loading / cleaning
# -----------------------------
def load_site_dataframe(args):
    path = args.data_path
    # --- Auto-download if file is missing ---
    url          = "https://sid.erda.dk/share_redirect/c7LF5NaYvH"
    archive_name = "data.tar.xz"
    if not os.path.exists(path):
        if not os.path.exists(archive_name):
            print("Downloading archive (~1.1 GB)...")
            urllib.request.urlretrieve(url, archive_name)
            print("Download complete.")
        print("Extracting archive...")
        with tarfile.open(archive_name, "r:xz") as tar:
            tar.extractall()
        print("Extraction complete.")
    # -----------------------------------------
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv") or path.endswith(".csv.gz"):
        df = pd.read_csv(path)
    else:
        raise ValueError("data_path must be .parquet, .csv, or .csv.gz")

    required = {"smiles", "elec_sites", "MAA_values"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                 errors="ignore").copy()

    df["MAA_values"] = pd.to_numeric(df["MAA_values"], errors="coerce")
    df["elec_site_int"] = df["elec_sites"].apply(parse_site)

    df = df.dropna(subset=["smiles", "MAA_values", "elec_site_int"])
    df["elec_site_int"] = df["elec_site_int"].astype(int)

    n0 = len(df)
    df = df[df["MAA_values"].between(args.min_maa, args.max_maa)].copy()

    if args.max_rows is not None and len(df) > args.max_rows:
        # Sample at the MOLECULE level to preserve structure
        unique_smiles = df["smiles"].unique()
        rng = np.random.default_rng(args.seed)
        sampled_smiles = rng.choice(unique_smiles,
                                    size=min(args.max_rows, len(unique_smiles)),
                                    replace=False)
        df = df[df["smiles"].isin(sampled_smiles)].copy()
        print(f"  Sampled to {len(sampled_smiles)} molecules ({len(df)} rows).")

    df = df.reset_index(drop=True)

    print(f"\nLoaded rows before MAA filter: {n0}")
    print(f"Rows after MAA filter [{args.min_maa}, {args.max_maa}]: {len(df)}")
    print(f"MAA mean={df['MAA_values'].mean():.3f}, std={df['MAA_values'].std():.3f}, "
          f"min={df['MAA_values'].min():.3f}, max={df['MAA_values'].max():.3f}")

    if "Set" in df.columns:
        print(f"\nOriginal 'Set' column distribution (for reference):")
        print(df["Set"].value_counts().to_string())

    print(f"\nApplying molecule-level split (strategy: {args.split_strategy})...")
    df = make_molecule_split(df, args)

    # Strict verification — zero-overlap guaranteed
    verify_no_smiles_overlap(df)

    print("\nSplit row counts:")
    print(df["split"].value_counts().to_string())
    print("\nSplit molecule counts:")
    mol_counts = df.groupby("split")["canon_smiles"].nunique()
    print(mol_counts.to_string())

    return df.reset_index(drop=True)


def build_graphs(df, y_mean, y_std):
    graphs = []
    skipped = 0

    for idx, row in df.iterrows():
        y_norm = (float(row["MAA_values"]) - y_mean) / y_std
        g = smiles_site_to_graph(
            smiles=row["smiles"],
            site_idx=int(row["elec_site_int"]),
            y_value=y_norm,
            row_idx=idx,
        )
        if g is None:
            skipped += 1
        else:
            graphs.append(g)

    if skipped:
        print(f"Skipped {skipped} rows due to invalid SMILES/site indices.")
    return graphs


# -----------------------------
# Model (unchanged from v3)
# -----------------------------
class SiteAwareMPNN(nn.Module):
    """
    Site-aware MPNN with GINEConv + residual connections + multi-pooling.
    Architecture unchanged from v3; the split is the only change.
    """
    def __init__(self, atom_fdim, bond_fdim, hidden_dim=192, num_layers=5, dropout=0.10):
        super().__init__()

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_fdim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=bond_fdim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        pooled_dim = hidden_dim * 3

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        pooled = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ],
            dim=1,
        )

        return self.head(pooled).view(-1)


# -----------------------------
# Training / evaluation
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(batch)
        target = batch.y.view(-1)

        loss = F.smooth_l1_loss(pred, target)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        n = batch.num_graphs
        total_loss += loss.item() * n
        total_graphs += n

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std):
    model.eval()

    preds_norm, targets_norm = [], []
    smiles_list, row_indices, elec_sites = [], [], []

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch).detach().cpu().numpy()
        target = batch.y.view(-1).detach().cpu().numpy()

        preds_norm.append(pred)
        targets_norm.append(target)

        smiles_list.extend(batch.smiles)
        row_indices.extend([int(x) for x in batch.row_idx])
        elec_sites.extend([int(x) for x in batch.elec_site])

    preds_norm = np.concatenate(preds_norm)
    targets_norm = np.concatenate(targets_norm)

    preds = preds_norm * y_std + y_mean
    targets = targets_norm * y_std + y_mean

    rmse = math.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    pred_df = pd.DataFrame({
        "row_idx": row_indices,
        "smiles": smiles_list,
        "elec_site": elec_sites,
        "MAA_true": targets,
        "MAA_pred": preds,
        "abs_error": np.abs(preds - targets),
        "signed_error": preds - targets,
    })

    return {"rmse": rmse, "mae": mae, "r2": r2}, pred_df


def plot_parity(pred_df, metrics, out_path, split_strategy):
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["MAA_true"], pred_df["MAA_pred"], s=8, alpha=0.45)

    min_val = min(pred_df["MAA_true"].min(), pred_df["MAA_pred"].min())
    max_val = max(pred_df["MAA_true"].max(), pred_df["MAA_pred"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", linewidth=1)
    plt.xlabel("True site MAA (kcal/mol)")
    plt.ylabel("Predicted site MAA (kcal/mol)")
    plt.title(
        f"Site-aware GNN v4 [{split_strategy} mol-split]\n"
        f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss(history_df, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss (SmoothL1, normalized)")
    plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Site-aware GNN v4 training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(args):
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    df = load_site_dataframe(args)

    # Save split summary for traceability
    split_summary = df.groupby("split").agg(
        n_rows=("MAA_values", "count"),
        n_molecules=("canon_smiles", "nunique"),
        maa_mean=("MAA_values", "mean"),
        maa_std=("MAA_values", "std"),
    )
    split_summary.to_csv(out_dir / "split_summary.csv")
    print("\nSplit summary:")
    print(split_summary.to_string())

    train_df = df[df["split"].eq("train")].reset_index(drop=True)
    val_df   = df[df["split"].eq("val")].reset_index(drop=True)
    test_df  = df[df["split"].eq("test")].reset_index(drop=True)

    # Normalise using train statistics only
    y_mean = train_df["MAA_values"].mean()
    y_std  = train_df["MAA_values"].std()
    print(f"\nTrain target mean={y_mean:.3f}, std={y_std:.3f}")

    print("Building molecular graphs...")
    train_graphs = build_graphs(train_df, y_mean, y_std)
    val_graphs   = build_graphs(val_df,   y_mean, y_std)
    test_graphs  = build_graphs(test_df,  y_mean, y_std)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SiteAwareMPNN(
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    history = []

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_metrics, _ = evaluate(model, val_loader, device, y_mean, y_std)
        scheduler.step(val_metrics["rmse"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "y_mean": y_mean,
                "y_std": y_std,
                "atom_fdim": ATOM_FDIM,
                "bond_fdim": BOND_FDIM,
            }, out_dir / "best_gnn_model.pt")

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train loss={train_loss:.4f} | "
                f"val RMSE={val_metrics['rmse']:.2f} | val R²={val_metrics['r2']:.3f}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    print("Loading best model and evaluating on held-out test set...")
    checkpoint = torch.load(out_dir / "best_gnn_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_pred_df = evaluate(model, test_loader, device, y_mean, y_std)
    test_pred_df = test_pred_df.sort_values("row_idx").reset_index(drop=True)

    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    test_pred_df.sort_values("abs_error", ascending=False).head(args.n_outliers).to_csv(
        out_dir / "worst_outliers.csv", index=False
    )

    plot_parity(test_pred_df, test_metrics, out_dir / "parity_plot.png", args.split_strategy)
    plot_loss(history_df, out_dir / "loss_curve.png")

    summary = {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "n_train_graphs": len(train_graphs),
        "n_val_graphs": len(val_graphs),
        "n_test_graphs": len(test_graphs),
        "n_train_mols": train_df["canon_smiles"].nunique(),
        "n_val_mols": val_df["canon_smiles"].nunique(),
        "n_test_mols": test_df["canon_smiles"].nunique(),
        "y_mean_train": y_mean,
        "y_std_train": y_std,
        "split_strategy": args.split_strategy,
        "maa_val_quantile": getattr(args, "maa_val_quantile", None),
        "maa_test_quantile": getattr(args, "maa_test_quantile", None),
        "min_maa": args.min_maa,
        "max_maa": args.max_maa,
    }

    pd.Series(summary).to_csv(out_dir / "summary_metrics.csv")

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS  (molecule-level split — no leakage)")
    print(f"  Split strategy : {args.split_strategy}")
    print(f"  RMSE : {test_metrics['rmse']:.3f}")
    print(f"  MAE  : {test_metrics['mae']:.3f}")
    print(f"  R²   : {test_metrics['r2']:.4f}")
    print("=" * 60)
    print(f"Results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Site-aware GNN v5 — molecule-level split incl. MAA-range split on df_elec.parquet."
    )

    parser.add_argument("--data_path", type=str, default="df_elec.parquet")
    parser.add_argument("--out_dir",   type=str, default="outputs_gnn_siteaware_v4")

    # ---- split ----
    parser.add_argument(
        "--split_strategy", type=str, default="random",
        choices=["random", "scaffold", "maa_range"],
        help=(
            "How to assign molecules to splits.\n"
            "  random    : shuffle unique canonical SMILES, then 80/10/10.\n"
            "  scaffold  : Bemis-Murcko scaffold split — stricter, tests\n"
            "              generalization to unseen chemical scaffolds.\n"
            "  maa_range : molecule-level split by max site MAA; train on\n"
            "              lower/moderate molecules, test on strongest electrophiles."
        ),
    )
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac",   type=float, default=0.10)
    # test_frac is implicitly 1 - train_frac - val_frac for random/scaffold splits

    # For split_strategy="maa_range":
    # train = molecules with max site MAA below maa_val_quantile
    # val   = molecules between maa_val_quantile and maa_test_quantile
    # test  = molecules above maa_test_quantile
    parser.add_argument("--maa_val_quantile",  type=float, default=0.80)
    parser.add_argument("--maa_test_quantile", type=float, default=0.90)

    # ---- training ----
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=15)
    parser.add_argument("--print_every", type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--hidden_dim",  type=int,   default=192)
    parser.add_argument("--num_layers",  type=int,   default=5)
    parser.add_argument("--dropout",     type=float, default=0.10)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-5)
    parser.add_argument("--grad_clip",   type=float, default=5.0)

    # ---- data ----
    parser.add_argument("--min_maa",  type=float, default=0.0)
    parser.add_argument("--max_maa",  type=float, default=400.0)
    parser.add_argument("--max_rows", type=int,   default=120000,
                        help="Max rows (sampled at molecule level). -1 = all.")

    # ---- misc ----
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_outliers",  type=int, default=50)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--cpu",         action="store_true")

    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 0:
        args.max_rows = None

    main(args)
