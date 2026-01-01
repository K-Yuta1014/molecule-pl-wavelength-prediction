# pl_gui.py
import os
import json
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

import tkinter as tk
from tkinter import ttk, messagebox

# RDKit
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Mordred
from mordred import Calculator, descriptors as mordred_descriptors
from rdkit.Chem import AllChem

# PIL (for Tk image)
from PIL import ImageTk

# sklearn utilities for xgb fallback
from sklearn.compose import TransformedTargetRegressor

# torch for NN
import torch
import torch.nn as nn


# =========================
# Config
# =========================

DESCRIPTOR_TYPES = ["rdkit", "mordred_2d", "mordred_3d"]

# preprocess rule candidates per descriptor_type
RULE_CANDIDATES = {
    "rdkit": [
        "outputs/descriptors/rdkit/preprocess_rule_rdkit.json",
        "outputs/descriptors/rdkit/preprocess_rule.json",
        "outputs/descriptors/preprocess_rule_rdkit.json",
        "outputs/descriptors/preprocess_rule.json",
    ],
    "mordred_2d": [
        "outputs/descriptors/mordred_2d/preprocess_rule.json",
    ],
    "mordred_3d": [
        "outputs/descriptors/mordred_3d/preprocess_rule.json",
    ],
}

# model artifact paths
MODEL_ARTIFACT_PATH = {
    "pls": "models/pls/artifact_pls_{dtype}.joblib",
    "rf":  "models/rf/artifact_rf_{dtype}.joblib",
    "xgb": "models/xgb/artifact_xgb_{dtype}.joblib",
    "lgb": "models/lgb/artifact_lgb_{dtype}.joblib",
    "nn":  "models/nn/artifact_nn_{dtype}.joblib",
}

# AD artifact paths
AD_ARTIFACT_PATH = {
    "ocsvm": "models/ad/artifact_ocsvm_{dtype}.joblib",
    "knn":   "models/ad/artifact_knn_{dtype}.joblib",
    "iso":   "models/ad/artifact_iso_{dtype}.joblib",
}


# =========================
# Utilities (loaders / preprocess)
# =========================

def _load_json_any(paths) -> dict:
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"preprocess rule not found. tried: {paths}")

def _load_joblib(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def transform_with_rule(X_raw: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    Apply saved preprocessing rule to X_raw.
    Expected rule keys:
      - feature_columns (list)
      - impute_means (dict)
    """
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError("X_raw must be a DataFrame")

    X = X_raw.copy()

    # coerce to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # inf -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # align columns to training feature space
    feat_cols = rule["feature_columns"]
    X = X.reindex(columns=feat_cols)

    # impute by training means
    means = pd.Series(rule.get("impute_means", {}), dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(means)

    # final safety: if still NaN, fill 0
    X = X.fillna(0.0)

    return X


# =========================
# Descriptor computation
# =========================

# RDKit descriptor calculator cache
_RDKit_DESCRIPTOR_NAMES = [name for (name, _) in Descriptors.descList]
_RDKit_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_RDKit_DESCRIPTOR_NAMES)

def compute_rdkit_descriptors_one(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES (RDKit MolFromSmiles returned None).")
    vals = _RDKit_CALC.CalcDescriptors(mol)
    df = pd.DataFrame([vals], columns=_RDKit_DESCRIPTOR_NAMES)
    return df

def compute_mordred_descriptors_one(smiles: str, mode: str) -> pd.DataFrame:
    """
    mode: 'mordred_2d' or 'mordred_3d'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES (RDKit MolFromSmiles returned None).")

    if mode == "mordred_2d":
        mols = [mol]
        calc = Calculator(mordred_descriptors, ignore_3D=True)
        df = calc.pandas(mols)

    elif mode == "mordred_3d":
        # build 3D mol (H-added + embed)
        molh = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(molh, AllChem.ETKDG())
        if ret != 0:
            raise ValueError("3D embedding failed (EmbedMolecule returned non-zero).")
        molhs = [molh]
        calc = Calculator(mordred_descriptors, ignore_3D=False)
        df = calc.pandas(molhs)

    else:
        raise ValueError(mode)

    # mordred sometimes yields object dtype; coerce to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =========================
# Model prediction (sklearn + NN)
# =========================

def predict_sklearn_artifact_safe(artifact: dict, X: pd.DataFrame, model_kind: str) -> float:
    """
    Standard route: artifact["model"].predict(X)
    Fallback route: if AttributeError due to sklearn/xgboost tags issue,
      manually run scaler->estimator->inverse_transform for TransformedTargetRegressor.
    """
    model = artifact["model"]

    try:
        y = model.predict(X)
        return float(np.asarray(y).ravel()[0])

    except AttributeError:
        # Fallback mainly for xgb in some envs
        if isinstance(model, TransformedTargetRegressor):
            # regressor_ should exist after fit
            x_pipe = model.regressor_
            if hasattr(x_pipe, "named_steps") and "x_scaler" in x_pipe.named_steps:
                x_scaler = x_pipe.named_steps["x_scaler"]
                X_scaled = x_scaler.transform(X)

                # final estimator: try known names first, else last step
                if model_kind in x_pipe.named_steps:
                    est = x_pipe.named_steps[model_kind]
                else:
                    est = x_pipe.steps[-1][1]

                y_scaled = est.predict(X_scaled)
                y_pred = model.transformer_.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()[0]
                return float(y_pred)

        raise  # unknown case


def define_model_from_params(params: dict, input_dim: int) -> nn.Module:
    """
    NN architecture must match training.
    Expected keys in params: n_layers, hidden_dim, dropout_rate, activation, use_batchnorm
    """
    layers = []

    n_layers = int(params["n_layers"])
    hidden_dim = int(params["hidden_dim"])
    dropout_rate = float(params["dropout_rate"])
    activation_name = params["activation"]  # "relu" or "leaky_relu"
    use_batchnorm = bool(params["use_batchnorm"])

    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if activation_name == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def predict_nn_artifact(artifact_nn: dict, X: pd.DataFrame, device: str = "cpu") -> float:
    cols = artifact_nn["feature_columns"]
    X_aligned = X.reindex(columns=cols)

    x_scaler = artifact_nn["x_scaler"]
    y_scaler = artifact_nn["y_scaler"]
    params = artifact_nn["hparams"]
    input_dim = int(artifact_nn["input_dim"])

    X_scaled = x_scaler.transform(X_aligned)

    model = define_model_from_params(params, input_dim=input_dim)
    state = torch.load(artifact_nn["weights_path"], map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        y_scaled = model(x_t).detach().cpu().numpy().reshape(-1, 1)

    y_pred = y_scaler.inverse_transform(y_scaled).ravel()[0]
    return float(y_pred)


# =========================
# AD evaluation
# =========================

@dataclass
class ADResult:
    ocsvm_density: float
    in_ad_ocsvm: bool
    knn_dist: float
    in_ad_knn: bool
    iso_score: float
    in_ad_iso: bool

    @property
    def in_all_ads(self) -> bool:
        return bool(self.in_ad_ocsvm and self.in_ad_knn and self.in_ad_iso)


def eval_ads(dtype: str, X_processed: pd.DataFrame) -> ADResult:
    # Load artifacts
    art_ocsvm = _load_joblib(AD_ARTIFACT_PATH["ocsvm"].format(dtype=dtype))
    art_knn   = _load_joblib(AD_ARTIFACT_PATH["knn"].format(dtype=dtype))
    art_iso   = _load_joblib(AD_ARTIFACT_PATH["iso"].format(dtype=dtype))

    # Align to AD feature space
    cols = art_ocsvm["feature_columns"]
    X_ad = X_processed.reindex(columns=cols)

    # Scale by AD scaler (train-fit)
    scaler = art_ocsvm["scaler"]  # same scaler design you used for AD
    X_scaled = scaler.transform(X_ad)

    # OCSVM
    ocsvm = art_ocsvm["model"]
    density = float(ocsvm.decision_function(X_scaled).ravel()[0])
    thr_ocsvm = float(art_ocsvm["threshold"])
    in_ocsvm = bool(density >= thr_ocsvm)

    # kNN (NearestNeighbors)
    knn = art_knn["model"]
    k = int(art_knn["k"])
    dist, _ = knn.kneighbors(X_scaled, n_neighbors=k)
    mean_dist = float(dist.mean(axis=1)[0])
    thr_knn = float(art_knn["threshold"])
    in_knn = bool(mean_dist <= thr_knn)

    # IsolationForest
    iso = art_iso["model"]
    score = float(iso.decision_function(X_scaled).ravel()[0])
    thr_iso = float(art_iso["threshold"])
    in_iso = bool(score >= thr_iso)

    return ADResult(
        ocsvm_density=density, in_ad_ocsvm=in_ocsvm,
        knn_dist=mean_dist, in_ad_knn=in_knn,
        iso_score=score, in_ad_iso=in_iso
    )


# =========================
# GUI
# =========================

class PLPredictorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PL Predictor (tkinter)")
        self.geometry("980x620")

        # cache
        self._rule_cache: Dict[str, dict] = {}
        self._model_cache: Dict[Tuple[str, str], dict] = {}  # (kind, dtype) -> artifact

        # UI
        self._build_ui()

        # holder for image
        self._imgtk = None

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Top controls
        top = ttk.Frame(frm)
        top.pack(fill="x")

        ttk.Label(top, text="SMILES:").pack(side="left")
        self.smiles_var = tk.StringVar()
        self.smiles_entry = ttk.Entry(top, textvariable=self.smiles_var, width=80)
        self.smiles_entry.pack(side="left", padx=8)

        ttk.Label(top, text="Descriptor:").pack(side="left", padx=(10, 0))
        self.dtype_var = tk.StringVar(value="mordred_3d")
        self.dtype_combo = ttk.Combobox(top, textvariable=self.dtype_var, values=DESCRIPTOR_TYPES, state="readonly", width=12)
        self.dtype_combo.pack(side="left", padx=8)

        self.btn = ttk.Button(top, text="Predict", command=self.on_predict)
        self.btn.pack(side="left", padx=8)

        # Main area: left image, right text
        main = ttk.Frame(frm)
        main.pack(fill="both", expand=True, pady=10)

        left = ttk.Frame(main, width=360)
        left.pack(side="left", fill="y")

        self.img_label = ttk.Label(left, text="Molecule will appear here", anchor="center")
        self.img_label.pack(fill="both", expand=True)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.out_text = tk.Text(right, height=30, width=80)
        self.out_text.pack(fill="both", expand=True)

        # Footer
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self.status_var).pack(anchor="w")

    def _append_text(self, s: str):
        self.out_text.insert("end", s + "\n")
        self.out_text.see("end")

    def _clear_text(self):
        self.out_text.delete("1.0", "end")

    def _get_rule(self, dtype: str) -> dict:
        if dtype in self._rule_cache:
            return self._rule_cache[dtype]
        rule = _load_json_any(RULE_CANDIDATES[dtype])
        self._rule_cache[dtype] = rule
        return rule

    def _get_model_artifact(self, kind: str, dtype: str) -> dict:
        key = (kind, dtype)
        if key in self._model_cache:
            return self._model_cache[key]
        path = MODEL_ARTIFACT_PATH[kind].format(dtype=dtype)
        art = _load_joblib(path)
        self._model_cache[key] = art
        return art

    def _draw_molecule(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES (RDKit).")
        img = Draw.MolToImage(mol, size=(340, 340))
        self._imgtk = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._imgtk, text="")

    def _compute_processed_features(self, smiles: str, dtype: str) -> pd.DataFrame:
        # raw descriptor
        if dtype == "rdkit":
            X_raw = compute_rdkit_descriptors_one(smiles)
        elif dtype in ("mordred_2d", "mordred_3d"):
            X_raw = compute_mordred_descriptors_one(smiles, dtype)
        else:
            raise ValueError(dtype)

        # preprocess
        rule = self._get_rule(dtype)
        X_processed = transform_with_rule(X_raw, rule)
        return X_processed

    def on_predict(self):
        smiles = self.smiles_var.get().strip()
        dtype = self.dtype_var.get().strip()

        if not smiles:
            messagebox.showerror("Error", "SMILES is empty.")
            return
        if dtype not in DESCRIPTOR_TYPES:
            messagebox.showerror("Error", f"Unknown descriptor_type: {dtype}")
            return

        self.btn.configure(state="disabled")
        self.status_var.set("Running...")

        try:
            self._clear_text()
            self._append_text(f"Descriptor type: {dtype}")
            self._append_text(f"SMILES: {smiles}")
            self._append_text("")

            # draw
            self._draw_molecule(smiles)

            # compute processed features
            Xp = self._compute_processed_features(smiles, dtype)

            # ----------------
            # Predictions
            # ----------------
            preds: Dict[str, float] = {}

            for kind in ["pls", "rf", "xgb", "lgb"]:
                art = self._get_model_artifact(kind, dtype)
                cols = art["feature_columns"]
                X_model = Xp.reindex(columns=cols)
                preds[kind] = predict_sklearn_artifact_safe(art, X_model, model_kind=kind)

            # NN
            art_nn = self._get_model_artifact("nn", dtype)
            preds["nn"] = predict_nn_artifact(art_nn, Xp, device="cpu")

            pred_mean = float(np.mean(list(preds.values())))

            self._append_text("=== PL Predictions ===")
            for k in ["pls", "rf", "xgb", "lgb", "nn"]:
                self._append_text(f"{k:>4}: {preds[k]:.3f}")
            self._append_text(f"{'mean':>4}: {pred_mean:.3f}  (n_models={len(preds)})")
            self._append_text("")

            # ----------------
            # AD
            # ----------------
            ad = eval_ads(dtype, Xp)

            self._append_text("=== AD ===")
            self._append_text(f"ocsvm: density={ad.ocsvm_density:.6f}  -> {ad.in_ad_ocsvm}")
            self._append_text(f" knn:  dist={ad.knn_dist:.6f}      -> {ad.in_ad_knn}")
            self._append_text(f" iso: score={ad.iso_score:.6f}    -> {ad.in_ad_iso}")
            self._append_text(f"in_all_ads: {ad.in_all_ads}")

            self.status_var.set("Done.")

        except Exception as e:
            self.status_var.set("Error.")
            msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", msg)

        finally:
            self.btn.configure(state="normal")


def run_cli():
    print("GUI display is not available. Running in CLI mode.")
    print("Type SMILES and descriptor_type. Ctrl+C to exit.\n")

    while True:
        try:
            smiles = input("SMILES> ").strip()
            if not smiles:
                continue
            dtype = input("descriptor_type (rdkit/mordred_2d/mordred_3d)> ").strip()
            if dtype not in DESCRIPTOR_TYPES:
                print("Invalid descriptor_type.\n")
                continue

            # compute + predict using the same logic as GUI (minimal duplication)
            rule = _load_json_any(RULE_CANDIDATES[dtype])

            if dtype == "rdkit":
                X_raw = compute_rdkit_descriptors_one(smiles)
            else:
                X_raw = compute_mordred_descriptors_one(smiles, dtype)

            Xp = transform_with_rule(X_raw, rule)

            preds = {}
            for kind in ["pls", "rf", "xgb", "lgb"]:
                art = _load_joblib(MODEL_ARTIFACT_PATH[kind].format(dtype=dtype))
                cols = art["feature_columns"]
                X_model = Xp.reindex(columns=cols)
                preds[kind] = predict_sklearn_artifact_safe(art, X_model, model_kind=kind)

            art_nn = _load_joblib(MODEL_ARTIFACT_PATH["nn"].format(dtype=dtype))
            preds["nn"] = predict_nn_artifact(art_nn, Xp, device="cpu")

            pred_mean = float(np.mean(list(preds.values())))

            ad = eval_ads(dtype, Xp)

            print("\n=== Predictions ===")
            for k in ["pls", "rf", "xgb", "lgb", "nn"]:
                print(f"{k:>4}: {preds[k]:.3f}")
            print(f"{'mean':>4}: {pred_mean:.3f} (n_models=5)")

            print("\n=== AD ===")
            print(f"ocsvm: density={ad.ocsvm_density:.6f} -> {ad.in_ad_ocsvm}")
            print(f" knn:  dist={ad.knn_dist:.6f}      -> {ad.in_ad_knn}")
            print(f" iso: score={ad.iso_score:.6f}    -> {ad.in_ad_iso}")
            print(f"in_all_ads: {ad.in_all_ads}\n")

        except KeyboardInterrupt:
            print("\nbye")
            break
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}\n")
            print(traceback.format_exc())


if __name__ == "__main__":
    import sys
    from tkinter import TclError

    if sys.platform.startswith("win"):
        try:
            app = PLPredictorGUI()
            app.mainloop()
        except (TclError, RuntimeError) as e:
            print(f"[GUI failed on Windows] {type(e).__name__}: {e}")
            run_cli()
    else:
        if os.environ.get("DISPLAY", "").strip() == "":
            run_cli()
        else:
            try:
                app = PLPredictorGUI()
                app.mainloop()
            except (TclError, RuntimeError) as e:
                print(f"[GUI failed] {type(e).__name__}: {e}")
                run_cli()

