#!/usr/bin/env python3
"""
Evaluación HONESTA del TFT-lite de volatilidad y persistencia en metadata.json.

Este modelo es un forecaster de CUANTILES de volatilidad (q10/q50/q90) cuya
señal de "anomalía" es vol_real > q90. No existe un ground truth de anomalías
etiquetado, así que un F1/MCC "de detección" no tiene base — las métricas
correctas para este diseño son:

  • Pinball loss por cuantil (la loss nativa del problema)
  • Cobertura empírica: % de casos con vol_real ≤ q90 (ideal ≈ 90%) y
    dentro del intervalo q10–q90 (ideal ≈ 80%)
  • MAE/RMSE del q50 vs volatilidad realizada, comparado con un baseline
    naïve (persistencia: predecir que la vol de los próximos días = vol_5d actual)

Reconstruye el dataset igual que train() (mismos tickers/orden/split 80-20)
y evalúa sobre el 20% final. Guarda los resultados en metadata.json["eval_honesta"].
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch

from train import (ARTIFACTS, TICKERS, TFTLite, build_dataset, build_features,
                   download_real_returns, garch_series)

QUANTILES = (0.10, 0.50, 0.90)


def pinball(y, q_pred, q):
    err = y - q_pred
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def main():
    # ── Reconstruir dataset exactamente como train() ─────────────────────────
    all_X, all_y, naive_pred = [], [], []
    n_real = n_synth = 0
    for ticker, meta in TICKERS.items():
        real = download_real_returns(ticker, meta)
        if real is not None:
            ret, vol = real
            n_real += 1
        else:
            seed = abs(hash(ticker)) % 1000
            ret, vol = garch_series(1500, meta["vol_base"], seed=seed)
            n_synth += 1
        feat = build_features(ret, vol, len(ret))
        X, y = build_dataset(feat, vol)
        all_X.append(X)
        all_y.append(y)
        # baseline naïve: vol_5d del último día de la ventana (feature 2)
        naive_pred.append(X[:, -1, 2])
        print(f"  {ticker}: {len(X)} muestras ({'real' if real is not None else 'sintético'})")

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    naive = np.concatenate(naive_pred)

    scaler = joblib.load(ARTIFACTS / "scaler.joblib")
    B, T, F = X_all.shape
    X_flat = scaler.transform(X_all.reshape(-1, F)).reshape(B, T, F).astype(np.float32)

    ckpt = torch.load(ARTIFACTS / "tft_model.pt", map_location="cpu", weights_only=False)
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    model = TFTLite()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_train = int(0.8 * len(X_flat))
    X_va, y_va, naive_va = X_flat[n_train:], y_all[n_train:], naive[n_train:]
    print(f"\nEvaluando sobre {len(X_va)} muestras de validación "
          f"({n_real} tickers reales, {n_synth} sintéticos)")

    with torch.no_grad():
        pred = model(torch.tensor(X_va)).numpy()        # (B, horizon, 3)
    # desnormalizar (target en log1p z-score)
    pred_real = np.expm1(np.clip(pred * y_std + y_mean, -10, 10))
    q10, q50, q90 = pred_real[:, :, 0], pred_real[:, :, 1], pred_real[:, :, 2]

    # ── Métricas ──────────────────────────────────────────────────────────────
    mae = float(np.mean(np.abs(q50 - y_va)))
    rmse = float(np.sqrt(np.mean((q50 - y_va) ** 2)))
    naive_mat = np.repeat(naive_va[:, None], y_va.shape[1], axis=1)
    mae_naive = float(np.mean(np.abs(naive_mat - y_va)))

    pb = {f"q{int(q*100)}": round(pinball(y_va, pred_real[:, :, i], q), 6)
          for i, q in enumerate(QUANTILES)}
    cov_q90 = float(np.mean(y_va <= q90))            # ideal 0.90
    cov_q10 = float(np.mean(y_va >= q10))            # ideal 0.90
    cov_interval = float(np.mean((y_va >= q10) & (y_va <= q90)))  # ideal 0.80
    anomaly_rate = float(np.mean(y_va > q90))        # ≈ 1 - cov_q90

    print(f"MAE q50={mae:.5f} (naïve {mae_naive:.5f}) | RMSE={rmse:.5f}")
    print(f"Cobertura: ≤q90 {cov_q90:.3f} (ideal .90) | ≥q10 {cov_q10:.3f} | "
          f"intervalo {cov_interval:.3f} (ideal .80) | tasa anomalía {anomaly_rate:.3f}")
    print(f"Pinball: {pb}")

    # ── Persistir en metadata.json ─────────────────────────────────────────────
    meta_path = ARTIFACTS / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["eval_honesta"] = {
        "nota": ("El modelo predice cuantiles de volatilidad; la 'anomalía' es "
                 "vol_real>q90 por diseño, sin ground truth etiquetado — por eso "
                 "se reporta calibración, no F1/MCC."),
        "mae_q50": round(mae, 6),
        "rmse_q50": round(rmse, 6),
        "mae_baseline_naive": round(mae_naive, 6),
        "mejora_vs_naive_pct": round((1 - mae / mae_naive) * 100, 1),
        "pinball_loss": pb,
        "cobertura_q90": round(cov_q90, 4),
        "cobertura_q10": round(cov_q10, 4),
        "cobertura_intervalo_q10_q90": round(cov_interval, 4),
        "tasa_anomalias_val": round(anomaly_rate, 4),
        "n_val": int(len(X_va)),
        "datos": f"yfinance real: {n_real} tickers | GARCH sintético: {n_synth}",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\n✅ metadata.json actualizado con eval_honesta")


if __name__ == "__main__":
    main()
