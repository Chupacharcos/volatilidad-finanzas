"""
TFT-lite — Detección de Volatilidad Anómala
============================================
Temporal Fusion Transformer simplificado para predicción de volatilidad
financiera a 5 días hábiles.

Arquitectura:
  - Variable Selection Network (proyección lineal con gating sigmoidal)
  - LSTM encoder bidireccional (lookback=30)
  - Temporal Self-Attention (4 cabezas)
  - Salida cuantílica: q10 (banda inferior), q50 (media), q90 (banda superior)

Datos: sintéticos con dinámica GARCH(1,1) para 5 activos del mercado español.
Anomalía: vol_real > banda_q90 del modelo.
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── Configuración ─────────────────────────────────────────────────────────────

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Tickers reales — se descargan de Yahoo Finance (5 años de datos)
TICKERS = {
    "IBEX35": {"nombre": "IBEX 35",   "sector": "Índice",  "precio_base": 10_800.0, "vol_base": 0.012, "yf_symbol": "^IBEX"},
    "SAN":    {"nombre": "Santander", "sector": "Banca",   "precio_base": 4.20,     "vol_base": 0.018, "yf_symbol": "SAN.MC"},
    "ITX":    {"nombre": "Inditex",   "sector": "Retail",  "precio_base": 48.50,    "vol_base": 0.014, "yf_symbol": "ITX.MC"},
    "BBVA":   {"nombre": "BBVA",      "sector": "Banca",   "precio_base": 9.80,     "vol_base": 0.017, "yf_symbol": "BBVA.MC"},
    "TEF":    {"nombre": "Telefónica","sector": "Telecom", "precio_base": 4.10,     "vol_base": 0.016, "yf_symbol": "TEF.MC"},
}

LOOKBACK  = 30   # días de contexto
HORIZON   = 5    # días de predicción
N_FEAT    = 9    # features por paso temporal
HIDDEN    = 64
N_HEADS   = 4
N_LAYERS  = 2
DROPOUT   = 0.10
EPOCHS    = 80
LR        = 1e-3
BATCH     = 64

# ── Descarga de datos reales (yfinance) ───────────────────────────────────────

def download_real_returns(ticker_id: str, meta: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Descarga retornos diarios reales de Yahoo Finance (5 años).
    Calcula volatilidad realizada como std móvil de retornos.
    Fuente: https://finance.yahoo.com — acceso público sin API key.
    """
    try:
        import yfinance as yf
        symbol = meta.get("yf_symbol", ticker_id)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5y", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 100:
            return None
        closes = hist["Close"].dropna().values.astype(float)
        ret = np.diff(np.log(closes))       # log-retornos diarios
        # Volatilidad realizada: std móvil 10d (anualizada → diaria)
        vol = np.array([
            float(np.std(ret[max(0, t-10):t+1])) if t >= 2 else abs(ret[t])
            for t in range(len(ret))
        ])
        # Clip extremos
        ret = np.clip(ret, -0.15, 0.15)
        vol = np.clip(vol, 1e-5, 0.15)
        print(f"  {ticker_id} ({symbol}): {len(ret)} días reales | vol_media={vol.mean():.4f}")
        return ret, vol
    except Exception as e:
        print(f"  {ticker_id}: yfinance error ({e}), usando GARCH sintético")
        return None


# ── Generación de datos GARCH (fallback) ──────────────────────────────────────

def garch_series(n, vol_base, omega=8e-6, alpha=0.12, beta=0.82, seed=0):
    """Genera serie de retornos con dinámica GARCH(1,1) (fallback si yfinance falla)."""
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n)
    ret    = np.zeros(n)
    sigma2[0] = vol_base ** 2
    for t in range(1, n):
        sigma2[t] = omega + alpha * ret[t-1]**2 + beta * sigma2[t-1]
        if rng.random() < 0.005:
            sigma2[t] *= rng.uniform(3.0, 6.0)
        ret[t] = math.sqrt(sigma2[t]) * rng.standard_normal()
    return ret, np.sqrt(sigma2)


def build_features(ret: np.ndarray, vol: np.ndarray, n_days: int):
    """Construye matriz de features [T, N_FEAT]."""
    T = len(ret)
    feat = np.zeros((T, N_FEAT))
    for t in range(T):
        # 0: log-return clipeado
        feat[t, 0] = np.clip(ret[t], -0.15, 0.15)
        # 1: volatilidad realizada 10d
        feat[t, 1] = float(np.std(ret[max(0, t-10):t+1])) if t > 0 else vol[t]
        # 2: volatilidad realizada 5d
        feat[t, 2] = float(np.std(ret[max(0, t-5):t+1]))  if t > 0 else vol[t]
        # 3: vol GARCH implícita
        feat[t, 3] = vol[t]
        # 4: |return| (proxy de skew)
        feat[t, 4] = abs(ret[t])
        # 5-6: día de la semana (sin/cos)
        dow = t % 5
        feat[t, 5] = math.sin(2 * math.pi * dow / 5)
        feat[t, 6] = math.cos(2 * math.pi * dow / 5)
        # 7-8: mes del año (sin/cos)
        month = (t // 21) % 12
        feat[t, 7] = math.sin(2 * math.pi * month / 12)
        feat[t, 8] = math.cos(2 * math.pi * month / 12)
    return feat.astype(np.float32)


def build_dataset(feat: np.ndarray, target_vol: np.ndarray):
    """Crea pares (X, y) con ventana deslizante."""
    X, y = [], []
    T = len(feat)
    for i in range(LOOKBACK, T - HORIZON + 1):
        X.append(feat[i - LOOKBACK:i])                 # (lookback, n_feat)
        y.append(target_vol[i:i + HORIZON])            # (horizon,)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ── Modelo TFT-lite ───────────────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """Selección y proyección de features con gating."""
    def __init__(self, n_feat, hidden):
        super().__init__()
        self.proj  = nn.Linear(n_feat, hidden)
        self.gate  = nn.Linear(n_feat, hidden)

    def forward(self, x):
        # x: (B, T, n_feat)
        return self.proj(x) * torch.sigmoid(self.gate(x))


class TFTLite(nn.Module):
    """
    Temporal Fusion Transformer simplificado.

    Flujo:
      input → VSN → LSTM → Multi-head Self-Attention → Linear → quantiles
    """
    def __init__(self, n_feat=N_FEAT, hidden=HIDDEN, n_heads=N_HEADS,
                 n_layers=N_LAYERS, horizon=HORIZON, dropout=DROPOUT):
        super().__init__()
        self.vsn      = VariableSelectionNetwork(n_feat, hidden)
        self.lstm     = nn.LSTM(hidden, hidden, num_layers=n_layers,
                                batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attn     = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                              batch_first=True)
        self.norm1    = nn.LayerNorm(hidden)
        self.ff       = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.norm2    = nn.LayerNorm(hidden)
        self.drop     = nn.Dropout(dropout)
        # 3 cuantiles × horizon
        self.out      = nn.Linear(hidden, horizon * 3)
        self.horizon  = horizon

    def forward(self, x):
        # x: (B, T, n_feat)
        z = self.vsn(x)                                 # (B, T, H)
        z, _ = self.lstm(z)                             # (B, T, H)
        # Temporal self-attention sobre todos los pasos
        attn_out, _ = self.attn(z, z, z)
        z = self.norm1(z + self.drop(attn_out))
        z = self.norm2(z + self.drop(self.ff(z)))
        # Usar último paso temporal
        last = z[:, -1, :]                              # (B, H)
        out  = self.out(last)                           # (B, horizon*3)
        return out.view(-1, self.horizon, 3)            # (B, horizon, 3)


def quantile_loss(pred, target, quantiles=(0.10, 0.50, 0.90)):
    """Pinball loss para tres cuantiles."""
    loss = 0.0
    for i, q in enumerate(quantiles):
        err  = target - pred[:, :, i]
        loss += (q * err.clamp(min=0) + (1 - q) * (-err).clamp(min=0)).mean()
    return loss / len(quantiles)

# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("TFT-lite — Entrenamiento de volatilidad anómala")
    print("=" * 60)

    all_X, all_y = [], []
    ticker_stats = {}
    n_real, n_synthetic = 0, 0

    for ticker, meta in TICKERS.items():
        # Intentar datos reales de Yahoo Finance
        real = download_real_returns(ticker, meta)
        if real is not None:
            ret, vol = real
            n_real += 1
            data_source = "Yahoo Finance (real)"
        else:
            # Fallback a GARCH sintético
            seed = abs(hash(ticker)) % 1000
            ret, vol = garch_series(1500, meta["vol_base"], seed=seed)
            n_synthetic += 1
            data_source = "GARCH sintético (fallback)"

        feat = build_features(ret, vol, len(ret))
        X, y = build_dataset(feat, vol)
        all_X.append(X)
        all_y.append(y)
        ticker_stats[ticker] = {
            "vol_media":   float(np.mean(vol)),
            "vol_max":     float(np.max(vol)),
            "n_dias":      len(ret),
            "data_source": data_source,
        }
        print(f"  {ticker}: {len(X)} muestras | vol_media={np.mean(vol):.4f} | {data_source}")

    print(f"\n  Datos reales: {n_real} tickers | Sintéticos: {n_synthetic} tickers")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Normalizar features
    B, T, F = X_all.shape
    scaler  = StandardScaler()
    X_flat  = scaler.fit_transform(X_all.reshape(-1, F)).reshape(B, T, F).astype(np.float32)

    # Normalizar target (vol en escala log para mejor convergencia)
    y_log   = np.log1p(y_all).astype(np.float32)
    y_mean  = float(y_log.mean())
    y_std   = float(y_log.std()) + 1e-8
    y_norm  = ((y_log - y_mean) / y_std).astype(np.float32)

    # Split train/val (80/20)
    n_train = int(0.8 * len(X_flat))
    X_tr, X_va = X_flat[:n_train], X_flat[n_train:]
    y_tr, y_va = y_norm[:n_train], y_norm[n_train:]

    model     = TFTLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_val  = float("inf")
    best_state = None

    print(f"\nEntrenando {EPOCHS} épocas | train={n_train} | val={len(X_va)}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx  = np.random.permutation(n_train)
        loss_ep = 0.0
        n_batches = 0
        for start in range(0, n_train, BATCH):
            b_idx = idx[start:start + BATCH]
            xb = torch.tensor(X_flat[b_idx])
            yb = torch.tensor(y_norm[b_idx])
            pred = model(xb)                        # (B, horizon, 3)
            loss = quantile_loss(pred, yb)          # yb: (B, horizon)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_ep += loss.item()
            n_batches += 1

        # Validación
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_va)
            yv = torch.tensor(y_va)
            pv = model(xv)
            val_loss = quantile_loss(pv, yv).item()

        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Época {epoch:3d} | train={loss_ep/n_batches:.4f} | val={val_loss:.4f}")

    # Restaurar mejor modelo
    model.load_state_dict(best_state)

    # MAE en val (cuantil q50 = índice 1)
    model.eval()
    with torch.no_grad():
        pv_best = model(torch.tensor(X_va))
        q50_norm = pv_best[:, :, 1].numpy()
        q50_log  = q50_norm * y_std + y_mean
        q50_real = np.expm1(np.clip(q50_log, -10, 10))
        y_real   = np.expm1(np.clip(y_va * y_std + y_mean, -10, 10))
        mae  = float(np.mean(np.abs(q50_real - y_real)))
        rmse = float(np.sqrt(np.mean((q50_real - y_real) ** 2)))

    print(f"\nMejor val_loss={best_val:.4f} | MAE={mae:.5f} | RMSE={rmse:.5f}")

    # ── Guardar artifacts ──────────────────────────────────────────────────────
    checkpoint = {
        "state_dict": model.state_dict(),
        "n_feat":     N_FEAT,
        "hidden":     HIDDEN,
        "n_heads":    N_HEADS,
        "n_layers":   N_LAYERS,
        "horizon":    HORIZON,
        "lookback":   LOOKBACK,
        "y_mean":     y_mean,
        "y_std":      y_std,
    }
    torch.save(checkpoint, ARTIFACTS / "tft_model.pt")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")

    metadata = {
        "modelo":          "TFT-lite (VSN + LSTM + Multi-head Self-Attention)",
        "arquitectura":    "Variable Selection → LSTM(64,2) → MHA(4h) → Quantile Output",
        "n_features":      N_FEAT,
        "features":        ["log_return","vol_10d","vol_5d","vol_garch","abs_return","dow_sin","dow_cos","month_sin","month_cos"],
        "hidden_dim":      HIDDEN,
        "n_heads":         N_HEADS,
        "n_layers":        N_LAYERS,
        "lookback":        LOOKBACK,
        "horizon":         HORIZON,
        "cuantiles":       [0.10, 0.50, 0.90],
        "deteccion":       "Anomalía cuando vol_real > q90 del modelo",
        "tickers":         list(TICKERS.keys()),
        "ticker_meta":     TICKERS,
        "dias_entrenamiento": len(X_all),
        "epochs":          EPOCHS,
        "mae":             round(mae, 6),
        "rmse":            round(rmse, 6),
        "val_loss":        round(best_val, 4),
        "datos":           f"Yahoo Finance (real): {n_real} tickers | GARCH fallback: {n_synthetic} tickers",
    }
    with open(ARTIFACTS / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n✅ Artifacts guardados en artifacts/")
    print(f"   MAE volatilidad = {mae:.5f} | RMSE = {rmse:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
