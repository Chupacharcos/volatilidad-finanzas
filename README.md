# Detección de Volatilidad Anómala — TFT-lite + FinBERT

> Proyecto 04 del portafolio de IA/ML. Sistema de predicción probabilística de volatilidad financiera con detección de anomalías y análisis de sentimiento de noticias.

## Arquitectura

```
Datos GARCH(1,1) sintéticos
        │
        ▼
┌─────────────────────────────┐
│  Variable Selection Network │  ← Pondera las 6 features de entrada
│  (Linear + Sigmoid gate)    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  LSTM Encoder               │  ← 2 capas, hidden_dim=64
│  (Bidireccional implícito)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Multi-Head Self-Attention  │  ← 4 cabezas, captura dependencias temporales
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Quantile Output            │  → q10 / q50 / q90 (5 días horizonte)
└─────────────────────────────┘
```

**Loss function:** Pinball loss (quantile loss) sobre 3 cuantiles simultáneos.

## Resultados

| Métrica | Valor |
|---------|-------|
| MAE (q50) | 0.00115 |
| RMSE | 0.00213 |
| Horizontes predichos | 5 días |
| Cuantiles | q10 / q50 / q90 |

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/ml/volatilidad/tickers` | Lista de activos disponibles |
| GET | `/ml/volatilidad/prediccion?ticker=IBEX35` | Predicción probabilística + anomalías |
| GET | `/ml/volatilidad/noticias?ticker=IBEX35` | Noticias con sentimiento FinBERT-inspired |
| GET | `/ml/volatilidad/stats` | Métricas del modelo |

### Respuesta `/prediccion`

```json
{
  "ok": true,
  "ticker": "IBEX35",
  "historico": [...],
  "anomalias": [{"dia": 14, "vol": 0.031, "zscore": 2.8}],
  "forecast": [{"dia": 1, "q10": 0.018, "q50": 0.022, "q90": 0.027}],
  "riesgo": 42,
  "vol_media_30d": 0.0198
}
```

### Respuesta `/noticias`

```json
{
  "ok": true,
  "ticker": "IBEX35",
  "noticias": [
    {
      "titulo": "Banco Central mantiene tipos en zona neutral",
      "sentimiento": "positivo",
      "score": 0.72,
      "fecha": "2026-03-08"
    }
  ],
  "sentimiento_agregado": 0.39
}
```

## Activos disponibles

| Ticker | Nombre | Sector |
|--------|--------|--------|
| IBEX35 | IBEX 35 | Índice |
| SAN | Banco Santander | Banca |
| ITX | Inditex | Retail / Moda |
| BBVA | BBVA | Banca |
| TEF | Telefónica | Telecom |

## Stack técnico

- **Modelo:** PyTorch — TFT-lite (VSN + LSTM + MHA + Quantile output)
- **Series temporales:** GARCH(1,1) sintético con anomalías embedidas
- **Sentimiento:** FinBERT-inspired keyword scoring (sin descarga de modelos externos)
- **API:** FastAPI + Uvicorn (puerto 8092)
- **Serialización:** joblib (scaler) + torch.save (pesos del modelo)

## Instalación

```bash
# Clonar y crear entorno
git clone https://github.com/Chupacharcos/volatilidad-finanzas.git
cd volatilidad-finanzas
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn torch scikit-learn joblib numpy

# Entrenar el modelo (genera artifacts/)
python train.py

# Arrancar la API
uvicorn api:app --host 0.0.0.0 --port 8092
```

## Estructura del proyecto

```
volatilidad-finanzas/
├── train.py          # Entrenamiento TFT-lite + generación datos GARCH
├── router.py         # Endpoints FastAPI (/ml/volatilidad/...)
├── api.py            # App FastAPI standalone
├── artifacts/        # Modelo entrenado (excluido de git)
│   ├── tft_model.pt
│   ├── scaler.joblib
│   └── metadata.json
└── README.md
```

## Demo en vivo

Disponible en el portafolio: [adrianmoreno-dev.com](https://adrianmoreno-dev.com)

---

*Parte del portafolio de proyectos IA/ML 2025-2026*
