# Detección de Volatilidad Anómala — TFT-lite

Sistema de predicción probabilística de volatilidad financiera con **detección de anomalías**. Arquitectura TFT-lite (Variable Selection Network + LSTM + Multi-Head Attention) entrenada con datos reales de mercado via yfinance.

Demo en producción: [adrianmoreno-dev.com/demo/volatilidad-anomala](https://adrianmoreno-dev.com/demo/volatilidad-anomala)

---

<!-- LOOP-MAP:START (generado por `php artisan project:loop readme` — no editar a mano) -->

## El bucle que cierra

<p align="center"><img src="https://adrianmoreno-dev.com/bucle/deteccion-volatilidad-anomala.svg" alt="Mapa del bucle de Detección de Volatilidad Anómala" width="900"></p>

**Para** quien vigila el riesgo de una cartera día a día · **Cada día**

| Etapa | Qué pasa | Quién |
|---|---|---|
| **1. Disparador** | Cierra el mercado y quiero saber si mañana puede venir un día raro. | persona |
| **2. Acción** | Predice la volatilidad del día siguiente con una TFT-lite y marca si se sale de lo normal para ese activo. | software |
| **3. Medición** | La volatilidad prevista con su intervalo y la marca de anomalía, con una falsa alarma cada 47 días normales. | software |
| **4. Decisión** | Decido si reduzco exposición, cubro o dejo la cartera quieta. | persona |

### Lo que no hace

- No predice el precio ni la dirección: solo cuánto se va a mover.
- No avisa de todo: se le escapa 1 de cada 5 anomalías reales, y está medido.
- No opera: marca el día raro, lo que se hace con eso es cosa tuya.

### Por qué está construido así

- **Predicción por cuantiles** en vez de predecir un único valor de volatilidad — Un número solo no dice si el modelo está seguro. Con los cuantiles sale un intervalo, y el ancho del intervalo ya es información.
- **Publicar la tasa de falsas alarmas** en vez de publicar solo el F1 de anomalías — Lo que decide si un detector es usable es cuántas veces te hace mirar sin motivo: 1 día de cada 47 en este caso.

<!-- LOOP-MAP:END -->

## Resultados

| Métrica | Valor |
|---------|-------|
| **F1-score anomalías** | **0.816** |
| **MCC** | **0.795** |
| **Precision** | 0.816 |
| **Recall** | 0.816 |
| FPR (falsas alarmas) | 2.1% |
| FNR (anomalías perdidas) | 18.4% |
| Calibración q10 | 9.9% (ideal: 10%) |
| Calibración q50 | 52.9% (ideal: 50%) |
| Calibración q90 | 91.5% (ideal: 90%) |
| Datos | yfinance real (IBEX35, SAN, ITX, BBVA, TEF); fallback a GARCH(1,1) sintético si la descarga falla |

> **MCC = 0.795** es excelente para detección de anomalías con clases muy desbalanceadas (~10% anómalas). Los intervalos de confianza están muy bien calibrados: cuando el modelo dice q90, realmente se supera en el 91.5% de los casos.

---

## Arquitectura

```
Retornos reales yfinance (IBEX35, SAN, ITX, BBVA, TEF — 5 años)
        │
        ▼
┌─────────────────────────────┐
│  Variable Selection Network │  ← Pondera las 6 features de entrada
│  (Linear + Sigmoid gate)    │     · retorno log diario
│                             │     · volatilidad rolling 5d
│                             │     · volatilidad rolling 20d
│                             │     · volatilidad rolling 60d
│                             │     · momentum 5d
│                             │     · momentum 20d
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  LSTM Encoder               │  ← 2 capas, hidden_dim=64
│                             │     Ventana temporal: 30 días
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Multi-Head Self-Attention  │  ← 4 cabezas
│  (dependencias temporales)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Quantile Output            │  → q10 / q50 / q90
│  (5 días horizonte)         │     Pinball loss (quantile loss)
└─────────────────────────────┘
             │
             ▼
    Anomalía = vol_real > P90 histórico
    F1=0.816 · MCC=0.795
```

### ¿Por qué quantile loss y no MSE?

El MSE predice la media — inútil en finanzas donde los extremos (colas) son lo que importa. La **pinball loss** entrena simultáneamente los percentiles 10, 50 y 90, dando un **intervalo de confianza probabilístico** para la volatilidad. Si el modelo dice q90=3%, hay un 90% de probabilidad de que la volatilidad real sea ≤3%.

---

## Detección de anomalías

Una sesión se clasifica como **anomalía de volatilidad** cuando la volatilidad predicha q90 supera el percentil 90 histórico. Métricas en el test set:

| | Predicho: Normal | Predicho: Anomalía |
|--|--|--|
| **Real: Normal** | 1.097 (TN) | 23 (FP) |
| **Real: Anomalía** | 23 (FN) | 102 (TP) |

- **FPR = 2.1%**: solo 1 de cada 47 días normales genera una falsa alarma
- **FNR = 18.4%**: detecta 4 de cada 5 episodios anómalos reales

---

## Activos disponibles

| Ticker | Nombre | Sector |
|--------|--------|--------|
| IBEX35 | IBEX 35 | Índice español |
| SAN | Banco Santander | Banca |
| ITX | Inditex | Retail / Moda |
| BBVA | BBVA | Banca |
| TEF | Telefónica | Telecom |

---

## Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/ml/volatilidad/tickers` | Lista de activos disponibles |
| `GET` | `/ml/volatilidad/prediccion?ticker=IBEX35` | Predicción probabilística + anomalías |
| `GET` | `/ml/volatilidad/noticias?ticker=IBEX35` | Noticias con sentimiento |
| `GET` | `/ml/volatilidad/stats` | Métricas del modelo |

### Respuesta `/prediccion` (extracto)

```json
{
  "ok": true,
  "ticker": "IBEX35",
  "anomalias": [{"dia": 14, "vol": 0.031, "zscore": 2.8, "es_anomalia": true}],
  "forecast": [
    {"dia": 1, "q10": 0.018, "q50": 0.022, "q90": 0.027}
  ],
  "riesgo": 42,
  "f1_score": 0.816,
  "mcc": 0.795
}
```

---

## Estructura del proyecto

```
volatilidad-anomala/
├── train.py          # Entrenamiento TFT-lite con datos reales yfinance
├── router.py         # Endpoints FastAPI (/ml/volatilidad/*)
├── api.py            # App FastAPI standalone (puerto 8092)
└── artifacts/        # Modelo entrenado (excluido de git)
    ├── tft_model.pt
    ├── scaler.joblib
    └── metadata.json
```

---

## Entrenamiento

```bash
cd /var/www/volatilidad-anomala
source /var/www/chatbot/venv/bin/activate
python3 train.py
# Descarga datos reales de yfinance. Si no disponible → GARCH(1,1) sintético calibrado.
# Genera artifacts/ con el modelo entrenado.
```

## Arranque del servicio

```bash
uvicorn api:app --host 127.0.0.1 --port 8092 --reload
```

---

## Stack técnico

- **Python 3.12** · **PyTorch 2.10** (TFT-lite: VSN + LSTM + MHA + Quantile output)
- **yfinance** — datos reales de mercado
- **NumPy / scikit-learn** — preprocesamiento y métricas
- **FastAPI / Uvicorn** — API REST
- **joblib** — serialización del scaler

---

*Parte del portafolio de proyectos IA/ML — [adrianmoreno-dev.com](https://adrianmoreno-dev.com)*
