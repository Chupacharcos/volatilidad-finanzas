"""FastAPI router — Detección de Volatilidad Anómala (TFT-lite + FinBERT-inspired)."""

import json, math, datetime, random
import numpy as np, torch, joblib
from pathlib import Path
from fastapi import APIRouter, Query

ARTIFACTS = Path(__file__).parent / "artifacts"
router    = APIRouter(prefix="/ml")

# ── Carga lazy del modelo ─────────────────────────────────────────────────────

_model_cache    = None
_scaler_cache   = None
_ckpt_cache     = None
_metadata_cache = {}

def _load_model():
    global _model_cache, _scaler_cache, _ckpt_cache, _metadata_cache
    if _model_cache is not None:
        return
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import TFTLite, LOOKBACK, HORIZON, N_FEAT, HIDDEN, N_HEADS, N_LAYERS

    ckpt = torch.load(ARTIFACTS / "tft_model.pt", map_location="cpu", weights_only=False)
    m    = TFTLite(n_feat=ckpt["n_feat"], hidden=ckpt["hidden"],
                   n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                   horizon=ckpt["horizon"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    _model_cache  = m
    _ckpt_cache   = ckpt
    _scaler_cache = joblib.load(ARTIFACTS / "scaler.joblib")
    with open(ARTIFACTS / "metadata.json", encoding="utf-8") as f:
        _metadata_cache = json.load(f)


# ── Generación de datos sintéticos ───────────────────────────────────────────

TICKERS_META = {
    # ── Renta variable española ─────────────────────────────────────────────
    "IBEX35": {"nombre": "IBEX 35",         "sector": "Índice",    "clase": "Bolsa",  "precio_base": 10_800.0, "vol_base": 0.012, "color": "#64ffda", "continuo": False},
    "SAN":    {"nombre": "Santander",        "sector": "Banca",     "clase": "Bolsa",  "precio_base": 4.20,     "vol_base": 0.018, "color": "#38bdf8", "continuo": False},
    "ITX":    {"nombre": "Inditex",          "sector": "Retail",    "clase": "Bolsa",  "precio_base": 48.50,    "vol_base": 0.014, "color": "#a78bfa", "continuo": False},
    "BBVA":   {"nombre": "BBVA",             "sector": "Banca",     "clase": "Bolsa",  "precio_base": 9.80,     "vol_base": 0.017, "color": "#fb923c", "continuo": False},
    "TEF":    {"nombre": "Telefónica",       "sector": "Telecom",   "clase": "Bolsa",  "precio_base": 4.10,     "vol_base": 0.016, "color": "#f472b6", "continuo": False},
    # ── Criptomonedas ──────────────────────────────────────────────────────
    "BTC":    {"nombre": "Bitcoin",          "sector": "Crypto",    "clase": "Crypto", "precio_base": 85_000.0, "vol_base": 0.038, "color": "#f7931a", "continuo": True},
    "ETH":    {"nombre": "Ethereum",         "sector": "Crypto",    "clase": "Crypto", "precio_base": 3_200.0,  "vol_base": 0.048, "color": "#627eea", "continuo": True},
    "BNB":    {"nombre": "BNB",              "sector": "Crypto",    "clase": "Crypto", "precio_base": 580.0,    "vol_base": 0.052, "color": "#f3ba2f", "continuo": True},
    "SOL":    {"nombre": "Solana",           "sector": "Crypto",    "clase": "Crypto", "precio_base": 175.0,    "vol_base": 0.068, "color": "#9945ff", "continuo": True},
    "XRP":    {"nombre": "XRP",              "sector": "Crypto",    "clase": "Crypto", "precio_base": 0.58,     "vol_base": 0.062, "color": "#346aa9", "continuo": True},
    # ── NFT / DeFi / Altcoins ──────────────────────────────────────────────
    "MANA":   {"nombre": "Decentraland",     "sector": "NFT/DeFi",  "clase": "NFT/DeFi","precio_base": 0.48,   "vol_base": 0.095, "color": "#ff2d55", "continuo": True},
    "SAND":   {"nombre": "The Sandbox",      "sector": "NFT/DeFi",  "clase": "NFT/DeFi","precio_base": 0.42,   "vol_base": 0.098, "color": "#04adef", "continuo": True},
    "APE":    {"nombre": "ApeCoin",          "sector": "NFT/DeFi",  "clase": "NFT/DeFi","precio_base": 1.20,   "vol_base": 0.105, "color": "#0063f5", "continuo": True},
    "UNI":    {"nombre": "Uniswap",          "sector": "NFT/DeFi",  "clase": "NFT/DeFi","precio_base": 8.50,   "vol_base": 0.072, "color": "#ff007a", "continuo": True},
    "LINK":   {"nombre": "Chainlink",        "sector": "NFT/DeFi",  "clase": "NFT/DeFi","precio_base": 14.80,  "vol_base": 0.078, "color": "#2a5ada", "continuo": True},
}


def _garch_series(n, vol_base, omega=8e-6, alpha=0.12, beta=0.82, seed=0):
    rng    = np.random.default_rng(seed)
    sigma2 = np.zeros(n)
    ret    = np.zeros(n)
    sigma2[0] = vol_base ** 2
    for t in range(1, n):
        sigma2[t] = omega + alpha * ret[t-1]**2 + beta * sigma2[t-1]
        if rng.random() < 0.005:
            sigma2[t] *= rng.uniform(3.0, 6.0)
        ret[t] = math.sqrt(sigma2[t]) * rng.standard_normal()
    return ret, np.sqrt(sigma2)


def _build_features_single(ret, vol, t_start, length):
    """Features para una ventana [t_start, t_start+length)."""
    feat = np.zeros((length, 9), dtype=np.float32)
    for i, t in enumerate(range(t_start, t_start + length)):
        feat[i, 0] = float(np.clip(ret[t], -0.15, 0.15))
        feat[i, 1] = float(np.std(ret[max(0, t-10):t+1]))
        feat[i, 2] = float(np.std(ret[max(0, t-5):t+1]))
        feat[i, 3] = vol[t]
        feat[i, 4] = abs(ret[t])
        dow = t % 5
        feat[i, 5] = math.sin(2 * math.pi * dow / 5)
        feat[i, 6] = math.cos(2 * math.pi * dow / 5)
        month = (t // 21) % 12
        feat[i, 7] = math.sin(2 * math.pi * month / 12)
        feat[i, 8] = math.cos(2 * math.pi * month / 12)
    return feat


def _trading_dates(n, end=None, continuo=False):
    """Genera n fechas hacia atrás desde end.
    continuo=True: todos los días (cripto 24/7). False: solo hábiles (bolsa)."""
    if end is None:
        end = datetime.date.today()
    dates, d = [], end
    while len(dates) < n:
        if continuo or d.weekday() < 5:
            dates.append(d)
        d -= datetime.timedelta(days=1)
    return list(reversed(dates))


def _forecast_dates(n, start=None, continuo=False):
    """Genera n fechas hacia adelante desde start."""
    if start is None:
        start = datetime.date.today()
    dates, d = [], start + datetime.timedelta(days=1)
    while len(dates) < n:
        if continuo or d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)
    return dates


# ── Predicción principal ──────────────────────────────────────────────────────

def _predict_ticker(ticker: str):
    _load_model()
    meta    = TICKERS_META[ticker]
    ckpt    = _ckpt_cache
    lookback = ckpt["lookback"]
    horizon  = ckpt["horizon"]
    y_mean   = ckpt["y_mean"]
    y_std    = ckpt["y_std"]

    seed = abs(hash(ticker)) % 1000
    N    = 600
    ret, vol = _garch_series(N, meta["vol_base"], seed=seed)

    # Usar los últimos `lookback` días como input
    t_end = N - horizon
    feat  = _build_features_single(ret, vol, t_end - lookback, lookback)
    feat_norm = _scaler_cache.transform(feat).astype(np.float32)

    inp  = torch.tensor(feat_norm[np.newaxis])          # (1, lookback, n_feat)
    with torch.no_grad():
        out = _model_cache(inp)[0].numpy()              # (horizon, 3)

    # Desnormalizar (escala log)
    def denorm(x):
        return float(np.expm1(np.clip(x * y_std + y_mean, -10, 10)))

    pred_q10 = [denorm(out[h, 0]) for h in range(horizon)]
    pred_q50 = [denorm(out[h, 1]) for h in range(horizon)]
    pred_q90 = [denorm(out[h, 2]) for h in range(horizon)]

    # Histórico: últimos 30 días
    hist_len  = 30
    hist_ret  = ret[t_end - hist_len:t_end]
    hist_vol  = vol[t_end - hist_len:t_end]
    continuo   = meta.get("continuo", False)
    hist_dates = _trading_dates(hist_len, continuo=continuo)

    # Precio histórico sintético (acumulación de retornos)
    precio = meta["precio_base"]
    precios = [precio]
    for r in hist_ret:
        precio = precios[-1] * (1 + r)
        precios.append(precio)
    precios = precios[1:]

    # Detección de anomalías históricas (vol > q90 modelo del día anterior)
    hist_q90 = [float(v * 1.8) for v in hist_vol]      # umbral simplificado

    historico = []
    for i in range(hist_len):
        anomalia = bool(hist_vol[i] > hist_q90[i])
        historico.append({
            "fecha":     hist_dates[i].isoformat(),
            "precio":    round(float(precios[i]), 4),
            "vol":       round(float(hist_vol[i]), 5),
            "vol_q90":   round(float(hist_q90[i]), 5),
            "anomalia":  anomalia,
            "retorno":   round(float(hist_ret[i]) * 100, 3),  # en %
        })

    # Predicción 5 días
    fcast_dates = _forecast_dates(horizon, continuo=continuo)
    # Proyección de precio (último precio × (1 + retorno esperado=0))
    ultimo_precio = float(precios[-1])
    prediccion = []
    for h in range(horizon):
        prediccion.append({
            "fecha":     fcast_dates[h].isoformat(),
            "vol_q10":   round(max(pred_q10[h], 0.001), 5),
            "vol_q50":   round(max(pred_q50[h], 0.001), 5),
            "vol_q90":   round(max(pred_q90[h], 0.001), 5),
            "precio_estimado": round(ultimo_precio * (1 + np.random.default_rng(seed+h).normal(0, pred_q50[h])), 4),
        })

    # Nivel de riesgo (0-100) basado en vol q50 relativa a la media histórica
    vol_hist_mean = float(np.mean(hist_vol))
    vol_pred_mean = float(np.mean(pred_q50))
    nivel_riesgo  = min(100, int((vol_pred_mean / (vol_hist_mean + 1e-8)) * 50))

    n_anomalias = sum(1 for h in historico if h["anomalia"])

    return {
        "ticker":         ticker,
        "nombre":         meta["nombre"],
        "sector":         meta["sector"],
        "clase":          meta["clase"],
        "color":          meta["color"],
        "continuo":       meta.get("continuo", False),
        "historico":      historico,
        "prediccion":     prediccion,
        "nivel_riesgo":   nivel_riesgo,
        "n_anomalias":    n_anomalias,
        "vol_media_hist": round(vol_hist_mean, 5),
        "vol_pred_media": round(vol_pred_mean, 5),
    }


# ── Noticias con sentimiento (FinBERT-inspired) ───────────────────────────────

# Corpus de noticias plantilla por sector
_NOTICIAS = {
    "Índice": [
        ("El {n} sube un {p:.1f}% impulsado por el sector bancario",         0.78),
        ("El {n} cae un {p:.1f}% ante la incertidumbre macroeconómica",      -0.72),
        ("El {n} consolida niveles tras semanas de alta volatilidad",         0.10),
        ("Inversores extranjeros aumentan posiciones en bolsa española",       0.65),
        ("BCE mantiene tipos: mercados reaccionan con cautela",                0.05),
        ("El {n} marca máximos anuales con volumen récord de negociación",    0.88),
        ("Tensiones geopolíticas presionan a la baja los índices europeos",   -0.68),
        ("S&P 500 arrastra al {n} tras dato de inflación en EE.UU.",          -0.55),
        ("Goldman Sachs eleva su objetivo para la renta variable española",    0.70),
        ("El {n} cotiza plano a la espera del dato de empleo en la Eurozona",  0.02),
    ],
    "Banca": [
        ("{n} bate previsiones con un beneficio neto de {b:.0f}M€ en el trimestre", 0.82),
        ("{n} reduce su exposición a deuda soberana en mercados emergentes",         -0.45),
        ("{n} anuncia un dividendo extraordinario de {d:.2f}€ por acción",           0.75),
        ("Moody's mejora la perspectiva de {n} a positiva",                          0.68),
        ("{n} sufre caída del {p:.1f}% tras datos de morosidad por encima de lo esperado", -0.79),
        ("BCE eleva el colchón de capital requerido: {n} entre los más afectados",   -0.62),
        ("{n} cierra la mayor emisión de bonos verdes de su historia: {b:.0f}M€",    0.60),
        ("Analistas de JPMorgan reiteran 'overweight' en {n} con precio objetivo revisado al alza", 0.72),
        ("{n} recorta plantilla en un {p:.1f}% como parte de su plan de digitalización", -0.35),
        ("Fusión exploratoria entre {n} y rival europeo: mercado reacciona positivamente", 0.58),
    ],
    "Retail": [
        ("{n} supera las expectativas de ventas con un crecimiento del {p:.1f}% YoY", 0.85),
        ("{n} acelera su expansión en Asia con {b:.0f} nuevas tiendas en 2026",        0.70),
        ("Desaceleración del consumo lastra las previsiones de {n} para el Q3",        -0.60),
        ("{n} lanza nueva línea de lujo: analistas anticipan márgenes del {p:.1f}%",   0.55),
        ("Huelga de distribución impacta temporalmente a {n}",                          -0.48),
        ("{n} eleva su dividendo un {p:.1f}% tras récord de ingresos",                  0.80),
        ("El fortalecimiento del euro reduce la competitividad de {n} fuera de Europa", -0.40),
        ("Fitch reafirma rating A+ de {n}: sólida generación de caja",                  0.65),
    ],
    "Telecom": [
        ("{n} gana {b:.0f}M€ en el semestre, superando el consenso de mercado",        0.75),
        ("{n} acelera el despliegue de fibra óptica con inversión de {b:.0f}M€",        0.52),
        ("Regulador europeo abre expediente a {n} por prácticas anticompetitivas",      -0.70),
        ("{n} reduce deuda neta un {p:.1f}% gracias a la venta de torres",              0.60),
        ("Alta competencia en precios presiona los márgenes de {n}",                    -0.55),
        ("{n} lidera el despliegue de 5G en España con cobertura del {p:.1f}% del territorio", 0.65),
        ("Analistas rebajan recomendación de {n} a 'neutral' tras revisión de previsiones", -0.42),
    ],
    "Crypto": [
        ("{n} supera los ${b:.0f} tras anuncio de ETF spot aprobado por la SEC",        0.92),
        ("Whales mueven {b:.0f}M$ en {n}: mercado anticipa movimiento brusco",          -0.30),
        ("{n} cae un {p:.1f}% tras liquidaciones masivas en derivados",                 -0.82),
        ("Adopción institucional: fondo soberano de Noruega añade {n} a su cartera",    0.88),
        ("Halving de {n}: histórico catalizador alcista en el corto-medio plazo",       0.78),
        ("Regulador de EE.UU. impone multa millonaria a exchange: {n} entre los afectados", -0.65),
        ("{n} marca máximos históricos con volumen récord en las últimas 24h",          0.95),
        ("Hack a protocolo DeFi drena liquidez: {n} cae un {p:.1f}% en horas",         -0.88),
        ("BlackRock aumenta su posición en {n} en {b:.0f}M$",                          0.82),
        ("Nodo de {n} experimenta congestión: fees suben un {p:.1f}%",                 -0.38),
        ("China relaja restricciones sobre {n}: entrada de capital asiático esperada",  0.70),
        ("Fed mantiene tipos altos: activos de riesgo como {n} bajo presión",          -0.58),
        ("{n} integrado como medio de pago en plataforma con {b:.0f}M de usuarios",    0.75),
        ("Ballenatos acumulan {n} en silencio: on-chain data muestra compras masivas", 0.60),
        ("Liquidaciones en {p:.1f}M$ de posiciones largas en {n} en 1 hora",          -0.72),
    ],
    "NFT/DeFi": [
        ("Volumen de NFTs en {n} supera los {b:.0f}M$ en la última semana",            0.80),
        ("{n} sufre exploit en su contrato inteligente: {b:.0f}M$ comprometidos",      -0.95),
        ("Metaverso de {n} atrae a {b:.0f} usuarios activos en evento especial",       0.72),
        ("Liquidity mining en {n}: APY del {p:.1f}% atrae nuevo capital",              0.65),
        ("{n} pierde un {p:.1f}% tras caída de volumen en mercados NFT",               -0.70),
        ("Colección blue-chip en {n} se vende por {b:.0f}ETH, impulsando el token",   0.85),
        ("DAO de {n} vota para quemar el {p:.1f}% del supply: deflación esperada",     0.68),
        ("Grandes marcas abandonan proyectos NFT: {n} entre los perjudicados",         -0.75),
        ("{n} lanza V2 con mejoras en liquidez concentrada y menor slippage",           0.70),
        ("Reguladores de la UE clasifican tokens DeFi como valores: {n} en el punto de mira", -0.65),
        ("Bridge cross-chain de {n} sufre ataque: fondos en riesgo",                   -0.90),
        ("Inversión de {b:.0f}M$ en {n} liderada por a16z y Paradigm",                0.88),
        ("Floor price de NFTs en {n} cae un {p:.1f}% en 24h: mercado en corrección",  -0.62),
        ("{n} integra zkRollup: fees caen un {p:.1f}% y velocidad se multiplica",      0.75),
        ("Influencer {n} acusado de pump-and-dump: token pierde {p:.1f}% en minutos",  -0.85),
    ],
}


def _generar_noticias(ticker: str, n: int = 5):
    meta   = TICKERS_META[ticker]
    sector = meta["sector"]
    nombre = meta["nombre"]
    corpus = _NOTICIAS.get(sector, _NOTICIAS["Índice"])

    # Seed reproducible pero que varía por día (noticias "de hoy")
    day_seed = int(datetime.date.today().strftime("%Y%m%d")) + abs(hash(ticker)) % 100
    rng      = random.Random(day_seed)

    seleccionadas = rng.sample(corpus, min(n, len(corpus)))
    noticias = []
    for i, (plantilla, base_score) in enumerate(seleccionadas):
        # Formatear plantilla con valores aleatorios realistas
        texto = plantilla.format(
            n=nombre,
            p=rng.uniform(1.2, 8.5),
            b=rng.uniform(200, 2000),
            d=rng.uniform(0.05, 0.35),
        )
        # Añadir ruido al score (-0.15 a +0.15)
        score = max(-1.0, min(1.0, base_score + rng.uniform(-0.15, 0.15)))

        if score >  0.25:
            sentimiento = "positivo"
        elif score < -0.25:
            sentimiento = "negativo"
        else:
            sentimiento = "neutro"

        fecha = (datetime.date.today() - datetime.timedelta(days=i)).isoformat()
        noticias.append({
            "titulo":      texto,
            "sentimiento": sentimiento,
            "score":       round(score, 3),
            "fecha":       fecha,
        })

    sentimiento_agg = sum(n["score"] for n in noticias) / len(noticias)
    if sentimiento_agg > 0.20:
        label_agg = "Positivo"
    elif sentimiento_agg < -0.20:
        label_agg = "Negativo"
    else:
        label_agg = "Neutro"

    return {
        "noticias":            noticias,
        "sentimiento_agregado": round(sentimiento_agg, 3),
        "label_agregado":       label_agg,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/volatilidad/tickers")
def tickers():
    return {"tickers": [{"id": tid, **{k: v for k, v in meta.items() if k != "color"}}
                         for tid, meta in TICKERS_META.items()]}


@router.get("/volatilidad/prediccion")
def prediccion(ticker: str = Query("IBEX35")):
    ticker = ticker.upper()
    if ticker not in TICKERS_META:
        return {"ok": False, "error": f"Ticker '{ticker}' no disponible. Usa: {list(TICKERS_META)}"}
    try:
        data = _predict_ticker(ticker)
        return {"ok": True, **data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/volatilidad/noticias")
def noticias(ticker: str = Query("IBEX35"), n: int = Query(5, ge=1, le=10)):
    ticker = ticker.upper()
    if ticker not in TICKERS_META:
        return {"ok": False, "error": f"Ticker '{ticker}' no disponible"}
    try:
        return {"ok": True, "ticker": ticker, **_generar_noticias(ticker, n)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/volatilidad/stats")
def stats():
    try:
        _load_model()
        return {"ok": True, **_metadata_cache}
    except Exception as e:
        return {"ok": False, "error": str(e)}
