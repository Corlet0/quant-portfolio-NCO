
# ==============================================================================
# 1. UNIVERSO GLOBAL MULTI-REGIÓN (EE. UU. + EUROPA + ASIA)
# ==============================================================================
print("1. Descargando índices de EE. UU., Europa y Asia con sectores...")

headers = {"User-Agent": "Mozilla/5.0"}

# ------------------------------------------------------------------------------
# 1.1. ESTADOS UNIDOS: S&P 500 (~503 activos)
# ------------------------------------------------------------------------------
url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
req_500 = urllib.request.Request(url_sp500, headers=headers)
with urllib.request.urlopen(req_500) as resp:
    t_500 = pd.read_html(io.StringIO(resp.read().decode("utf-8")))[0]

df_us = t_500[["Symbol", "Security", "GICS Sector"]].copy()
df_us.columns = ["ticker", "nombre", "sector"]
df_us["ticker"] = df_us["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()

# ------------------------------------------------------------------------------
# 1.2. EUROPA: EURO STOXX 50 (~50 gigantes europeos)
# ------------------------------------------------------------------------------
try:
    url_stoxx = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    req_stoxx = urllib.request.Request(url_stoxx, headers=headers)
    with urllib.request.urlopen(req_stoxx) as resp:
        tablas_stoxx = pd.read_html(io.StringIO(resp.read().decode("utf-8")))

    # La tabla de componentes suele ser la 3ª o 4ª en esta página
    t_stoxx = None
    for t in tablas_stoxx:
        if "Ticker" in t.columns and ("Sector" in t.columns or "Supersector" in t.columns):
            t_stoxx = t
            break

    if t_stoxx is not None:
        col_sec = "Sector" if "Sector" in t_stoxx.columns else "Supersector"
        col_nom = "Company" if "Company" in t_stoxx.columns else "Security"
        df_eu = t_stoxx[["Ticker", col_nom, col_sec]].copy()
        df_eu.columns = ["ticker", "nombre", "sector"]
        df_eu["ticker"] = df_eu["ticker"].astype(str).str.strip()
    else:
        df_eu = pd.DataFrame(columns=["ticker", "nombre", "sector"])
except Exception as e:
    print(f"Aviso al cargar Europa: {e}")
    df_eu = pd.DataFrame(columns=["ticker", "nombre", "sector"])

# ------------------------------------------------------------------------------
# 1.3. ASIA: Nikkei 225 de Japón (~225 activos de Tokio con sufijo .T)
# ------------------------------------------------------------------------------
try:
    url_nikkei = "https://en.wikipedia.org/wiki/Nikkei_225"
    req_nikkei = urllib.request.Request(url_nikkei, headers=headers)
    with urllib.request.urlopen(req_nikkei) as resp:
        tablas_nikkei = pd.read_html(io.StringIO(resp.read().decode("utf-8")))

    t_nikkei = None
    for t in tablas_nikkei:
        if "Symbol" in t.columns and "Sector" in t.columns:
            t_nikkei = t
            break

    if t_nikkei is not None:
        df_asia = t_nikkei[["Symbol", "Company", "Sector"]].copy()
        df_asia.columns = ["ticker", "nombre", "sector"]
        # Las acciones japonesas en Yahoo Finance usan código numérico + .T (ej. 7203.T para Toyota)
        df_asia["ticker"] = df_asia["ticker"].astype(str).str.strip() + ".T"
    else:
        df_asia = pd.DataFrame(columns=["ticker", "nombre", "sector"])
except Exception as e:
    print(f"Aviso al cargar Asia: {e}")
    df_asia = pd.DataFrame(columns=["ticker", "nombre", "sector"])

# ------------------------------------------------------------------------------
# 1.4. Consolidación Global del Catálogo Base
# ------------------------------------------------------------------------------
df_metadata = pd.concat([df_us, df_eu, df_asia], ignore_index=True)
df_metadata = df_metadata.drop_duplicates(subset=["ticker"]).set_index("ticker")

# Limpieza básica de nombres de sectores equivalentes
df_metadata["sector"] = df_metadata["sector"].replace({
    "Automobiles & Parts": "Consumer Discretionary",
    "Banks": "Financials",
    "Insurance": "Financials",
    "Chemicals": "Materials",
    "Basic Resources": "Materials",
    "Food, Beverage & Tobacco": "Consumer Staples",
    "Personal Care, Drug & Grocery Stores": "Consumer Staples",
    "Health Care": "Health Care",
    "Industrial Goods & Services": "Industrials",
    "Technology": "Information Technology",
    "Telecommunications": "Communication Services",
    "Utilities": "Utilities",
    "Energy": "Energy",
    "Real Estate": "Real Estate"
})

print(f"-> Catálogo global listo: {len(df_metadata)} activos para descargar.")
# ==============================================================================
# 2. DESCARGA A 3.5 AÑOS (~882 RUEDAS) EN LOTES CONTROLADOS
# ==============================================================================
fecha_fin = pd.Timestamp.today()
fecha_inicio = fecha_fin - pd.DateOffset(days=int(3.5 * 365.25))

inicio_str = fecha_inicio.strftime("%Y-%m-%d")
fin_str = fecha_fin.strftime("%Y-%m-%d")

print(f"2. Descargando cotizaciones desde {inicio_str} hasta {fin_str}...")

tickers_totales = df_metadata.index.tolist()
chunk_size = 100
precios_list = []
volumenes_list = []

for i in range(0, len(tickers_totales), chunk_size):
    lote = tickers_totales[i : i + chunk_size]
    try:
        data = yf.download(
            tickers=lote,
            start=inicio_str,
            end=fin_str,
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if not data.empty and "Close" in data and "Volume" in data:
            precios_list.append(data["Close"])
            volumenes_list.append(data["Volume"])
    except Exception as e:
        print(f"Aviso en lote {i}-{i + chunk_size}: {e}")
    time.sleep(0.3)

df_precios_raw = pd.concat(precios_list, axis=1)
df_volumenes_raw = pd.concat(volumenes_list, axis=1)

df_precios_raw = df_precios_raw.loc[:, ~df_precios_raw.columns.duplicated()]
df_volumenes_raw = df_volumenes_raw.loc[:, ~df_volumenes_raw.columns.duplicated()]

print(f"-> Descarga finalizada: {df_precios_raw.shape[1]} activos recibidos.")

# ==============================================================================
# 3. PROTOCOLO DE CRIBA MEJORADA (HISTORIAL + ADTV + SHARPE/MOMENTUM)
# ==============================================================================
print("3. Aplicando protocolo de criba a 100 activos...")

comunes = df_precios_raw.columns.intersection(df_volumenes_raw.columns)
p = df_precios_raw[comunes].copy()
v = df_volumenes_raw[comunes].copy()

# 3.1. Continuidad temporal a 3.5 años (descarta IPOs o series incompletas)
min_trading_days = 840
historial_valido = p.notna().sum(axis=0) >= min_trading_days
p = p.loc[:, historial_valido]
v = v.loc[:, historial_valido]

p = p.ffill(limit=2).dropna(how="any", axis=0)
v = v.loc[p.index].fillna(0.0)

# 3.2. Filtro de liquidez reciente (ADTV a 6 meses >= 10M USD)
liquidity_window = 126
min_adtv = 10_000_000.0

p_reciente = p.iloc[-liquidity_window:]
v_reciente = v.iloc[-liquidity_window:]
adtv = (p_reciente * v_reciente).mean(axis=0)

tickers_liquidos = adtv[adtv >= min_adtv].index

df_candidatos = (
    pd.DataFrame({"adtv": adtv[tickers_liquidos]})
    .join(df_metadata[["nombre", "sector"]], how="inner")
    .dropna(subset=["sector"])
)

# Preselección de los 20 más líquidos por sector
df_top_liquidos = (
    df_candidatos.sort_values(by=["sector", "adtv"], ascending=[True, False])
    .groupby("sector")
    .head(20)
    .copy()
)

# 3.3. Ratio Sharpe / Momentum a 1 año (252 días de trading)
p_momentum = p[df_top_liquidos.index].iloc[-252:]
retornos_1y = p_momentum.pct_change().dropna(how="any")

retorno_anual = retornos_1y.mean() * 252
volatilidad_anual = retornos_1y.std() * np.sqrt(252)
sharpe_ratio = retorno_anual / (volatilidad_anual + 1e-8)

df_top_liquidos["sharpe_1y"] = sharpe_ratio
df_top_liquidos["retorno_1y"] = retorno_anual
df_top_liquidos["volatilidad_1y"] = volatilidad_anual

# 3.4. Desduplicación de Clases de Acciones y Selección Final
# -------------------------------------------------------------
# 3.4.1. Crear un identificador de empresa limpio para detectar clases duales
# (Ej: 'Alphabet Inc. (Class A)' y 'Alphabet Inc. (Class C)' -> 'Alphabet Inc')
# (Ej: 'Fox Corporation (Class A)' y 'Fox Corporation (Class B)' -> 'Fox Corporation')
df_top_liquidos["empresa_raiz"] = (
    df_top_liquidos["nombre"]
    .str.replace(r"\(Class [A-Z]\)", "", regex=True)
    .str.replace(r"Class [A-Z]", "", regex=True)
    .str.strip()
)

# 3.4.2. Ordenar por sector y mejor Sharpe
df_sorted = df_top_liquidos.sort_values(
    by=["sector", "sharpe_1y"], ascending=[True, False]
)

# 3.4.3. Desduplicar empresas repetidas (si una empresa tiene 2 clases, dejamos la de mejor Sharpe)
df_unicas = df_sorted.drop_duplicates(
    subset=["sector", "empresa_raiz"], keep="first"
).copy()

# 3.4.4. Seleccionar los 9 líderes por sector
universo_final = df_unicas.groupby("sector").head(9).copy()

# 3.4.5. Rellenar hasta alcanzar exactamente los 100 activos con los mejores Sharpe restantes (sin duplicados)
if len(universo_final) < 100:
  restantes = df_unicas.loc[
      ~df_unicas.index.isin(universo_final.index)
  ].sort_values(by="sharpe_1y", ascending=False)
  faltantes = 100 - len(universo_final)
  universo_final = pd.concat([universo_final, restantes.head(faltantes)])

# 3.4.6. Recomputar ranking sectorial
universo_final["ranking_sectorial"] = (
    universo_final.groupby("sector")["sharpe_1y"]
    .rank(ascending=False, method="first")
    .astype(int)
)

# ==============================================================================
# 4. MATRIZ DE PRECIOS LIMPIA (T x 100) LISTA PARA CLUSTERING
# ==============================================================================
tickers_seleccionados = universo_final.index.tolist()
df_precios_100 = df_precios_raw[tickers_seleccionados].dropna(how="any", axis=0)

print("\n" + "=" * 60)
print(f"CRIBA COMPLETADA: {len(tickers_seleccionados)} ACTIVOS SELECCIONADOS")
print(f"Matriz de precios: {df_precios_100.shape[0]} sesiones x {df_precios_100.shape[1]} activos")
print("=" * 60)

pd.set_option("display.max_rows", 115)
columnas_ver = ["nombre", "sector", "ranking_sectorial", "sharpe_1y", "retorno_1y", "adtv"]
display(universo_final[columnas_ver].sort_values(by=["sector", "ranking_sectorial"]))

# 1. La variable como lista de Python pura con los 100 tickers
tickers_seleccionados = universo_final.index.tolist()

# 2. Imprimir la lista en pantalla
print(f"Total tickers: {len(tickers_seleccionados)}")
print(tickers_seleccionados)

# 3. Si prefieres tener un DataFrame exclusivo solo con los tickers y su nombre
df_tickers = universo_final[["nombre", "sector"]].reset_index()
df_tickers.columns = ["ticker", "nombre", "sector"]
display(df_tickers)

# 1. Calcular los retornos simples directamente sobre la matriz de 100 precios ya descargada
returns = df_precios_100.pct_change().dropna(how="any")

# 2. Comprobar dimensiones (debe salir 867 sesiones x 100 activos)
print(f"Retornos listos: {returns.shape[0]} sesiones x {returns.shape[1]} activos.")

print(df_tickers)

tickers_lista = df_tickers["ticker"].tolist()
print(tickers_lista)
