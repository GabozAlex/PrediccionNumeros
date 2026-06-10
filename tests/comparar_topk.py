import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from collections import defaultdict
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES, ANIMAL_A_NUM_INT
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

CONFIG = {
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'logger_name': 'comp',
    'max_numero': 37,
    'modelos_dir': '/tmp/sim_modelos',
}

EXCEL_FILES = [
    'data/LottoActivoINT.xlsx',
    'data/LottoActivoRDInt.xlsx',
    'data/SelvaPlus.xlsx',
    'data/LaGranjita.xlsx',
]

FEATURES = [
    'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num',
    'Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
    'Frecuencia_10', 'Sorteos_Desde_Aparicion',
    'Color_Previo', 'Paridad_Previo',
    'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N', 'Prob_Cooc', 'Lift_Dia'
]

def load_data(path):
    datos = pd.read_excel(path)
    datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
    datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce')
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
    datos['Hora'] = datos['Hora'].astype(str).str.strip().str.zfill(8)
    datos['Timestamp'] = pd.to_datetime(datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce')
    datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
    datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip()
    datos = datos.sort_values(by='Timestamp').reset_index(drop=True)
    datos['Num_Int'] = datos['Animal'].map(ANIMAL_A_NUM_INT)
    datos = datos.dropna(subset=['Num_Int']).reset_index(drop=True)
    datos['Num_Int'] = datos['Num_Int'].astype(int)
    return datos

def eval_analytical_k(loteria, d, fechas_train, fechas_test, ks):
    d_train = d[d['Fecha'].isin(fechas_train)]
    dp_train = loteria.preparar_datos_markov(d_train)
    trans_g, total_g, _, _ = loteria.construir_matrices_markov(dp_train)

    # Frecuencia hora
    freq_hora = {}
    for hora in d_train['Solo_hora'].unique():
        sub = d_train[d_train['Solo_hora'] == hora]
        freq_hora[hora] = sub['Num_Int'].value_counts(normalize=True).mul(100)

    # Co-ocurrencia
    cooc = defaultdict(lambda: defaultdict(int))
    cooc_tot = defaultdict(int)
    for _, g in d_train.groupby('Fecha'):
        nums = set(g['Num_Int'].unique())
        for n in nums:
            cooc_tot[n] += 1
            for n2 in nums:
                if n2 != n:
                    cooc[n][n2] += 1

    hits = {k: defaultdict(int) for k in ks}
    total = 0
    for _, grupo in d[d['Fecha'].isin(fechas_test)].groupby('Fecha'):
        grupo = grupo.sort_values('Hora')
        for i in range(1, len(grupo)):
            ant = int(grupo.iloc[i-1]['Num_Int'])
            sig = int(grupo.iloc[i]['Num_Int'])
            hp = grupo.iloc[i-1]['Hora']
            total += 1

            # G
            g_dict = {}
            if ant in total_g and total_g[ant] > 0:
                t_ = total_g[ant]
                g_dict = {n: c / t_ * 100 for n, c in trans_g[ant].items()}
            g_sorted = sorted(g_dict, key=g_dict.get, reverse=True)

            # F
            h12 = pd.to_datetime(hp, format='%H:%M:%S').strftime('%I:%M %p')
            f_series = freq_hora.get(h12, pd.Series(dtype=float))
            f_sorted = list(f_series.sort_values(ascending=False).index)

            # C
            c_dict = {}
            if cooc_tot.get(ant, 0) > 0:
                c_dict = {n: c / cooc_tot[ant] * 100 for n, c in cooc[ant].items()}
            c_sorted = sorted(c_dict, key=c_dict.get, reverse=True)

            for k in ks:
                if sig in set(g_sorted[:k]): hits[k]['G'] += 1
                if sig in set(f_sorted[:k]): hits[k]['F'] += 1
                if sig in set(c_sorted[:k]): hits[k]['C'] += 1
    return total, hits

def eval_ml_k(name, clf, df_feat, fechas_train, fechas_test, ks):
    available = [f for f in FEATURES if f in df_feat.columns]
    train = df_feat[df_feat['Fecha'].isin(fechas_train)].copy()
    test = df_feat[df_feat['Fecha'].isin(fechas_test)].copy()
    if 'Hora_Sorteo' not in train.columns:
        train['Hora_Sorteo'] = train['Hora'].astype(str).str.strip().str.zfill(8)
        test['Hora_Sorteo'] = test['Hora'].astype(str).str.strip().str.zfill(8)
    train = train.dropna(subset=available + ['Hora_Sorteo', 'Num_Int'])
    test = test.dropna(subset=available + ['Hora_Sorteo', 'Num_Int'])

    train_classes = set(train['Num_Int'].unique())
    test = test[test['Num_Int'].isin(train_classes)]

    X_train = train[available + ['Hora_Sorteo']]
    y_train = train['Num_Int']
    X_test = test[available + ['Hora_Sorteo']]
    y_test = test['Num_Int']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Hora_Sorteo']),
        ('num', 'passthrough', available)
    ])
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)

    classes = pipe.classes_
    y_proba = pipe.predict_proba(X_test)
    hits = {}
    for k in ks:
        topk_idx = np.argsort(y_proba, axis=1)[:, -k:][:, ::-1]
        hits[k] = sum(1 for i in range(len(y_test)) if y_test.iloc[i] in classes[topk_idx[i]])
    return len(y_test), hits

KS = [25, 10, 5]
print(f"\n{'='*100}")
print(f"  COMPARACION TOP-k: ANALITICOS vs ML (walk-forward 70/30)")
print(f"{'='*100}")

for fname in EXCEL_FILES:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    print(f"\n{'─'*100}")
    print(f"  {fname}")
    print(f"{'─'*100}")

    loteria = Loteria(CONFIG)
    datos = load_data(path)
    df_feat = loteria.agregar_caracteristicas_avanzadas(datos)

    fechas = sorted(df_feat['Fecha'].unique())
    split = int(len(fechas) * 0.7)
    fechas_train = set(fechas[:split])
    fechas_test = fechas[split:]

    total_analitico, hits_analitico = eval_analytical_k(loteria, datos, fechas_train, fechas_test, KS)

    rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    total_rf, hits_rf = eval_ml_k('RF', rf, df_feat, fechas_train, fechas_test, KS)

    xgb = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=42, n_jobs=-1, verbosity=0)
    try:
        total_xgb, hits_xgb = eval_ml_k('XGB', xgb, df_feat, fechas_train, fechas_test, KS)
    except Exception as e:
        print(f"  XGBoost skipped ({e})")
        total_xgb, hits_xgb = 0, {}

    header = f"  {'k':>4} | {'G':>8} {'F':>8} {'C':>8} | {'RF':>8} {'XGB':>8}"
    print(f"\n{header}")
    print(f"  {'─'*len(header)}")
    for k in KS:
        ga = hits_analitico[k]['G'] / total_analitico * 100
        fa = hits_analitico[k]['F'] / total_analitico * 100
        ca = hits_analitico[k]['C'] / total_analitico * 100
        rf_pct = hits_rf[k] / total_rf * 100 if total_rf else 0
        xgb_pct = hits_xgb[k] / total_xgb * 100 if total_xgb else 0
        aleatorio = k / 38 * 100
        print(f"  k={k:>2}  | {ga:>6.1f}% {fa:>6.1f}% {ca:>6.1f}% | {rf_pct:>6.1f}% {xgb_pct:>6.1f}%  (azar: {aleatorio:.0f}%)")

# Summary
print(f"\n\n{'='*100}")
print(f"  RESUMEN: MEJOR MODELO POR LOTERIA Y TOP-k")
print(f"{'='*100}")
print(f"  {'Lotería':<24} {'':>10} {'Top-25':>10} {'Top-10':>10} {'Top-5':>10}")
print(f"  {'─'*64}")
for fname in EXCEL_FILES:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    loteria = Loteria(CONFIG)
    datos = load_data(path)
    df_feat = loteria.agregar_caracteristicas_avanzadas(datos)
    fechas = sorted(df_feat['Fecha'].unique())
    split = int(len(fechas) * 0.7)
    fechas_train = set(fechas[:split])
    fechas_test = fechas[split:]
    total_a, hits_a = eval_analytical_k(loteria, datos, fechas_train, fechas_test, KS)
    total_rf, hits_rf = eval_ml_k('RF', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1), df_feat, fechas_train, fechas_test, KS)
    best = {}
    for k in KS:
        analitico_best = max(hits_a[k]['G'], hits_a[k]['F'], hits_a[k]['C'])
        rf_k = hits_rf[k]
        best[k] = ('RF', rf_k) if rf_k >= analitico_best else ('G/F/C', analitico_best)
    print(f"  {fname:<24} {'':>10} {best[25][0]:>4} {best[25][1]/total_rf*100:>5.1f}% {best[10][0]:>4} {best[10][1]/total_rf*100:>5.1f}% {best[5][0]:>4} {best[5][1]/total_rf*100:>5.1f}%")
