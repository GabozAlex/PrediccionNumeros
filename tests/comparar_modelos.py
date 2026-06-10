import sys, os, time
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

def eval_analytical(loteria, d, fechas_train, fechas_test, top_k=25):
    """Evalua estrategias analiticas (como comparar_estrategias pero devuelve dict)."""
    d_train = d[d['Fecha'].isin(fechas_train)]
    dp_train = loteria.preparar_datos_markov(d_train)
    trans_g, total_g, trans_h, total_h = loteria.construir_matrices_markov(dp_train)

    freq_por_hora = {}
    for hora in d_train['Solo_hora'].unique():
        sub = d_train[d_train['Solo_hora'] == hora]
        freq_por_hora[hora] = sub['Num_Int'].value_counts(normalize=True).mul(100)

    cooc_global = defaultdict(lambda: defaultdict(int))
    cooc_total_global = defaultdict(int)
    for _, g in d_train.groupby('Fecha'):
        nums = set(g['Num_Int'].unique())
        for n in nums:
            cooc_total_global[n] += 1
            for n2 in nums:
                if n2 != n:
                    cooc_global[n][n2] += 1

    hits = defaultdict(int)
    total = 0
    for fecha, grupo in d[d['Fecha'].isin(fechas_test)].groupby('Fecha'):
        grupo = grupo.sort_values('Hora')
        for i in range(1, len(grupo)):
            ant = int(grupo.iloc[i-1]['Num_Int'])
            sig = int(grupo.iloc[i]['Num_Int'])
            hp = grupo.iloc[i-1]['Hora']

            g_dict = {}
            if ant in total_g and total_g[ant] > 0:
                t = total_g[ant]
                g_dict = {n: c / t * 100 for n, c in trans_g[ant].items()}
            g_top = set(sorted(g_dict, key=g_dict.get, reverse=True)[:top_k])

            h12 = pd.to_datetime(hp, format='%H:%M:%S').strftime('%I:%M %p')
            f_series = freq_por_hora.get(h12, pd.Series(dtype=float))
            f_top = set(f_series.head(top_k).index)

            c_dict = {n: c / cooc_total_global[ant] * 100 for n, c in cooc_global[ant].items()} if cooc_total_global.get(ant, 0) > 0 else {}
            c_top = set(sorted(c_dict, key=c_dict.get, reverse=True)[:top_k])

            total += 1
            if sig in g_top: hits['G'] += 1
            if sig in f_top: hits['F'] += 1
            if sig in c_top: hits['C'] += 1
    hits['total'] = total
    return hits

def eval_ml(name, clf, df_feat, fechas_train, fechas_test, top_k=25):
    available = [f for f in FEATURES if f in df_feat.columns]
    train = df_feat[df_feat['Fecha'].isin(fechas_train)].copy()
    test = df_feat[df_feat['Fecha'].isin(fechas_test)].copy()

    if 'Hora_Sorteo' not in train.columns:
        train['Hora_Sorteo'] = train['Hora'].astype(str).str.strip().str.zfill(8)
        test['Hora_Sorteo'] = test['Hora'].astype(str).str.strip().str.zfill(8)

    train = train.dropna(subset=available + ['Hora_Sorteo', 'Num_Int'])
    test = test.dropna(subset=available + ['Hora_Sorteo', 'Num_Int'])

    X_train = train[available + ['Hora_Sorteo']]
    y_train = train['Num_Int']
    X_test = test[available + ['Hora_Sorteo']]
    y_test = test['Num_Int']

    t0 = time.time()
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Hora_Sorteo']),
        ('num', 'passthrough', available)
    ])
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    if hasattr(clf, 'classes_'):
        pipe.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train, clf__classes=np.arange(38))
    t_elapsed = time.time() - t0

    classes = pipe.classes_
    y_proba = pipe.predict_proba(X_test)
    topk_idx = np.argsort(y_proba, axis=1)[:, -top_k:][:, ::-1]
    hits = sum(1 for i in range(len(y_test)) if y_test.iloc[i] in classes[topk_idx[i]])

    print(f"    {name}: {hits}/{len(y_test)} = {hits/len(y_test)*100:.1f}% ({t_elapsed:.1f}s)")
    return {'hits': hits, 'total': len(y_test), 'time': t_elapsed, 'name': name}

print(f"\n{'='*90}")
print(f"  COMPARACION COMPLETA: MODELOS ANALITICOS vs PREDICTIVOS (Top-25, 70/30 walk-forward)")
print(f"{'='*90}")

all_results = []
for fname in EXCEL_FILES:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    if not os.path.exists(path):
        continue

    print(f"\n{'─'*90}")
    print(f"  {fname}")
    print(f"{'─'*90}")

    loteria = Loteria(CONFIG)
    datos = load_data(path)
    df_feat = loteria.agregar_caracteristicas_avanzadas(datos)

    fechas = sorted(df_feat['Fecha'].unique())
    split = int(len(fechas) * 0.7)
    fechas_train = set(fechas[:split])
    fechas_test = fechas[split:]
    print(f"  Train: {len(fechas_train)} dias, Test: {len(fechas_test)} dias, Registros: {len(datos)}")

    # --- Analiticos ---
    print(f"\n  MODELOS ANALITICOS:")
    a = eval_analytical(loteria, datos, fechas_train, fechas_test, top_k=25)
    t = a['total']
    for k in ['G', 'F', 'C']:
        pct = a[k] / t * 100 if t > 0 else 0
        print(f"    {k:>4}: {a[k]:>4}/{t} = {pct:.1f}%")

    # --- RF ---
    print(f"\n  MODELOS PREDICTIVOS (18 features):")
    rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    r = eval_ml('RF', rf, df_feat, fechas_train, fechas_test)

    xgb = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=42, n_jobs=-1, verbosity=0)
    x = eval_ml('XGBoost', xgb, df_feat, fechas_train, fechas_test)

    all_results.append({
        'file': fname,
        'total': t,
        'analitico_g': a['G'], 'analitico_f': a['F'], 'analitico_c': a['C'],
        'rf_hits': r['hits'], 'rf_total': r['total'],
        'xgb_hits': x['hits'], 'xgb_total': x['total'],
    })

# Summary
print(f"\n\n{'='*90}")
print(f"  TABLA COMPARATIVA FINAL (Top-25)")
print(f"{'='*90}")
print(f"  {'Archivo':<24} {'G(Markov)':>10} {'F(Frec)':>10} {'C(Cooc)':>10} {'RF':>10} {'XGBoost':>10}")
print(f"  {'─'*74}")
for r in all_results:
    t = r['total']
    print(f"  {r['file']:<24} {r['analitico_g']/t*100:>8.1f}%  {r['analitico_f']/t*100:>8.1f}%  {r['analitico_c']/t*100:>8.1f}%  {r['rf_hits']/r['rf_total']*100:>8.1f}%  {r['xgb_hits']/r['xgb_total']*100:>8.1f}%")
