import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from collections import defaultdict
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES, ANIMAL_A_NUM_INT

CONFIG = {
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'logger_name': 'comp',
    'max_numero': 37,
}

EXCEL_FILES = [
    'data/LottoActivoINT.xlsx',
    'data/LottoActivoRDInt.xlsx',
    'data/SelvaPlus.xlsx',
    'data/LaGranjita.xlsx',
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

def build_sources(d_train):
    loteria = Loteria(CONFIG)
    trans_prob, trans_total = loteria._transiciones_markov(d_train)
    dp = loteria.preparar_datos_markov(d_train)
    _, _, trans_h, total_h = loteria.construir_matrices_markov(dp, incluir_trasnocho=False)

    freq_hora = {}
    for hora in d_train['Solo_hora'].unique():
        sub = d_train[d_train['Solo_hora'] == hora]
        freq_hora[hora] = sub['Num_Int'].value_counts(normalize=True).mul(100)

    cooc = defaultdict(lambda: defaultdict(int))
    cooc_tot = defaultdict(int)
    for _, g in d_train.groupby('Fecha'):
        nums = set(g['Num_Int'].unique())
        for n in nums:
            cooc_tot[n] += 1
            for n2 in nums:
                if n2 != n:
                    cooc[n][n2] += 1

    return trans_prob, trans_total, trans_h, total_h, freq_hora, cooc, cooc_tot

def fill_mh_scores(ultimo_num, trans_prob, trans_total, mh_scores, top_k):
    if len(mh_scores) >= top_k:
        return
    if ultimo_num not in trans_total or trans_total[ultimo_num] <= 0:
        return
    candidates = [(n2, trans_prob.get((ultimo_num, n2), 0)) for n2 in range(38)
                  if n2 not in mh_scores and trans_prob.get((ultimo_num, n2), 0) > 0]
    candidates.sort(key=lambda x: x[1], reverse=True)
    for n2, p in candidates:
        mh_scores[n2] = p
        if len(mh_scores) >= top_k:
            break

def eval_transition(na, ns, hp, hn, trans_prob, trans_total, trans_h, total_h, freq_hora, cooc, cooc_tot, top_k):
    par = (hp, hn)
    h12 = pd.to_datetime(hn, format='%H:%M:%S').strftime('%I:%M %p')

    gs = {}
    if na in trans_total and trans_total[na] > 0:
        for n in range(38):
            p = trans_prob.get((na, n), 0)
            if p > 0:
                gs[n] = p

    hs = {}
    if par in trans_h and na in total_h.get(par, {}):
        tot_hh = total_h[par][na]
        if tot_hh > 0:
            for n2, cnt in trans_h[par][na].items():
                hs[n2] = cnt / tot_hh * 100
    fill_mh_scores(na, trans_prob, trans_total, hs, top_k)

    fs = freq_hora.get(h12, pd.Series(dtype=float))

    cs = {}
    if cooc_tot.get(na, 0) > 0:
        cs = {n: c / cooc_tot[na] * 100 for n, c in cooc[na].items()}

    m_hh = total_h.get(par, {}).get(na, 0)
    w_hh = min(0.9, max(0.1, m_hh / 50))

    # GxH
    todos = set(gs) | set(hs)
    gxh = {}
    for n2 in todos:
        gxh[n2] = gs.get(n2, 0) * (1 - w_hh) + hs.get(n2, 0) * w_hh

    # Full
    w_h = w_hh
    w_g = 0.40 - w_h * 0.5
    w_f = 0.20
    w_c = 0.40 - w_h * 0.5
    full = {}
    for n2 in range(38):
        if n2 == na:
            continue
        full[n2] = gs.get(n2, 0) * w_g + hs.get(n2, 0) * w_h + fs.get(n2, 0) * w_f + cs.get(n2, 0) * w_c

    g_top = set(sorted(gs, key=gs.get, reverse=True)[:top_k])
    h_top = set(sorted(hs, key=hs.get, reverse=True)[:top_k])
    f_top = set(fs.head(top_k).index)
    c_top = set(sorted(cs, key=cs.get, reverse=True)[:top_k])
    gxh_top = set(sorted(gxh, key=gxh.get, reverse=True)[:top_k])
    full_top = set(sorted(full, key=full.get, reverse=True)[:top_k])

    return {
        'G': ns in g_top, 'H': ns in h_top, 'F': ns in f_top,
        'C': ns in c_top, 'GxH': ns in gxh_top, 'Full': ns in full_top
    }

KS = [25, 10, 5]

print(f"\n{'='*100}")
print(f"  WALK-FORWARD: FULL vs GxH vs F vs C (70/30)")
print(f"{'='*100}")

all_results = {}
for fname in EXCEL_FILES:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    nombre = fname.replace('data/', '').replace('.xlsx', '')
    
    datos = load_data(path)
    fechas = sorted(datos['Fecha'].unique())
    split = int(len(fechas) * 0.7)
    fechas_train = set(fechas[:split])
    fechas_test = fechas[split:]

    d_train = datos[datos['Fecha'].isin(fechas_train)]
    trans_prob, trans_total, trans_h, total_h, freq_hora, cooc, cooc_tot = build_sources(d_train)

    hits = {k: defaultdict(int) for k in KS}
    total_trans = 0

    for fecha, grupo in datos[datos['Fecha'].isin(fechas_test)].groupby('Fecha'):
        grupo = grupo.sort_values('Hora')
        for i in range(1, len(grupo)):
            ant = int(grupo.iloc[i-1]['Num_Int'])
            sig = int(grupo.iloc[i]['Num_Int'])
            hp = grupo.iloc[i-1]['Hora']
            hn = grupo.iloc[i]['Hora']

            for k in KS:
                res = eval_transition(ant, sig, hp, hn, trans_prob, trans_total,
                                       trans_h, total_h, freq_hora, cooc, cooc_tot, k)
                for model in ['G', 'H', 'F', 'C', 'GxH', 'Full']:
                    if res[model]:
                        hits[k][model] += 1
            total_trans += 1

    all_results[nombre] = {'hits': hits, 'total': total_trans}

    print(f"\n{'─'*100}")
    print(f"  {nombre}  ({total_trans} transiciones en test)")
    print(f"{'─'*100}")
    header = f"  {'k':>4} | {'G':>8} {'H':>8} {'F':>8} {'C':>8} | {'GxH':>8} {'Full':>8} | {'Azar':>8}"
    print(f"\n{header}")
    print(f"  {'─'*len(header)}")
    for k in KS:
        vals = []
        for model in ['G', 'H', 'F', 'C', 'GxH', 'Full']:
            pct = hits[k][model] / total_trans * 100 if total_trans else 0
            vals.append(f"{pct:>6.1f}%")
        azar = k / 38 * 100
        print(f"  k={k:>2}  | {' '.join(vals)} | {azar:>6.1f}%")

# Summary
print(f"\n\n{'='*100}")
print(f"  TABLA RESUMEN: MEJOR % POR TOP-k")
print(f"{'='*100}")
print(f"  {'Lotería':<24} {'Top-25':<24} {'Top-10':<24} {'Top-5':<24}")
print(f"  {'─'*96}")
for nombre, res in all_results.items():
    t = res['total']
    rows = []
    for k in KS:
        best_model = max(['G', 'H', 'F', 'C', 'GxH', 'Full'],
                        key=lambda m: res['hits'][k][m])
        best_pct = res['hits'][k][best_model] / t * 100 if t else 0
        azar = k / 38 * 100
        rows.append(f"{best_model:>4} {best_pct:>5.1f}% (azar {azar:.0f}%)")
    print(f"  {nombre:<24} {rows[0]:<24} {rows[1]:<24} {rows[2]:<24}")

# Full vs GxH detail
print(f"\n\n{'='*100}")
print(f"  FULL vs GxH: donde gana cada uno")
print(f"{'='*100}")
print(f"  {'Lotería':<24} {'Top-25':<30} {'Top-10':<30} {'Top-5':<30}")
print(f"  {'─'*96}")
for nombre, res in all_results.items():
    t = res['total']
    rows = []
    for k in KS:
        full = res['hits'][k]['Full'] / t * 100 if t else 0
        gxh = res['hits'][k]['GxH'] / t * 100 if t else 0
        diff = full - gxh
        rows.append(f"Full {full:>5.1f}%  GxH {gxh:>5.1f}%  Δ {diff:+.1f}%")
    print(f"  {nombre:<24} {rows[0]:<30} {rows[1]:<30} {rows[2]:<30}")
