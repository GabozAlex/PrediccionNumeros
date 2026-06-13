import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from collections import defaultdict
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES, ANIMAL_A_NUM_INT

CONFIG = {
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'logger_name': 'sim',
    'max_numero': 37,
    'modelos_dir': '/tmp/sim_modelos',
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

def evaluar_walk_forward(loteria, datos, top_k=38, train_pct=0.7):
    fechas_completas = datos.groupby('Fecha').size()
    fechas_completas = fechas_completas[fechas_completas == 12].index
    d = datos[datos['Fecha'].isin(fechas_completas)].copy()
    parejas = loteria.get_parejas_horarias()
    
    fechas_ordenadas = sorted(d['Fecha'].unique())
    split_idx = int(len(fechas_ordenadas) * train_pct)
    fechas_train = set(fechas_ordenadas[:split_idx])
    fechas_test = fechas_ordenadas[split_idx:]
    
    print(f"  Train: {len(fechas_train)} dias, Test: {len(fechas_test)} dias")
    
    d_train = d[d['Fecha'].isin(fechas_train)]
    dp_train = loteria.preparar_datos_markov(d_train)
    trans_g, total_g, trans_h, total_h = loteria.construir_matrices_markov(dp_train, incluir_trasnocho=False)
    
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
    
    resultados = defaultdict(lambda: {'total': 0, 'g': 0, 'h': 0, 'f': 0, 'c': 0, 'gxh': 0, 'full': 0})
    
    for fecha, grupo in d[d['Fecha'].isin(fechas_test)].groupby('Fecha'):
        grupo = grupo.sort_values('Hora')
        for i in range(len(parejas)):
            o, dest = parejas[i]
            ant = int(grupo.iloc[i]['Num_Int'])
            sig = int(grupo.iloc[i+1]['Num_Int'])
            
            g_dict = {}
            if ant in total_g and total_g[ant] > 0:
                t = total_g[ant]
                g_dict = {n: c / t * 100 for n, c in trans_g[ant].items()}
            g_top = set(sorted(g_dict, key=g_dict.get, reverse=True)[:top_k])
            
            par = (o, dest)
            h_dict = {}
            if ant in total_h.get(par, {}) and total_h[par][ant] > 0:
                t = total_h[par][ant]
                h_dict = {n: c / t * 100 for n, c in trans_h[par][ant].items()}
            h_top = set(sorted(h_dict, key=h_dict.get, reverse=True)[:top_k])
            
            h12 = pd.to_datetime(o, format='%H:%M:%S').strftime('%I:%M %p')
            f_series = freq_por_hora.get(h12, pd.Series(dtype=float))
            f_top = set(f_series.head(top_k).index)
            
            c_dict = {n: c / cooc_total_global[ant] * 100 for n, c in cooc_global[ant].items()} if cooc_total_global.get(ant, 0) > 0 else {}
            c_top = set(sorted(c_dict, key=c_dict.get, reverse=True)[:top_k])
            
            gxh_scores = {}
            for n in range(38):
                pg = g_dict.get(n, 0)
                ph = h_dict.get(n, 0)
                if pg > 0 or ph > 0:
                    gxh_scores[n] = pg + ph
            gxh_top = set(sorted(gxh_scores, key=gxh_scores.get, reverse=True)[:top_k])
            
            w_h = min(0.35, max(0.05, total_h.get(par, {}).get(ant, 0) / 100 * 0.35))
            w_g = 0.40 - w_h * 0.5
            w_f = 0.20
            w_c = 0.40 - w_h * 0.5
            full_scores = {}
            for n in range(38):
                if n == ant:
                    continue
                pg = g_dict.get(n, 0)
                ph = h_dict.get(n, 0)
                pf = f_series.get(n, 0)
                pc = c_dict.get(n, 0)
                full_scores[n] = pg * w_g + ph * w_h + pf * w_f + pc * w_c
            full_top = set(sorted(full_scores, key=full_scores.get, reverse=True)[:top_k])
            
            resultados[par]['total'] += 1
            if sig in g_top: resultados[par]['g'] += 1
            if sig in h_top: resultados[par]['h'] += 1
            if sig in f_top: resultados[par]['f'] += 1
            if sig in c_top: resultados[par]['c'] += 1
            if sig in gxh_top: resultados[par]['gxh'] += 1
            if sig in full_top: resultados[par]['full'] += 1
    
    totals = {k: sum(r[k] for r in resultados.values()) for k in ['g','h','f','c','gxh','full']}
    totals['total'] = sum(r['total'] for r in resultados.values())
    return totals

print(f"\n{'='*80}")
print(f"  WALK-FORWARD SIMULATION (70% train, 30% test, Top-25)")
print(f"{'='*80}")

for fname in EXCEL_FILES:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    print(f"\n{'─'*80}")
    print(f"  {fname}")
    print(f"{'─'*80}")
    
    if not os.path.exists(path):
        print(f"  NO ENCONTRADO")
        continue
    
    loteria = Loteria(CONFIG)
    datos = load_data(path)
    print(f"  Registros: {len(datos)}")
    
    totals = evaluar_walk_forward(loteria, datos, top_k=38, train_pct=0.7)
    t = totals['total']
    print(f"\n  {'Estrategia':<20} {'Aciertos':>10} {'Total':>8} {'%':>8}")
    print(f"  {'─'*48}")
    for key, label in [('g','G (Markov global)'),('h','H (Markov x hora)'),
                       ('f','F (Frec. hora)'),('c','C (Co-ocurrencia)'),
                       ('gxh','GxH (Global x Hora)'),('full','Full (4 fuentes)')]:
        pct = totals[key] / t * 100 if t > 0 else 0
        print(f"  {label:<20} {totals[key]:>10} {t:>8} {pct:>7.1f}%")
