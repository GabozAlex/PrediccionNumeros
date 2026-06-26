#!/usr/bin/env python3
"""
Evalúa modelos Markov / MkHora / GxHora / Markov x Dia / FrecHora / RF / XGB / RF+XGB
calculando precisión Top-5 y Top-10 y desgloses por las 12 horas.

Genera salida en pantalla y guarda results_topk.json
"""
from lotto_activo import analizador, CONFIG
from utils import load_and_prepare_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np, pandas as pd, json


def build_transitions(df, X_all, fechas_test):
    idxs = list(df.index)
    transitions = []
    for pos in range(1, len(idxs)):
        prev_idx = idxs[pos - 1]
        cur_idx = idxs[pos]
        if df.loc[prev_idx, 'Fecha'] != df.loc[cur_idx, 'Fecha']:
            continue
        if df.loc[cur_idx, 'Fecha'].date() not in fechas_test:
            continue
        if prev_idx not in X_all.index or cur_idx not in X_all.index:
            continue
        transitions.append((prev_idx, cur_idx))
    return transitions


def topk_from_scores(scores_dict, k):
    # scores_dict: mapping num->score
    top = sorted(scores_dict, key=scores_dict.get, reverse=True)[:k]
    return set(top)


def main():
    print('Cargando datos...')
    df = load_and_prepare_data(CONFIG['excel_file'], analizador)
    df = analizador.agregar_caracteristicas_avanzadas(df)
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])

    X_all, Y_all, le_y, numeric_features, categorical_features, available_features = analizador.preparar_datos_ml_completo(df)
    fecha_series = df.loc[X_all.index, 'Fecha'].dt.date
    fechas = sorted(fecha_series.unique())
    split = int(len(fechas) * 0.7)
    fechas_train = set(fechas[:split])
    fechas_test = set(fechas[split:])

    train_mask = fecha_series.isin(fechas_train)
    X_train = X_all.loc[train_mask]
    y_train = Y_all.loc[train_mask]

    transitions = build_transitions(df, X_all, fechas_test)
    print('Transiciones de test:', len(transitions))

    # Build analytic models from training data
    d_train = df[df['Fecha'].dt.date.isin(fechas_train)].copy()
    trans_prob, trans_total = analizador._transiciones_markov(d_train)
    d_prep = analizador.preparar_datos_markov(d_train)
    _, _, trans_h, total_h = analizador.construir_matrices_markov(d_prep, incluir_trasnocho=False)
    freq_hora = analizador._frecuencias_hora(d_train, 'Solo_hora')
    matrices_por_dia = analizador.get_matriz_markov_por_dia(d_train)

    # train RF and XGB
    pipe_rf = analizador.crear_pipeline_ml(RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1), numeric_features, categorical_features)
    pipe_rf.fit(X_train, y_train)
    clf_rf = pipe_rf.named_steps['classifier']
    # XGBoost: use a smaller, faster config to keep runtime reasonable
    pipe_xgb = analizador.crear_pipeline_ml(XGBClassifier(n_estimators=80, max_depth=4, random_state=42, verbosity=0, n_jobs=4, eval_metric='mlogloss'), numeric_features, categorical_features)
    pipe_xgb.fit(X_train, y_train)
    clf_xgb = pipe_xgb.named_steps['classifier']

    classes = clf_rf.classes_

    HORA_ORDER = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                  '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']

    # counters
    models = ['Markov','FrecHora','MkHora','GxHora','MarkovDia','RF','XGB','RF+XGB']
    stats = {m: {'top5':0,'top10':0,'total':0, 'by_hour':{h:{'top5':0,'top10':0,'n':0} for h in HORA_ORDER}} for m in models}

    for prev_idx, cur_idx in transitions:
        prev_state = df.loc[prev_idx]
        actual = df.loc[cur_idx]
        num_real = int(actual['Num_Int'])
        hora_real = actual['Solo_hora']
        # Skip if hour not in order mapping? We'll still count

        ultimo_num = int(prev_state['Num_Int'])

        # MARKOV global
        markov_scores = {n: trans_prob.get((ultimo_num, n), 0) for n in range(38)}
        # get topk for markov
        for k in (5,10):
            top = topk_from_scores(markov_scores, k)
            hit = num_real in top
            stats['Markov'][f'top{k}'] += int(hit)

        # FrecHora
        hourly_scores = freq_hora.get(hora_real, {})
        hourly_scores_full = {n: hourly_scores.get(n, 0) for n in range(38)}
        for k in (5,10):
            top = topk_from_scores(hourly_scores_full, k)
            stats['FrecHora'][f'top{k}'] += int(num_real in top)

        # MkHora (Markov x Hora)
        pareja_h = (prev_state['Hora'], actual['Hora'])
        markov_hora_scores = {}
        if pareja_h in trans_h and ultimo_num in total_h.get(pareja_h, {}):
            tot_hh = total_h[pareja_h][ultimo_num]
            if tot_hh > 0:
                markov_hora_scores = {n2: cnt / tot_hh * 100 for n2, cnt in trans_h[pareja_h][ultimo_num].items()}
        # fill to 38 with global trans_prob
        if len(markov_hora_scores) < 38:
            for n in range(38):
                if n not in markov_hora_scores:
                    markov_hora_scores[n] = trans_prob.get((ultimo_num, n), 0)
        for k in (5,10):
            top = topk_from_scores(markov_hora_scores, k)
            stats['MkHora'][f'top{k}'] += int(num_real in top)

        # GxHora (weighted)
        m_hh = total_h.get(pareja_h, {}).get(ultimo_num, 0)
        w_hh = min(0.9, max(0.1, m_hh / 50))
        todos_hh = set(markov_scores.keys()) | set(markov_hora_scores.keys())
        gxh = {n: markov_scores.get(n, 0) * (1 - w_hh) + markov_hora_scores.get(n, 0) * w_hh for n in todos_hh}
        for k in (5,10):
            top = topk_from_scores(gxh, k)
            stats['GxHora'][f'top{k}'] += int(num_real in top)

        # Markov x Dia
        dia = actual['Fecha'].day_name()
        if dia in matrices_por_dia:
            td_prob, td_total = matrices_por_dia[dia]
            markov_dia_scores = {n: td_prob.get((ultimo_num, n), 0) for n in range(38)}
        else:
            markov_dia_scores = markov_scores
        for k in (5,10):
            top = topk_from_scores(markov_dia_scores, k)
            stats['MarkovDia'][f'top{k}'] += int(num_real in top)

        # RF / XGB predictions: build X from prev_state
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion', 'Color_Previo', 'Paridad_Previo',
                              'Prob_Cooc', 'Lift_Dia', 'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num']
        available_numeric = [f for f in numeric_candidates if f in df.columns]
        X_dict = prev_state[available_numeric + ['Hora_Sorteo']].to_dict()
        X = pd.DataFrame([X_dict])
        if X.isnull().any().any():
            rf_top = []
            xgb_top = []
            rf_probs = None
            xgb_probs = None
        else:
            rf_probs = pipe_rf.predict_proba(X)[0]
            xgb_probs = pipe_xgb.predict_proba(X)[0]
            rf_order = np.argsort(rf_probs)[::-1]
            xgb_order = np.argsort(xgb_probs)[::-1]
            rf_top = classes[rf_order[:10]]
            xgb_top = classes[xgb_order[:10]]

        for k in (5,10):
            stats['RF'][f'top{k}'] += int(num_real in list(rf_top[:k]))
            stats['XGB'][f'top{k}'] += int(num_real in list(xgb_top[:k]))

        # RF+XGB combined (average probs)
        if rf_probs is not None and xgb_probs is not None:
            combined_probs = (rf_probs + xgb_probs) / 2.0
            order = np.argsort(combined_probs)[::-1]
            combined_top = classes[order[:10]]
        else:
            combined_top = []
        for k in (5,10):
            stats['RF+XGB'][f'top{k}'] += int(num_real in list(combined_top[:k]))

        # update totals and per-hour counts
        for m in models:
            stats[m]['total'] += 1
            if hora_real in HORA_ORDER:
                stats[m]['by_hour'][hora_real]['n'] += 1
                for k in (5,10):
                    stats[m]['by_hour'][hora_real][f'top{k}'] += stats[m][f'top{k}']  # we'll correct later

    # The above per-hour accumulation incorrectly added cumulative counts; recompute per-hour properly
    # We'll recompute per-hour by iterating transitions again but only counting hits per hour
    # reset per-hour
    for m in models:
        for h in HORA_ORDER:
            stats[m]['by_hour'][h] = {'top5':0,'top10':0,'n':0}

    for prev_idx, cur_idx in transitions:
        prev_state = df.loc[prev_idx]
        actual = df.loc[cur_idx]
        num_real = int(actual['Num_Int'])
        hora_real = actual['Solo_hora']
        ultimo_num = int(prev_state['Num_Int'])

        # recompute same as above but only update per-hour
        markov_scores = {n: trans_prob.get((ultimo_num, n), 0) for n in range(38)}
        hourly_scores = freq_hora.get(hora_real, {})
        hourly_scores_full = {n: hourly_scores.get(n, 0) for n in range(38)}
        pareja_h = (prev_state['Hora'], actual['Hora'])
        markov_hora_scores = {}
        if pareja_h in trans_h and ultimo_num in total_h.get(pareja_h, {}):
            tot_hh = total_h[pareja_h][ultimo_num]
            if tot_hh > 0:
                markov_hora_scores = {n2: cnt / tot_hh * 100 for n2, cnt in trans_h[pareja_h][ultimo_num].items()}
        if len(markov_hora_scores) < 38:
            for n in range(38):
                if n not in markov_hora_scores:
                    markov_hora_scores[n] = trans_prob.get((ultimo_num, n), 0)
        m_hh = total_h.get(pareja_h, {}).get(ultimo_num, 0)
        w_hh = min(0.9, max(0.1, m_hh / 50))
        todos_hh = set(markov_scores.keys()) | set(markov_hora_scores.keys())
        gxh = {n: markov_scores.get(n, 0) * (1 - w_hh) + markov_hora_scores.get(n, 0) * w_hh for n in todos_hh}
        dia = actual['Fecha'].day_name()
        if dia in matrices_por_dia:
            td_prob, td_total = matrices_por_dia[dia]
            markov_dia_scores = {n: td_prob.get((ultimo_num, n), 0) for n in range(38)}
        else:
            markov_dia_scores = markov_scores

        # RF/XGB
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion', 'Color_Previo', 'Paridad_Previo',
                              'Prob_Cooc', 'Lift_Dia', 'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num']
        available_numeric = [f for f in numeric_candidates if f in df.columns]
        X_dict = prev_state[available_numeric + ['Hora_Sorteo']].to_dict()
        X = pd.DataFrame([X_dict])
        if X.isnull().any().any():
            rf_probs = xgb_probs = None
            rf_top = xgb_top = []
        else:
            rf_probs = pipe_rf.predict_proba(X)[0]
            xgb_probs = pipe_xgb.predict_proba(X)[0]
            rf_top = classes[np.argsort(rf_probs)[::-1][:10]]
            xgb_top = classes[np.argsort(xgb_probs)[::-1][:10]]
        combined_top = []
        if rf_probs is not None and xgb_probs is not None:
            combined_top = classes[np.argsort((rf_probs + xgb_probs)/2.0)[::-1][:10]]

        # increment per-hour
        if hora_real in HORA_ORDER:
            for k in (5,10):
                stats['Markov']['by_hour'][hora_real][f'top{k}'] += int(num_real in topk_from_scores(markov_scores, k))
                stats['FrecHora']['by_hour'][hora_real][f'top{k}'] += int(num_real in topk_from_scores(hourly_scores_full, k))
                stats['MkHora']['by_hour'][hora_real][f'top{k}'] += int(num_real in topk_from_scores(markov_hora_scores, k))
                stats['GxHora']['by_hour'][hora_real][f'top{k}'] += int(num_real in topk_from_scores(gxh, k))
                stats['MarkovDia']['by_hour'][hora_real][f'top{k}'] += int(num_real in topk_from_scores(markov_dia_scores, k))
                stats['RF']['by_hour'][hora_real][f'top{k}'] += int(num_real in list(rf_top[:k]))
                stats['XGB']['by_hour'][hora_real][f'top{k}'] += int(num_real in list(xgb_top[:k]))
                stats['RF+XGB']['by_hour'][hora_real][f'top{k}'] += int(num_real in list(combined_top[:k]))
            stats['Markov']['by_hour'][hora_real]['n'] += 1
            # use same n for all
            for m in models:
                stats[m]['by_hour'][hora_real]['n'] = stats['Markov']['by_hour'][hora_real]['n']

    # finalize totals
    out = {'summary':{}, 'by_hour':{}}
    for m in models:
        total = stats[m]['total']
        out['summary'][m] = {
            'top5': f"{stats[m]['top5']}/{total} = {stats[m]['top5']/total*100:.2f}%" if total>0 else None,
            'top10': f"{stats[m]['top10']}/{total} = {stats[m]['top10']/total*100:.2f}%" if total>0 else None,
            'total': total
        }
    for h in HORA_ORDER:
        out['by_hour'][h] = {}
        for m in models:
            n = stats[m]['by_hour'][h]['n']
            t5 = stats[m]['by_hour'][h]['top5']
            t10 = stats[m]['by_hour'][h]['top10']
            out['by_hour'][h][m] = {'n': n, 'top5': f"{t5}/{n} = {t5/n*100:.2f}%" if n>0 else None, 'top10': f"{t10}/{n} = {t10/n*100:.2f}%" if n>0 else None}

    import os
    os.makedirs('results', exist_ok=True)
    with open('results/results_topk.json','w') as f:
        json.dump(out,f,indent=2)
    print('Resultados guardados en results/results_topk.json')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
