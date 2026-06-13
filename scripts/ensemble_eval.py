#!/usr/bin/env python3
"""
Ensemble evaluation script: combine RF predict_proba with analytical Full model
and evaluate Top-5/Top-25 over a walk-forward test split.

Usage: python3 scripts/ensemble_eval.py
"""
from lotto_activo import CONFIG, analizador
from utils import load_and_prepare_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np, pandas as pd
import json


def build_test_transitions(df, X_all, fechas_test):
    # returns list of prev indices, X_rows (DataFrame), reals (list)
    idxs = list(df.index)
    rows = []
    reals = []
    prev_idxs = []
    for pos in range(1, len(idxs)):
        prev_idx = idxs[pos - 1]
        cur_idx = idxs[pos]
        if df.loc[prev_idx, 'Fecha'] != df.loc[cur_idx, 'Fecha']:
            continue
        if df.loc[cur_idx, 'Fecha'].date() not in fechas_test:
            continue
        if prev_idx not in X_all.index or cur_idx not in X_all.index:
            continue
        rows.append(X_all.loc[prev_idx].values)
        reals.append(int(df.loc[cur_idx, 'Num_Int']))
        prev_idxs.append(prev_idx)
    X_rows = pd.DataFrame(rows, columns=X_all.columns)
    return prev_idxs, X_rows, reals


def analytical_probs_for(prev_idx, cur_idx, d_train):
    # Use get_prediccion_combinada on training data to get scores for top 38
    # if prev_idx/cur_idx are outside d_train, fallback zeros
    if prev_idx not in d_train.index or cur_idx not in d_train.index:
        return np.zeros(38)
    num_origen = int(d_train.loc[prev_idx, 'Num_Int'])
    hora_origen = d_train.loc[prev_idx, 'Hora']
    hora_dest = d_train.loc[cur_idx, 'Hora']
    # if any missing, fallback zeros
    if num_origen is None or hora_origen is None or hora_dest is None:
        return np.zeros(38)
    items = anal.get_prediccion_combinada(d_train, num_origen, hora_origen, hora_dest, top_k=38, incluir_trasnocho=False)
    vec = np.zeros(38)
    for n, score, muestras in items:
        vec[int(n)] = float(score)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec


if __name__ == '__main__':
    print('Loading data and features...')
    df = load_and_prepare_data(CONFIG['excel_file'], analizador)
    df = analizador.agregar_caracteristicas_avanzadas(df)
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])

    # prepare ML X/Y
    X_all, Y_all, le_y, numeric_features, categorical_features, available_features = analizador.preparar_datos_ml_completo(df)

    # dates split
    fecha_series = df.loc[X_all.index, 'Fecha'].dt.date
    fechas = sorted(fecha_series.unique())
    split = int(len(fechas) * 0.7)
    fechas_train = set(fechas[:split])
    fechas_test = set(fechas[split:])

    train_mask = fecha_series.isin(fechas_train)
    X_train = X_all.loc[train_mask]
    y_train = Y_all.loc[train_mask]

    # build test transitions
    prev_idxs, X_rows, reals = build_test_transitions(df, X_all, fechas_test)
    print('Train rows:', len(X_train), 'Test transitions:', len(X_rows))

    # train RF pipeline
    pipe = analizador.crear_pipeline_ml(RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1), numeric_features, categorical_features)
    pipe.fit(X_train, y_train)
    rf_proba = pipe.predict_proba(X_rows)
    rf_classes = pipe.named_steps['classifier'].classes_

    # prepare training df for analytical model sources (use same d_train as get_prediccion_combinada expects full dataframe)
    d_train = df[df['Fecha'].dt.date.isin(fechas_train)].copy()
    # alias anal for readability
    anal = analizador

    # build analytical probas by calling get_prediccion_combinada per transition
    anal_probas = []
    for i, prev_idx in enumerate(prev_idxs):
        pos = df.index.get_loc(prev_idx)
        cur_idx = df.index[pos + 1]
        vec = analytical_probs_for(prev_idx, cur_idx, d_train)
        # align to rf_classes order
        if not np.array_equal(rf_classes, np.arange(38)):
            # reorder
            aligned = np.array([vec[int(c)] for c in rf_classes])
        else:
            aligned = vec
        anal_probas.append(aligned)
    anal_probas = np.vstack(anal_probas)
    print('Analytical probas shape:', anal_probas.shape)

    # ensemble grid (fine search)
    import numpy as _np
    weights = list(_np.round(_np.arange(0.0, 1.0001, 0.05), 2))
    results = {}
    for w in weights:
        combined = w * rf_proba + (1 - w) * anal_probas
        # compute topk
        def score_topk(mat, reals, k):
            hits = 0
            n = len(reals)
            for i in range(n):
                order = np.argsort(mat[i])[::-1][:k]
                preds = rf_classes[order]
                if reals[i] in preds:
                    hits += 1
            return hits, n, hits / n * 100

        r5 = score_topk(combined, reals, 5)
        r25 = score_topk(combined, reals, 25)
        results[w] = {'top5': r5, 'top25': r25}
        print(f'w={w}: Top-5 {r5[2]:.2f}% ({r5[0]}/{r5[1]}), Top-25 {r25[2]:.2f}% ({r25[0]}/{r25[1]})')

    with open('ensemble_results.json', 'w') as f:
        json.dump({str(k): {'top5': v['top5'], 'top25': v['top25']} for k, v in results.items()}, f, indent=2)

    print('Done. Results saved to ensemble_results.json')
