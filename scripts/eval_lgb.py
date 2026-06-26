#!/usr/bin/env python3
"""
Evaluates LightGBM vs Random Forest vs XGBoost on a time-series walk-forward split.
Reports Top-5, Top-10, Top-25 precision per model, plus per-hour breakdown.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lotto_activo import analizador, CONFIG
from utils import load_and_prepare_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import json


def topk_accuracy(y_true, y_proba, classes, k=25):
    hits = 0
    for i in range(len(y_true)):
        order = np.argsort(y_proba[i])[::-1][:k]
        preds = classes[order]
        if y_true[i] in preds:
            hits += 1
    return hits / len(y_true), hits, len(y_true)


def main():
    print("=" * 70)
    print("  EVALUACION COMPARATIVA: LGB vs RF vs XGB")
    print("=" * 70)

    print("\nCargando datos...")
    df = load_and_prepare_data(CONFIG['excel_file'], analizador)
    df = analizador.agregar_caracteristicas_avanzadas(df)

    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])

    X_all, Y_all, le_y, numeric_features, categorical_features, available_features = \
        analizador.preparar_datos_ml_completo(df)

    fecha_series = df.loc[X_all.index, 'Fecha'].dt.date
    fechas = sorted(fecha_series.unique())
    split_idx = int(len(fechas) * 0.75)
    fechas_train = set(fechas[:split_idx])
    fechas_test = set(fechas[split_idx:])

    train_mask = fecha_series.isin(fechas_train)
    test_mask = fecha_series.isin(fechas_test)

    X_train = X_all.loc[train_mask]
    y_train = Y_all.loc[train_mask]
    X_test = X_all.loc[test_mask]
    y_test = Y_all.loc[test_mask]

    print(f"  Train: {len(X_train)} muestras ({len(fechas_train)} días)")
    print(f"  Test:  {len(X_test)} muestras ({len(fechas_test)} días)")
    print(f"  Features: {len(available_features)}")
    print(f"  Clases: {len(le_y.classes_)}")

    pipe_rf = analizador.crear_pipeline_ml(
        RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        numeric_features, categorical_features
    )
    pipe_xgb = analizador.crear_pipeline_ml(
        XGBClassifier(n_estimators=80, max_depth=4, random_state=42, verbosity=0, n_jobs=4, eval_metric='mlogloss'),
        numeric_features, categorical_features
    )
    pipe_lgb = analizador.crear_pipeline_ml(
        lgb.LGBMClassifier(n_estimators=100, num_leaves=31, max_depth=5,
                           learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbose=-1),
        numeric_features, categorical_features
    )

    print("\nEntrenando Random Forest...")
    pipe_rf.fit(X_train, y_train)
    clf_rf = pipe_rf.named_steps['classifier']
    classes_rf = clf_rf.classes_

    print("Entrenando XGBoost...")
    pipe_xgb.fit(X_train, y_train)
    clf_xgb = pipe_xgb.named_steps['classifier']
    classes_xgb = clf_xgb.classes_

    print("Entrenando LightGBM...")
    pipe_lgb.fit(X_train, y_train)
    clf_lgb = pipe_lgb.named_steps['classifier']
    classes_lgb = clf_lgb.classes_

    print("\nEvaluando en test...")
    y_proba_rf = pipe_rf.predict_proba(X_test)
    y_proba_xgb = pipe_xgb.predict_proba(X_test)
    y_proba_lgb = pipe_lgb.predict_proba(X_test)

    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    models = {
        'RF':  (y_proba_rf, classes_rf),
        'XGB': (y_proba_xgb, classes_xgb),
        'LGB': (y_proba_lgb, classes_lgb),
    }

    print(f"\n{'='*60}")
    print(f"  PRECISION GLOBAL EN TEST ({len(X_test)} muestras)")
    print(f"{'='*60}")
    print(f"{'Modelo':<10} {'Top-5':<15} {'Top-10':<15} {'Top-25':<15}")
    print(f"{'-'*55}")

    results_summary = {}
    for name, (proba, classes) in models.items():
        row = {}
        for k in [5, 10, 25]:
            acc, hits, total = topk_accuracy(y_test_arr, proba, classes, k=k)
            row[f'top{k}'] = {'acc': acc, 'hits': hits, 'total': total}
            print(f"{name:<10} {hits}/{total} = {acc*100:.2f}%{'':<5} ", end='')
        print()
        results_summary[name] = row

    models_data = {name: proba for name, (proba, _) in models.items()}
    models_classes = {name: classes for name, (_, classes) in models.items()}

    for k in [5, 10, 25]:
        if k == 25:
            print(f"\n{'='*60}")
            print(f"  MEJOR MODELO TOP-{k}")
            print(f"{'='*60}")
            best = max(results_summary, key=lambda m: results_summary[m][f'top{k}']['acc'])
            print(f"  {best} = {results_summary[best][f'top{k}']['acc']*100:.2f}%")
            print(f"\n  DIFERENCIAS vs LGB:")
            for m in models:
                if m == 'LGB':
                    continue
                diff = results_summary['LGB'][f'top{k}']['acc'] - results_summary[m][f'top{k}']['acc']
                sign = "+" if diff > 0 else ""
                print(f"    LGB - {m}: {sign}{diff*100:.2f} puntos porcentuales")

    HORA_ORDER = ['08:00 AM', '09:00 AM', '10:00 AM', '11:00 AM', '12:00 PM',
                  '01:00 PM', '02:00 PM', '03:00 PM', '04:00 PM', '05:00 PM',
                  '06:00 PM', '07:00 PM']

    print(f"\n{'='*90}")
    print(f"  PRECISION POR HORA (TOP-25)")
    print(f"{'='*90}")
    header = f"{'Hora':<12}" + "".join(f"{m+' Top-25':<20}" for m in models) + f"{'Muestras':<10}"
    print(header)
    print(f"{'-'*len(header)}")

    df_test = df.loc[X_test.index].copy()
    df_test['y_true'] = y_test_arr

    hora_map = {
        '08:00:00': '08:00 AM', '09:00:00': '09:00 AM', '10:00:00': '10:00 AM',
        '11:00:00': '11:00 AM', '12:00:00': '12:00 PM', '13:00:00': '01:00 PM',
        '14:00:00': '02:00 PM', '15:00:00': '03:00 PM', '16:00:00': '04:00 PM',
        '17:00:00': '05:00 PM', '18:00:00': '06:00 PM', '19:00:00': '07:00 PM',
    }
    hora_col = df_test['Solo_hora'] if 'Solo_hora' in df_test.columns else df_test['Hora'].map(hora_map)

    all_scores = {}
    for name, (proba, classes) in models.items():
        all_scores[name] = {
            'by_hour': {h: {'hits': 0, 'total': 0} for h in HORA_ORDER}
        }

    for i in range(len(df_test)):
        h = hora_col.iloc[i] if hasattr(hora_col, 'iloc') else hora_col[i]
        if h not in HORA_ORDER:
            continue
        true_val = y_test_arr[i]
        for name, (proba, classes) in models.items():
            order = np.argsort(proba[i])[::-1][:25]
            preds = classes[order]
            if true_val in preds:
                all_scores[name]['by_hour'][h]['hits'] += 1
            all_scores[name]['by_hour'][h]['total'] += 1

    for h in HORA_ORDER:
        cells = [f"{h[:5]:<12}"]
        for name in models:
            bh = all_scores[name]['by_hour'][h]
            if bh['total'] > 0:
                pct = bh['hits'] / bh['total'] * 100
                cells.append(f"{bh['hits']}/{bh['total']}={pct:.1f}%{'':<9}")
            else:
                cells.append(f"{'N/A':<20}")
        cells.append(f"{all_scores['RF']['by_hour'][h]['total']:<10}")
        print("".join(cells))

    best_by_hour = {}
    for h in HORA_ORDER:
        best = max(models, key=lambda m: all_scores[m]['by_hour'][h].get('hits', 0) /
                  max(all_scores[m]['by_hour'][h].get('total', 1), 1) if all_scores[m]['by_hour'][h]['total'] > 0 else 0)
        best_by_hour[h] = best

    print(f"\n{'='*60}")
    print(f"  GANADOR POR HORA:")
    print(f"{'='*60}")
    for h in HORA_ORDER:
        bh = all_scores[best_by_hour[h]]['by_hour'][h]
        if bh['total'] > 0:
            print(f"  {h[:5]:<8} → {best_by_hour[h]:<5} ({bh['hits']}/{bh['total']} = {bh['hits']/bh['total']*100:.1f}%)")

    os.makedirs('results', exist_ok=True)
    out = {}
    for name, (proba, classes) in models.items():
        out[name] = {}
        for k in [5, 10, 25]:
            r = results_summary[name][f'top{k}']
            out[name][f'top{k}'] = {'pct': round(r['acc']*100, 2), 'hits': r['hits'], 'total': r['total']}
        out[name]['by_hour'] = {}
        for h in HORA_ORDER:
            bh = all_scores[name]['by_hour'][h]
            out[name]['by_hour'][h] = {'pct': round(bh['hits']/bh['total']*100, 2) if bh['total'] > 0 else 0,
                                       'hits': bh['hits'], 'total': bh['total']}

    with open('results/eval_lgb_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResultados guardados en results/eval_lgb_results.json")


if __name__ == '__main__':
    main()
