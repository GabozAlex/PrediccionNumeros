import sys
import pandas as pd
import datetime
import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import pickle
import json
from datetime import datetime, date

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import scipy.stats as stats

from utils import setup_logging, mostrar_menu, ANIMALES_38, GRUPOS_ANIMALES, HORA_MAP_12_TO_24


class Loteria:
    """Base class for lottery analysis and prediction (38 animals, 0-37 range)."""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config.get('logger_name', 'loteria'))
        self.animales_carac = {a: 0 for a in config['animales']}
        self.animal_a_grupo = {}
        for grupo, animales in config['grupos_animales'].items():
            for a in animales:
                self.animal_a_grupo[a] = grupo
        self.le_global = None

    def build_animal_a_grupo(self):
        result = {}
        for grupo, animales in self.config['grupos_animales'].items():
            for a in animales:
                result[a] = grupo
        return result

    def verificar_diccionario_animales(self):
        print("\nVERIFICANDO LISTA DE ANIMALES...")
        total_animales = len(self.animales_carac)
        print(f"  Total de animales: {total_animales}")
        if total_animales != len(self.config['animales']):
            print(f"ERROR: Se esperaban {len(self.config['animales'])} animales, pero hay {total_animales}")
            return False
        print("Lista correcta -", total_animales, "animales")
        print("\nLISTA COMPLETA DE ANIMALES:")
        for i, animal in enumerate(sorted(self.animales_carac.keys()), 1):
            print(f"   {i:2d}. {animal}")
        return True

    def validar_animal(self, animal):
        animal = animal.strip().upper()
        if animal not in self.animales_carac:
            animales_validos = ", ".join(sorted(self.animales_carac.keys()))
            raise ValueError(f"Animal '{animal}' no valido. Animales validos: {animales_validos}")
        return animal

    def validar_numero(self, numero):
        if not (0 <= numero <= self.config['max_numero']):
            raise ValueError(f"Numero {numero} fuera de rango. Debe ser entre 0-{self.config['max_numero']}")
        return numero

    def calcular_diferencia_ciclica(self, actual, previo, max_val=None):
        if max_val is None:
            max_val = self.config['max_numero'] + 1
        if pd.isna(actual) or pd.isna(previo):
            return np.nan
        actual = int(actual)
        previo = int(previo)
        diferencia_base = abs(actual - previo)
        diferencia_opuesta = max_val - diferencia_base
        return min(diferencia_base, diferencia_opuesta)

    def agregar_caracteristicas_avanzadas(self, datos):
        df = datos.copy()
        if 'Timestamp' in df.columns:
            df['Dia_Semana'] = df['Timestamp'].dt.dayofweek
            df['Mes'] = df['Timestamp'].dt.month
            df['Hora_Num'] = df['Timestamp'].dt.hour
            df['Es_Fin_Semana'] = df['Dia_Semana'].isin([5, 6]).astype(int)
        df['Posicion_Previo'] = df['Numero'].shift(1)
        df['Animal_Previo'] = df['Animal'].shift(1)
        df['Diferencia_Ciclica'] = df.apply(
            lambda row: self.calcular_diferencia_ciclica(row['Numero'], row['Posicion_Previo']),
            axis=1
        )
        df['Frecuencia_Animal_10'] = df.groupby('Animal')['Animal'].transform(
            lambda x: x.rolling(10, min_periods=1).count()
        )
        df['Repite_Animal'] = (df['Animal'] == df['Animal_Previo']).astype(int)
        df['Mismo_Animal_3_Sorteos'] = (df['Animal'] == df['Animal'].shift(1)) & (df['Animal'] == df['Animal'].shift(2))
        df['Mismo_Animal_3_Sorteos'] = df['Mismo_Animal_3_Sorteos'].astype(int)
        df['Media_Movil_5'] = df['Numero'].rolling(5, min_periods=1).mean()
        df['Std_Movil_5'] = df['Numero'].rolling(5, min_periods=1).std()

        def grupo_ruleta(numero):
            if numero <= 9:
                return 0
            elif numero <= 19:
                return 1
            elif numero <= 29:
                return 2
            else:
                return 3

        df['Grupo_Ruleta'] = df['Numero'].apply(grupo_ruleta)
        df['Grupo_Ruleta_Previo'] = df['Grupo_Ruleta'].shift(1)

        from collections import defaultdict
        # Expanding window features (no future leakage)
        hour_animal_cnt = defaultdict(lambda: defaultdict(int))
        hour_total_cnt = defaultdict(int)
        trans_cnt = defaultdict(lambda: defaultdict(int))
        trans_total_cnt = defaultdict(int)
        last_pos = {}
        prob_hora_vals = []
        prob_trans_vals = []
        freq10_vals = []
        distancia_vals = []
        for idx in range(len(df)):
            cur_animal = df.iloc[idx]['Animal']
            cur_hour = df.iloc[idx]['Solo_hora']
            # Update cumulative counts (include current draw)
            hour_animal_cnt[cur_hour][cur_animal] += 1
            hour_total_cnt[cur_hour] += 1
            prob_hora_vals.append(hour_animal_cnt[cur_hour][cur_animal] / hour_total_cnt[cur_hour] * 100)
            if idx > 0:
                prev = df.iloc[idx-1]['Animal']
                trans_cnt[prev][cur_animal] += 1
                trans_total_cnt[prev] += 1
                prob_trans_vals.append(trans_cnt[prev][cur_animal] / trans_total_cnt[prev] * 100)
            else:
                prob_trans_vals.append(0.0)
            # Frequency of this animal in last 10 draws
            start = max(0, idx - 9)
            freq10_vals.append(int((df.iloc[start:idx+1]['Animal'] == cur_animal).sum()))
            # Draws since this animal last appeared before current draw
            if cur_animal in last_pos:
                distancia_vals.append(idx - last_pos[cur_animal] - 1)
            else:
                distancia_vals.append(-1)
            last_pos[cur_animal] = idx
        df['Prob_Hist_Hora'] = prob_hora_vals
        df['Prob_Trans_Markov'] = prob_trans_vals
        df['Frecuencia_10'] = freq10_vals
        df['Sorteos_Desde_Aparicion'] = distancia_vals

        print(f"Caracteristicas avanzadas anadidas: {len(df.columns)} features totales")
        return df

    def preparar_datos_ml_completo(self, datos):
        df_ml = datos.copy()
        animales_validos = list(self.animales_carac.keys())
        df_ml = df_ml[df_ml['Animal'].isin(animales_validos)].copy()
        if len(df_ml) < 50:
            print(f"Advertencia: Solo {len(df_ml)} registros validos. Se recomiendan al menos 50.")
        # Filter to period where all animals are present
        if 'Fecha' in df_ml.columns:
            sorted_dates = sorted(df_ml['Fecha'].unique())
            all_animals = set(self.animales_carac.keys())
            seen_animals = set()
            min_date = sorted_dates[0]
            for d in sorted_dates:
                df_dia = df_ml[df_ml['Fecha'] == d]
                seen_animals.update(df_dia['Animal'].unique())
                if seen_animals == all_animals:
                    min_date = d
                    break
            before = len(df_ml)
            df_ml = df_ml[df_ml['Fecha'] >= min_date].copy()
            after = len(df_ml)
            if before != after:
                print(f"Filtrados datos desde {min_date}: {before} -> {after} registros (quitados {before-after} historicos)")
        numeric_features = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                            'Frecuencia_10', 'Sorteos_Desde_Aparicion']
        categorical_features = ['Hora_Sorteo']
        if 'Hora_Sorteo' not in df_ml.columns:
            df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip().str.zfill(8)
        # Always fit the label encoder on the full known animal list from config
        # to ensure a stable mapping between training and production/prediction.
        animales_en_datos = sorted(df_ml['Animal'].unique())
        le_y = LabelEncoder()
        try:
            animales_fijos = list(self.config.get('animales', self.animales_carac.keys()))
            le_y.fit(animales_fijos)
        except Exception:
            # Fallback: fit on animals present in the data
            le_y.fit(animales_en_datos)
        df_ml['Animal_Encoded'] = le_y.transform(df_ml['Animal'])
        Y = df_ml['Animal_Encoded']
        print(f"Clases configuradas: {len(le_y.classes_)} animales")
        print(f"Animales en datos: {len(animales_en_datos)} animales unicos")
        available_features = []
        for feature in numeric_features + categorical_features:
            if feature in df_ml.columns:
                available_features.append(feature)
        X = df_ml[available_features].copy()
        filas_antes = len(X)
        X = X.dropna()
        filas_despues = len(X)
        if filas_antes != filas_despues:
            print(f"Eliminadas {filas_antes - filas_despues} filas con valores NaN")
        Y = Y.loc[X.index]
        print(f"Datos ML preparados: {len(X)} muestras, {len(available_features)} caracteristicas")
        print(f"   Caracteristicas: {available_features}")
        return X, Y, le_y, numeric_features, categorical_features, available_features

    def crear_pipeline_ml(self, modelo, numeric_features, categorical_features):
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('num', 'passthrough', numeric_features)
            ],
            remainder='drop'
        )
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', modelo)])
        return clf

    def calcular_precision_top_k(self, y_real, y_proba, k=3):
        top_k_predicciones = np.argsort(y_proba, axis=1)[:, -k:]
        correctos = 0
        for i, real in enumerate(y_real):
            if real in top_k_predicciones[i]:
                correctos += 1
        return correctos / len(y_real)

    def entrenar_modelo_ml(self, X, Y, modelo, modelo_nombre, numeric_features, categorical_features):
        print(f"\nEntrenando {modelo_nombre} con {len(X)} sorteos")
        tscv = TimeSeriesSplit(n_splits=5)
        pipeline = self.crear_pipeline_ml(modelo, numeric_features, categorical_features)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            pipeline.fit(X_train, Y_train)
            accuracy = pipeline.score(X_test, Y_test)
            print(f"Precision de la validacion temporal: {accuracy:.2%}")
        return pipeline

    def predecir_top_k_por_hora(self, pipeline, le_y, df_ml, k=20):
        matriz_prediccion_ia = {}
        if 'Hora_Sorteo' not in df_ml.columns:
            df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip().str.zfill(8)
        horas_sorteo = sorted(df_ml['Hora_Sorteo'].unique())
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion']
        available_numeric = [f for f in numeric_candidates if f in df_ml.columns]
        all_features = available_numeric + ['Hora_Sorteo']
        print(f"\nGenerando Matriz de Prediccion TOP-{k} de la IA ({len(all_features)} features)...")
        for hora in horas_sorteo:
            df_hora = df_ml[df_ml['Hora_Sorteo'] == hora].iloc[[-1]].copy()
            if df_hora.isnull().any().any() or df_hora.empty:
                continue
            X_query = df_hora[all_features]
            try:
                y_proba = pipeline.predict_proba(X_query)[0]
                indices_top_k = np.argsort(y_proba)[::-1][:k]
                animales_predichos = le_y.inverse_transform(indices_top_k)
                matriz_prediccion_ia[hora] = animales_predichos.tolist()
            except Exception as e:
                print(f"Error en hora {hora}: {e}")
                continue
        print(f"Matriz generada con {len(matriz_prediccion_ia)} horas")
        return matriz_prediccion_ia

    def _top20_cv_scorer(self, estimator, X, y):
        try:
            y_proba = estimator.predict_proba(X)
            top20 = np.argsort(y_proba, axis=1)[:, -20:]
            correct = 0
            for i, true_val in enumerate(y):
                if true_val in top20[i]:
                    correct += 1
            return correct / len(y)
        except Exception:
            return 0.0

    def optimizar_hiperparametros_rf(self, X, Y, numeric_features, categorical_features):
        self.logger.info("Iniciando optimizacion de Random Forest...")
        param_dist = {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
        modelo_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        pipeline_base = self.crear_pipeline_ml(modelo_base, numeric_features, categorical_features)
        tscv = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            pipeline_base,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring=self._top20_cv_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1,
            error_score=0.0
        )
        random_search.fit(X, Y)
        self.logger.info(f"Mejores parametros RF: {random_search.best_params_}")
        self.logger.info(f"Mejor score RF (Top-20): {random_search.best_score_:.4f}")
        return random_search.best_estimator_

    def optimizar_hiperparametros_xgb(self, X, Y, numeric_features, categorical_features):
        """Optimiza hiperparametros para XGBoost"""
        self.logger.info("Iniciando optimizacion de XGBoost...")
        param_dist = {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            'classifier__gamma': [0, 0.1, 0.2, 0.3]
        }
        # Ensure all classes are present in the dataset for XGBoost
        y_unique = np.unique(Y)
        n_classes_expected = len(y_unique)
        # Remove rows with classes that would break TimeSeriesSplit folds
        # by ensuring each fold has at least min_samples_per_class
        min_samples = Y.value_counts().min()
        if min_samples < 3:
            self.logger.warning(f"Clase con solo {min_samples} muestras - puede causar errores en XGBoost")
        modelo_base = XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=1
        )
        pipeline_base = self.crear_pipeline_ml(modelo_base, numeric_features, categorical_features)
        tscv = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            pipeline_base,
            param_distributions=param_dist,
            n_iter=10,
            cv=tscv,
            scoring=self._top20_cv_scorer,
            n_jobs=1,
            random_state=42,
            verbose=1,
            error_score=0.0
        )
        random_search.fit(X, Y)
        self.logger.info(f"Mejores parametros XGB: {random_search.best_params_}")
        self.logger.info(f"Mejor score XGB (Top-20): {random_search.best_score_:.4f}")
        return random_search.best_estimator_

    def entrenar_modelo_con_optimizacion(self, X, Y, tipo_modelo, numeric_features, categorical_features):
        self.logger.info(f"Iniciando entrenamiento con optimizacion para {tipo_modelo}")
        start_time = datetime.now()
        if tipo_modelo == 'rf':
            modelo_optimizado = self.optimizar_hiperparametros_rf(X, Y, numeric_features, categorical_features)
            modelo_nombre = "Random Forest Optimizado"
        elif tipo_modelo == 'xgb':
            modelo_optimizado = self.optimizar_hiperparametros_xgb(X, Y, numeric_features, categorical_features)
            modelo_nombre = "XGBoost Optimizado"
        else:
            raise ValueError("Tipo de modelo no soportado")
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        top20_accuracies = []
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            if fold == 0:
                try:
                    modelo_optimizado.fit(X_train, Y_train)
                except Exception as e:
                    self.logger.warning(f"Fold 0 fit failed: {e}, using CV-fitted model")
            try:
                accuracy = modelo_optimizado.score(X_test, Y_test)
                y_proba = modelo_optimizado.predict_proba(X_test)
                top20_acc = self.calcular_precision_top_k(Y_test.values, y_proba, k=20)
                accuracies.append(accuracy)
                top20_accuracies.append(top20_acc)
                self.logger.info(f"Fold {fold+1}: Accuracy = {accuracy:.2%}, Top-20 = {top20_acc:.2%}")
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} evaluation failed: {e}")
        tiempo_entrenamiento = datetime.now() - start_time
        avg_accuracy = np.mean(accuracies)
        avg_top20 = np.mean(top20_accuracies)
        print(f"\nRESULTADOS {modelo_nombre}:")
        print(f"   Accuracy Promedio: {avg_accuracy:.2%}")
        print(f"   Top-20 Accuracy: {avg_top20:.2%}")
        print(f"   Tiempo entrenamiento: {tiempo_entrenamiento}")
        print(f"   Mejor Fold: {max(accuracies):.2%}")
        self.logger.info(f"Entrenamiento completado: {avg_accuracy:.2%} accuracy, {avg_top20:.2%} top-20")
        return modelo_optimizado

    def guardar_modelo(self, modelo, le_y, metricas, nombre_modelo):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_dir = f"{self.config['modelos_dir']}/{nombre_modelo}_{timestamp}"
        if not os.path.exists(self.config['modelos_dir']):
            os.makedirs(self.config['modelos_dir'])
        if not os.path.exists(modelo_dir):
            os.makedirs(modelo_dir)
        modelo_path = f"{modelo_dir}/modelo.pkl"
        with open(modelo_path, 'wb') as f:
            pickle.dump(modelo, f)
        le_path = f"{modelo_dir}/label_encoder.pkl"
        with open(le_path, 'wb') as f:
            pickle.dump(le_y, f)
        metricas_path = f"{modelo_dir}/metricas.json"
        with open(metricas_path, 'w') as f:
            json.dump(metricas, f, indent=2)
        info = {
            'nombre_modelo': nombre_modelo,
            'fecha_entrenamiento': timestamp,
            'caracteristicas_usadas': list(metricas.get('caracteristicas', [])),
            'num_muestras': metricas.get('num_muestras', 0)
        }
        info_path = f"{modelo_dir}/info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        self.logger.info(f"Modelo guardado en: {modelo_dir}")
        return modelo_dir

    def cargar_modelo(self, modelo_dir):
        try:
            with open(f"{modelo_dir}/modelo.pkl", 'rb') as f:
                modelo = pickle.load(f)
            with open(f"{modelo_dir}/label_encoder.pkl", 'rb') as f:
                le_y = pickle.load(f)
            with open(f"{modelo_dir}/metricas.json", 'r') as f:
                metricas = json.load(f)
            self.logger.info(f"Modelo cargado desde: {modelo_dir}")
            return modelo, le_y, metricas
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            return None, None, None

    def cargar_ultimo_modelo(self, tipo_modelo):
        # Try per-lottery directory first, then fallback to old flat modelos/
        search_dirs = [self.config['modelos_dir'], 'modelos']
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            modelos_dir = []
            for dir_name in os.listdir(search_dir):
                if dir_name.startswith(tipo_modelo):
                    modelos_dir.append(dir_name)
            if modelos_dir:
                modelos_dir.sort(reverse=True)
                ultimo_modelo_dir = os.path.join(search_dir, modelos_dir[0])
                modelo, le_y, metricas = self.cargar_modelo(ultimo_modelo_dir)
                if modelo:
                    print(f"Modelo cargado: {ultimo_modelo_dir}")
                    print(f"   Fecha entrenamiento: {metricas.get('fecha_entrenamiento', 'N/A')}")
                    print(f"   Muestras: {metricas.get('num_muestras', 'N/A')}")
                    acc = metricas.get('accuracy_promedio', None)
                    if acc is not None:
                        print(f"   Accuracy: {acc:.2%}")
                    else:
                        print(f"   Accuracy: N/A")
                return modelo, le_y, metricas
        print(f"No se encontraron modelos de tipo: {tipo_modelo}")
        return None, None, None

    def random_forest_optimizado(self, datos):
        try:
            self.logger.info("Ejecutando Random Forest Optimizado")
            datos_con_features = self.agregar_caracteristicas_avanzadas(datos.copy())
            X, Y, le_y, numeric_features, categorical_features, available_features = self.preparar_datos_ml_completo(datos_con_features)
            if len(X) < 50:
                self.logger.warning(f"Datos insuficientes: {len(X)} muestras (minimo 50 recomendado)")
                print("Se recomiendan al menos 50 muestras para optimizacion")
                return None
            modelo_optimizado = self.entrenar_modelo_con_optimizacion(
                X, Y, 'rf', numeric_features, categorical_features
            )
            matriz_prediccion = self.predecir_top_k_por_hora(
                modelo_optimizado, le_y, datos_con_features.copy(), k=20
            )
            metricas = {
                'accuracy_promedio': np.mean([modelo_optimizado.score(X, Y)]),
                'num_muestras': len(X),
                'caracteristicas': available_features,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            modelo_dir = self.guardar_modelo(modelo_optimizado, le_y, metricas, "random_forest")
            print(f"Modelo optimizado guardado en: {modelo_dir}")
            return matriz_prediccion
        except Exception as e:
            self.logger.error(f"Error en Random Forest optimizado: {e}")
            print(f"Error: {e}")
            return None

    def xgboost_optimizado(self, datos):
        try:
            self.logger.info("Ejecutando XGBoost Optimizado")
            datos_con_features = self.agregar_caracteristicas_avanzadas(datos.copy())
            X, Y, le_y, numeric_features, categorical_features, available_features = self.preparar_datos_ml_completo(datos_con_features)
            if len(X) < 50:
                self.logger.warning(f"Datos insuficientes: {len(X)} muestras")
                print("Se recomiendan al menos 50 muestras para optimizacion")
                return None
            modelo_optimizado = self.entrenar_modelo_con_optimizacion(
                X, Y, 'xgb', numeric_features, categorical_features
            )
            matriz_prediccion = self.predecir_top_k_por_hora(
                modelo_optimizado, le_y, datos_con_features.copy(), k=20
            )
            metricas = {
                'accuracy_promedio': np.mean([modelo_optimizado.score(X, Y)]),
                'num_muestras': len(X),
                'caracteristicas': available_features,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            modelo_dir = self.guardar_modelo(modelo_optimizado, le_y, metricas, "xgboost")
            print(f"Modelo optimizado guardado en: {modelo_dir}")
            return matriz_prediccion
        except Exception as e:
            self.logger.error(f"Error en XGBoost optimizado: {e}")
            print(f"Error: {e}")
            return None

    def evaluacion_estrategia_frecuencia(self, datos):
        print("\nEvaluacion Estrategia DINAMICA (BASE: Frecuencia Historica)")
        frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map = {}
        for hora_24h in frecuencia_completa['Hora'].unique():
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
            top_10_map[hora_24h] = top_10_lista
        print("Lista Top-10 generada para todas las horas.")
        self.simular_estrategia(datos, top_10_map)

    def evaluacion_estrategia_ia(self, datos, matriz_prediccion_ia):
        print("\nEvaluacion Estrategia DINAMICA (OPTIMIZADA: Prediccion de IA)")
        top_10_map = matriz_prediccion_ia
        print(f"Matriz de prediccion cargada con {len(top_10_map)} horas.")
        self.simular_estrategia(datos, top_10_map)

    def simular_estrategia(self, datos, top_10_map):
        HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
        GASTO_TARDE = 1200.0
        GANANCIA_POR_ACIERTO = 600.0
        if 'Fecha' not in datos.columns:
            datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
        resultados_simulacion = []
        for fecha, df_dia in datos.groupby('Fecha'):
            aciertos_manana = 0
            df_manana = df_dia[df_dia['Hora'].isin(HORAS_MANANA)].copy()
            for _, row in df_manana.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    aciertos_manana += 1
            jugar_tarde = (aciertos_manana <= 1)
            aciertos_tarde = 0
            ganancia_bruta_tarde = 0
            gasto_tarde = 0
            df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)].copy()
            if jugar_tarde:
                gasto_tarde = GASTO_TARDE
                for _, row in df_tarde.iterrows():
                    hora_filtro = row['Hora']
                    animal_salio = row['Animal']
                    if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                        aciertos_tarde += 1
                ganancia_bruta_tarde = aciertos_tarde * GANANCIA_POR_ACIERTO
            resultados_simulacion.append({
                'Fecha': fecha,
                'Aciertos_Manana': aciertos_manana,
                'Jugar_Tarde': 'SI' if jugar_tarde else 'NO',
                'Aciertos_Tarde': aciertos_tarde,
                'Gasto': gasto_tarde,
                'Ganancia_Bruta': ganancia_bruta_tarde,
                'Ganancia_Neta': ganancia_bruta_tarde - gasto_tarde
            })
        df_resultados = pd.DataFrame(resultados_simulacion)
        columnas_requeridas = ['Aciertos_Tarde', 'Jugar_Tarde', 'Ganancia_Neta']
        for col in columnas_requeridas:
            if col not in df_resultados.columns:
                print(f"ERROR CRITICO: Columna '{col}' no se creo en los resultados")
                print("Columnas disponibles:", list(df_resultados.columns))
                return None
        dias_completos = df_resultados[df_resultados['Aciertos_Tarde'].notna()]
        total_dias = len(dias_completos)
        total_dias_jugados = len(dias_completos[dias_completos['Jugar_Tarde'] == 'SI'])
        gasto_total = df_resultados['Gasto'].sum()
        ganancia_bruta_total = df_resultados['Ganancia_Bruta'].sum()
        ganancia_neta_total = df_resultados['Ganancia_Neta'].sum()
        print("\n" + "="*70)
        print("        RESUMEN DE LA EVALUACION DE LA ESTRATEGIA DINAMICA")
        print("="*70)
        print(f"Dias Completos Analizados: {total_dias}")
        print(f"Dias Jugados (con regla 0 o 1 acierto manana): {total_dias_jugados}")
        print("-" * 70)
        print(f"Gasto Total (solo en la Tarde): {gasto_total:,.2f} Bs")
        print(f"Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
        print(f"GANANCIA/PERDIDA NETA TOTAL: {ganancia_neta_total:,.2f} Bs")
        print("-" * 70)
        if gasto_total > 0:
            roi = (ganancia_neta_total / gasto_total) * 100
            print(f"Retorno de la Inversion (ROI): {roi:,.2f}%")
        if ganancia_neta_total > 0:
            print("\nLa estrategia dinamica genero ganancias en la simulacion.")
        elif ganancia_neta_total < 0:
            print("\nAdvertencia: La estrategia dinamica genero perdidas en la simulacion.")
        else:
            print("\nResultado: Punto de Equilibrio (Ganancia Neta = 0).")
        print("\nAuditoria Diaria de Dias Jugados ---")
        df_jugados = df_resultados[df_resultados['Jugar_Tarde'] == 'SI']
        if not df_jugados.empty:
            top_10_mejor = df_jugados.sort_values(by='Ganancia_Neta', ascending=False).head(10)
            top_10_peor = df_jugados.sort_values(by='Ganancia_Neta', ascending=True).head(10)
            print("\nTOP 10 Dias con Mayor Ganancia Neta:")
            print(top_10_mejor[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']].to_string(index=False))
            print("\nTOP 10 Dias con Mayor Perdida Neta:")
            print(top_10_peor[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']].to_string(index=False))
        else:
            print("No hubo dias suficientes en el historial para aplicar la estrategia.")
        return df_resultados

    def mostrar_matriz_prediccion(self, matriz_prediccion):
        print("\nMATRIZ DE PREDICCION - TOP 20 POR HORA")
        print("=" * 60)
        for hora, animales in sorted(matriz_prediccion.items()):
            print(f"Hora {hora}:")
            for i, animal in enumerate(animales[:20], 1):
                print(f"    {i:2d}. {animal}")
            print()

    def prediccion_hoy_ensemble(self, datos, modelo=None, le_y=None, k=20):
        from collections import defaultdict
        df = datos.copy()
        ultimo = df.iloc[-1]
        ultimo_animal = ultimo['Animal']
        ultimo_numero = int(ultimo['Numero'])
        trans_count = defaultdict(lambda: defaultdict(int))
        trans_total = defaultdict(int)
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                prev = df.iloc[i-1]['Animal']
                curr = df.iloc[i]['Animal']
                trans_count[prev][curr] += 1
                trans_total[prev] += 1
        trans_prob = {}
        for prev in trans_count:
            for curr, cnt in trans_count[prev].items():
                trans_prob[(prev, curr)] = cnt / trans_total[prev] * 100
        freq_hora = df.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100)
        prob_hora = {}
        for (hora, animal), prob in freq_hora.items():
            prob_hora.setdefault(hora, {})[animal] = prob
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion']
        available_numeric = [f for f in numeric_candidates if f in df.columns]
        animales_validos = list(self.animales_carac.keys())
        horas_del_dia = sorted(df['Hora'].unique())
        print("\n" + "=" * 74)
        print("  PREDICCION COMBINADA PARA HOY (Ensemble)")
        print("=" * 74)
        print(f"  Ultimo sorteo: {ultimo_animal} (#{ultimo_numero:02d}) a las {ultimo['Hora']}")
        print(f"  Modelos: Markov | Prob. Hora | ML {'SI' if modelo else 'NO'}")
        print("=" * 74)
        resultados = {}
        ml_top10_por_hora = {}
        if modelo is not None and le_y is not None and len(available_numeric) > 0:
            for hora_24h in horas_del_dia:
                try:
                    df_hora = df[df['Hora'] == hora_24h].iloc[[-1]].copy()
                    df_hora['Hora_Sorteo'] = hora_24h
                    X_query = df_hora[available_numeric + ['Hora_Sorteo']]
                    if not X_query.isnull().any().any():
                        y_proba = modelo.predict_proba(X_query)[0]
                        indices_top_k = np.argsort(y_proba)[::-1][:20]
                        ml_top10_por_hora[hora_24h] = set(le_y.inverse_transform(indices_top_k))
                except Exception:
                    pass
        for hora_24h in horas_del_dia:
            h_stripped = hora_24h.split(':')[0] + ':' + hora_24h.split(':')[1]
            try:
                hora_12h = datetime.strptime(h_stripped, '%H:%M').strftime('%I:%M %p').lstrip('0')
            except Exception:
                hora_12h = hora_24h
            markov_scores = {}
            if ultimo_animal in trans_total and trans_total[ultimo_animal] > 0:
                for animal in animales_validos:
                    p = trans_prob.get((ultimo_animal, animal), 0)
                    if p > 0:
                        markov_scores[animal] = p
            hourly_scores = prob_hora.get(hora_24h, {})
            ml_scores = {}
            ml_ok = hora_24h in ml_top10_por_hora
            if ml_ok:
                try:
                    df_hora = df[df['Hora'] == hora_24h].iloc[[-1]].copy()
                    df_hora['Hora_Sorteo'] = hora_24h
                    X_query = df_hora[available_numeric + ['Hora_Sorteo']]
                    if not X_query.isnull().any().any():
                        y_proba = modelo.predict_proba(X_query)[0]
                        for i, prob in enumerate(y_proba):
                            animal = le_y.inverse_transform([i])[0]
                            ml_scores[animal] = prob * 100
                except Exception:
                    ml_ok = False
            all_animals = set(list(markov_scores.keys()) + list(hourly_scores.keys()) + list(ml_scores.keys()))
            if not all_animals:
                all_animals = set(animales_validos[:20])
            max_m = max(markov_scores.values()) if markov_scores else 1
            max_h = max(hourly_scores.values()) if hourly_scores else 1
            max_ml = max(ml_scores.values()) if ml_scores else 1
            w_m, w_h, w_ml = (0.35, 0.35, 0.30) if ml_ok else (0.50, 0.50, 0)
            ensemble_scores = []
            ml_top10 = ml_top10_por_hora.get(hora_24h, set())
            for animal in all_animals:
                m = markov_scores.get(animal, 0)
                h = hourly_scores.get(animal, 0)
                ml = ml_scores.get(animal, 0)
                m_norm = m / max_m if max_m > 0 else 0
                h_norm = h / max_h if max_h > 0 else 0
                ml_norm = ml / max_ml if max_ml > 0 else 0
                models_cnt = (1 if m > 0 else 0) + (1 if h > 0 else 0) + (1 if animal in ml_top10 else 0)
                score = m_norm * w_m + h_norm * w_h + ml_norm * w_ml
                if models_cnt >= 2 and ml_ok:
                    score *= 1.15
                ensemble_scores.append((animal, score, m, h, ml, models_cnt))
            ensemble_scores.sort(key=lambda x: x[1], reverse=True)
            topk = ensemble_scores[:k]
            resultados[hora_24h] = topk
            print(f"\nHora {hora_12h}")
            print(f"   {'#':<3} {'Animal':<14} {'Ensemble':<9} {'Markov':<7} {'Hist':<7} {'ML':<8} {'M':<3}")
            print(f"   {'-'*53}")
            for i, (animal, ens, m, h, ml, mc) in enumerate(topk, 1):
                ms = f"{m:.1f}%" if m > 0 else "-"
                hs = f"{h:.1f}%" if h > 0 else "-"
                mls = f"{ml:.1f}%" if ml > 0 else "-"
                conf = "*" * mc if mc >= 2 else ""
                print(f"   {i:<3} {animal:<14} {ens*100:<9.1f} {ms:<7} {hs:<7} {mls:<8} {conf:<3}")
        pred_matrix = {h: [a[0] for a in r] for h, r in resultados.items()}
        return pred_matrix

    def prediccion_completa_hoy(self, datos, modelo_rf=None, le_rf=None, modelo_xgb=None, le_xgb=None):
        from collections import defaultdict
        df = datos.copy()
        if len(df) < 5:
            print("Pocos datos")
            return
        ultimo = df.iloc[-1]
        hora_actual = ultimo['Hora']
        horas_del_dia = sorted(df['Hora'].unique())
        if hora_actual in horas_del_dia:
            idx = horas_del_dia.index(hora_actual)
            if idx + 1 < len(horas_del_dia):
                hora_target = horas_del_dia[idx + 1]
            else:
                print(f"Ya paso la ultima hora del dia ({hora_actual}). No hay mas sorteos hoy.")
                return
        else:
            hora_target = horas_del_dia[0]
        try:
            target_12h = datetime.strptime(hora_target, '%H:%M:%S').strftime('%I:%M %p').lstrip('0')
        except Exception:
            target_12h = hora_target
        print(f"\n{'='*94}")
        print(f"  PREDICCION PARA LA SIGUIENTE HORA: {target_12h}")
        print(f"  Ultimo resultado: {ultimo['Animal']} (#{int(ultimo['Numero']):02d}) a las {ultimo['Hora']}")
        print(f"{'='*94}")
        trans_count = defaultdict(lambda: defaultdict(int))
        trans_total = defaultdict(int)
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                prev = df.iloc[i-1]['Animal']
                curr = df.iloc[i]['Animal']
                trans_count[prev][curr] += 1
                trans_total[prev] += 1
        trans_prob = {}
        for prev in trans_count:
            for curr, cnt in trans_count[prev].items():
                trans_prob[(prev, curr)] = cnt / trans_total[prev] * 100
        freq_hora = df.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100)
        prob_hora = {}
        for (hora, animal), prob in freq_hora.items():
            prob_hora.setdefault(hora, {})[animal] = prob
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion']
        available_numeric = [f for f in numeric_candidates if f in df.columns]
        animales_validos = list(self.animales_carac.keys())
        markov_scores = {}
        if ultimo['Animal'] in trans_total:
            for a in animales_validos:
                p = trans_prob.get((ultimo['Animal'], a), 0)
                if p > 0: markov_scores[a] = p
        markov_top = sorted(markov_scores, key=markov_scores.get, reverse=True)[:20]
        hourly_scores = prob_hora.get(hora_target, {})
        hourly_top = sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:20]
        rf_list = []
        xgb_list = []
        if modelo_rf and available_numeric:
            try:
                df_hora = df[df['Hora'] == hora_target].iloc[[-1]].copy()
                df_hora['Hora_Sorteo'] = hora_target
                X_q = df_hora[available_numeric + ['Hora_Sorteo']]
                if not X_q.isnull().any().any():
                    yp = modelo_rf.predict_proba(X_q)[0]
                    rf_list = le_rf.inverse_transform(np.argsort(yp)[::-1][:20]).tolist()
                    rf_scores = {a: yp[i]*100 for i, a in enumerate(le_rf.classes_)}
            except Exception:
                rf_scores = {}
        else:
            rf_scores = {}
        if modelo_xgb and available_numeric:
            try:
                df_hora = df[df['Hora'] == hora_target].iloc[[-1]].copy()
                df_hora['Hora_Sorteo'] = hora_target
                X_q = df_hora[available_numeric + ['Hora_Sorteo']]
                if not X_q.isnull().any().any():
                    yp = modelo_xgb.predict_proba(X_q)[0]
                    xgb_list = le_xgb.inverse_transform(np.argsort(yp)[::-1][:20]).tolist()
                    xgb_scores = {a: yp[i]*100 for i, a in enumerate(le_xgb.classes_)}
            except Exception:
                xgb_scores = {}
        else:
            xgb_scores = {}
        print(f"\n  {'#':<3} {'Markov':<18} {'Hist Hora':<18} {'RF':<20} {'XGB':<20}")
        print(f"  {'-'*80}")
        for i in range(20):
            m = f"{markov_top[i]:<10} {markov_scores[markov_top[i]]:.1f}%" if i < len(markov_top) else "-"
            h = f"{hourly_top[i]:<10} {hourly_scores[hourly_top[i]]:.1f}%" if i < len(hourly_top) else "-"
            r = f"{rf_list[i]:<10} {rf_scores.get(rf_list[i],0):.1f}%" if i < len(rf_list) and rf_list[i] in rf_scores else rf_list[i] if i < len(rf_list) else "-"
            x = f"{xgb_list[i]:<10} {xgb_scores.get(xgb_list[i],0):.1f}%" if i < len(xgb_list) and xgb_list[i] in xgb_scores else xgb_list[i] if i < len(xgb_list) else "-"
            print(f"  {i+1:<3} {m:<22} {h:<22} {r:<24} {x:<24}")
        all_animals = set(markov_top) | set(hourly_top) | set(rf_list) | set(xgb_list)
        scored = []
        for a in all_animals:
            cnt = sum([1 for lst in [markov_top, hourly_top, rf_list, xgb_list] if a in lst])
            score = 0
            if a in markov_top: score += (20 - markov_top.index(a)) * markov_scores.get(a, 0)
            if a in hourly_top: score += (20 - hourly_top.index(a)) * hourly_scores.get(a, 0)
            if a in rf_list: score += (20 - rf_list.index(a)) * rf_scores.get(a, 0)
            if a in xgb_list: score += (20 - xgb_list.index(a)) * xgb_scores.get(a, 0)
            scored.append((a, cnt, score))
        scored.sort(key=lambda x: (-x[1], -x[2]))
        print(f"\n  >>> RESUMEN: Interseccion de Modelos para {target_12h} <<<")
        print(f"  {'#':<3} {'Animal':<12} {'#Modelos':<9} {'Modelos':<12} {'Top en':<24}")
        print(f"  {'-'*60}")
        for i, (a, cnt, _) in enumerate(scored[:20], 1):
            mods = []
            pos = []
            if a in markov_top: mods.append("M"); pos.append(f"M#{markov_top.index(a)+1}")
            if a in hourly_top: mods.append("H"); pos.append(f"H#{hourly_top.index(a)+1}")
            if a in rf_list: mods.append("RF"); pos.append(f"RF#{rf_list.index(a)+1}")
            if a in xgb_list: mods.append("X"); pos.append(f"X#{xgb_list.index(a)+1}")
            print(f"  {i:<3} {a:<12} {cnt}/4{'':<6} {','.join(mods):<12} {','.join(pos):<24}")
        print(f"\n  Jugada recomendada: Top-3 con mayor consenso")
        return None

    def evaluar_predicciones_historicas(self, datos, modelo_rf=None, le_rf=None, modelo_xgb=None, le_xgb=None, n_ultimos=50):
        from collections import defaultdict
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        df_eval = df.tail(n_ultimos + 5).reset_index(drop=True)
        if 'Hora_Sorteo' not in df_eval.columns:
            df_eval['Hora_Sorteo'] = df_eval['Hora'].astype(str).str.strip().str.zfill(8)
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion']
        available_numeric = [f for f in numeric_candidates if f in df_eval.columns]
        animales_validos = list(self.animales_carac.keys())
        trans_count = defaultdict(lambda: defaultdict(int))
        trans_total = defaultdict(int)
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                prev = df.iloc[i-1]['Animal']
                curr = df.iloc[i]['Animal']
                trans_count[prev][curr] += 1
                trans_total[prev] += 1
        trans_prob = {}
        for prev in trans_count:
            for curr, cnt in trans_count[prev].items():
                trans_prob[(prev, curr)] = cnt / trans_total[prev] * 100
        freq_hora = df.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100)
        resultados = []
        rf_count = 0
        xgb_count = 0
        for i in range(4, len(df_eval) - 1):
            if i + 1 >= len(df_eval):
                break
            if df_eval.iloc[i]['Fecha'] != df_eval.iloc[i + 1]['Fecha']:
                continue
            prev_state = df_eval.iloc[i]
            actual = df_eval.iloc[i + 1]
            animal_real = actual['Animal']
            hora_real = actual['Hora']
            fecha = actual['Fecha']
            markov_scores = {}
            ultimo_animal = prev_state['Animal']
            if ultimo_animal in trans_total and trans_total[ultimo_animal] > 0:
                for a in animales_validos:
                    p = trans_prob.get((ultimo_animal, a), 0)
                    if p > 0:
                        markov_scores[a] = p
            markov_top = sorted(markov_scores, key=markov_scores.get, reverse=True)[:20]
            markov_rank = markov_top.index(animal_real) + 1 if animal_real in markov_top else None
            markov_full = sorted(markov_scores, key=markov_scores.get, reverse=True)
            markov_all_rank = markov_full.index(animal_real) + 1 if animal_real in markov_full else None
            hourly_scores = {}
            if hora_real in freq_hora.index:
                for a, p in freq_hora[hora_real].items():
                    hourly_scores[a] = p
            hourly_top = sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:20]
            hourly_rank = hourly_top.index(animal_real) + 1 if animal_real in hourly_top else None
            hourly_full = sorted(hourly_scores, key=hourly_scores.get, reverse=True)
            hourly_all_rank = hourly_full.index(animal_real) + 1 if animal_real in hourly_full else None
            max_markov = max(markov_scores.values()) if markov_scores else 1
            combined_mh_scores = {}
            for a in animales_validos:
                mp = markov_scores.get(a, 0) / max_markov * 100
                hp = hourly_scores.get(a, 0)
                combined_mh_scores[a] = mp + hp
            combined_mh_top = sorted(combined_mh_scores, key=combined_mh_scores.get, reverse=True)[:20]
            combined_mh_rank = combined_mh_top.index(animal_real) + 1 if animal_real in combined_mh_top else None
            rf_top = []
            rf_rank = None
            rf_all_rank = None
            if modelo_rf is not None and le_rf is not None and len(available_numeric) > 0:
                try:
                    X_dict = prev_state[available_numeric + ['Hora_Sorteo']].to_dict()
                    X = pd.DataFrame([X_dict])
                    if not X.isnull().any().any():
                        rf_count += 1
                        y_proba = modelo_rf.predict_proba(X)[0]
                        indices = np.argsort(y_proba)[::-1]
                        rf_top = le_rf.inverse_transform(indices[:20]).tolist()
                        rf_all = le_rf.inverse_transform(indices).tolist()
                        rf_rank = rf_top.index(animal_real) + 1 if animal_real in rf_top else None
                        rf_all_rank = rf_all.index(animal_real) + 1 if animal_real in rf_all else None
                except Exception:
                    pass
            xgb_top = []
            xgb_rank = None
            xgb_all_rank = None
            if modelo_xgb is not None and le_xgb is not None and len(available_numeric) > 0:
                try:
                    X_dict = prev_state[available_numeric + ['Hora_Sorteo']].to_dict()
                    X = pd.DataFrame([X_dict])
                    if not X.isnull().any().any():
                        xgb_count += 1
                        y_proba = modelo_xgb.predict_proba(X)[0]
                        indices = np.argsort(y_proba)[::-1]
                        xgb_top = le_xgb.inverse_transform(indices[:20]).tolist()
                        xgb_all = le_xgb.inverse_transform(indices).tolist()
                        xgb_rank = xgb_top.index(animal_real) + 1 if animal_real in xgb_top else None
                        xgb_all_rank = xgb_all.index(animal_real) + 1 if animal_real in xgb_all else None
                except Exception:
                    pass
            combined_rf_xgb_scores = {}
            if rf_top and xgb_top:
                rf_set = set(rf_top)
                xgb_set = set(xgb_top)
                for a in set(rf_top + xgb_top):
                    score = 0
                    if a in rf_top:
                        score += 20 - rf_top.index(a)
                    if a in xgb_top:
                        score += 20 - xgb_top.index(a)
                    combined_rf_xgb_scores[a] = score
                combined_rf_xgb_top = sorted(combined_rf_xgb_scores, key=combined_rf_xgb_scores.get, reverse=True)[:20]
                combined_rf_xgb_rank = combined_rf_xgb_top.index(animal_real) + 1 if animal_real in combined_rf_xgb_top else None
            else:
                combined_rf_xgb_top = []
                combined_rf_xgb_rank = None
            top1 = xgb_top[0] if xgb_top else rf_top[0] if rf_top else combined_mh_top[0] if combined_mh_top else markov_top[0] if markov_top else hourly_top[0] if hourly_top else "?"
            acertado = animal_real in (xgb_top or rf_top or combined_mh_top or markov_top or hourly_top)
            resultados.append({
                'fecha': fecha, 'hora': hora_real, 'real': animal_real,
                'predicho': top1, 'acertado': acertado,
                'markov_rank': markov_rank, 'markov_all_rank': markov_all_rank,
                'hourly_rank': hourly_rank, 'hourly_all_rank': hourly_all_rank,
                'combined_mh_rank': combined_mh_rank,
                'rf_rank': rf_rank, 'rf_all_rank': rf_all_rank,
                'xgb_rank': xgb_rank, 'xgb_all_rank': xgb_all_rank,
                'combined_rf_xgb_rank': combined_rf_xgb_rank,
            })
        print(f"\n{'='*90}")
        print(f"  EVALUACION AUTOMATICA: Prediccion vs Realidad")
        print(f"  Ultimos {len(resultados)} sorteos analizados")
        print(f"{'='*90}")
        print(f"{'Fecha':<12} {'Hora':<6} {'Predicho':<13} {'Real':<13} {'Hit':<4} {'Rk-M':<5} {'Rk-H':<5} {'Rk-C(M+H)':<10} {'Rk-RF':<6} {'Rk-XGB':<7} {'C(R+X)':<7}")
        print(f"{'-'*90}")
        for r in resultados[-20:]:
            ac = "OK" if r['acertado'] else "NO"
            rk_m = str(r['markov_rank']) if r['markov_rank'] else "-"
            rk_h = str(r['hourly_rank']) if r['hourly_rank'] else "-"
            rk_c = str(r['combined_mh_rank']) if r['combined_mh_rank'] else "-"
            rk_rf = str(r['rf_rank']) if r['rf_rank'] else "-"
            rk_x = str(r['xgb_rank']) if r['xgb_rank'] else "-"
            rk_crx = str(r['combined_rf_xgb_rank']) if r['combined_rf_xgb_rank'] else "-"
            print(f"{str(r['fecha']):<12} {r['hora']:<6} {r['predicho']:<13} {r['real']:<13} {ac:<4} {rk_m:<5} {rk_h:<5} {rk_c:<10} {rk_rf:<6} {rk_x:<7} {rk_crx:<7}")
        total = len(resultados)
        if total > 0:
            markov_hits = sum(1 for r in resultados if r['markov_rank'] is not None)
            hourly_hits = sum(1 for r in resultados if r['hourly_rank'] is not None)
            combined_mh_hits = sum(1 for r in resultados if r['combined_mh_rank'] is not None)
            rf_hits = sum(1 for r in resultados if r['rf_rank'] is not None)
            xgb_hits = sum(1 for r in resultados if r['xgb_rank'] is not None)
            combined_rx_hits = sum(1 for r in resultados if r['combined_rf_xgb_rank'] is not None)
            print(f"\nPRECISION TOP-20 por modelo:")
            print(f"  Markov (M):              {markov_hits}/{total} = {markov_hits/total*100:.1f}%")
            print(f"  Hist. Hora (H):          {hourly_hits}/{total} = {hourly_hits/total*100:.1f}%")
            print(f"  M + H:                   {combined_mh_hits}/{total} = {combined_mh_hits/total*100:.1f}%")
            if rf_count > 0:
                print(f"  Random Forest:           {rf_hits}/{rf_count} = {rf_hits/rf_count*100:.1f}%")
            if xgb_count > 0:
                print(f"  XGBoost:                 {xgb_hits}/{xgb_count} = {xgb_hits/xgb_count*100:.1f}%")
            if rf_count > 0 and xgb_count > 0:
                print(f"  RF + XGB:                {combined_rx_hits}/{total} = {combined_rx_hits/total*100:.1f}%")
        # ranks = {'markov_all_rank': 'Markov', 'hourly_all_rank': 'Hist Hora', 'rf_all_rank': 'Random Forest', 'xgb_all_rank': 'XGBoost'}
        # for key, nombre in ranks.items():
        #     dist = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, '11-15':0, '16-20':0, '21-38':0}
        #     count = 0
        #     for r in resultados:
        #         v = r.get(key)
        #         if v is not None:
        #             count += 1
        #             if v <= 10:
        #                 dist[v] += 1
        #             elif v <= 15:
        #                 dist['11-15'] += 1
        #             elif v <= 20:
        #                 dist['16-20'] += 1
        #             else:
        #                 dist['21-38'] += 1
        #     if count > 0:
        #         print(f"\n  DISTRIBUCION DE ACIERTOS - {nombre} ({count} evaluados):")
        #         print(f"  {'Rango':<10} {'Aciertos':<10} {'%':<8} {'Acumulado':<10}")
        #         print(f"  {'-'*40}")
        #         acum = 0
        #         for rng in [1,2,3,4,5,6,7,8,9,10,'11-15','16-20','21-38']:
        #             v = dist[rng]
        #             acum += v
        #             pct = v/count*100
        #             ac_pct = acum/count*100
        #             if isinstance(rng, int):
        #                 print(f"  #{rng:<8} {v:<10} {pct:<8.1f} {ac_pct:<10.1f}")
        #             else:
        #                 print(f"  {rng:<8} {v:<10} {pct:<8.1f} {ac_pct:<10.1f}")
        #         print(f"  {'Top-5':<10} {sum(dist[i] for i in range(1,6)):<10} {sum(dist[i] for i in range(1,6))/count*100:<8.1f} -")
        #         print(f"  {'Top-10':<10} {sum(dist[i] for i in range(1,11)):<10} {sum(dist[i] for i in range(1,11))/count*100:<8.1f} -")
        #         print(f"  {'Top-15':<10} {sum(dist[i] for i in range(1,11))+dist['11-15']:<10} {(sum(dist[i] for i in range(1,11))+dist['11-15'])/count*100:<8.1f} -")
        #         print(f"  {'Top-20':<10} {sum(dist[i] for i in range(1,11))+dist['11-15']+dist['16-20']:<10} {(sum(dist[i] for i in range(1,11))+dist['11-15']+dist['16-20'])/count*100:<8.1f} -")
        print(f"\n{'='*90}\n")
        return resultados

    def analizar_aciertos_por_dia_semana(self, datos):
        from collections import defaultdict
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        animales_validos = list(self.animales_carac.keys())
        trans_count = defaultdict(lambda: defaultdict(int))
        trans_total = defaultdict(int)
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                prev = df.iloc[i-1]['Animal']
                curr = df.iloc[i]['Animal']
                trans_count[prev][curr] += 1
                trans_total[prev] += 1
        trans_prob = {}
        for prev in trans_count:
            for curr, cnt in trans_count[prev].items():
                trans_prob[(prev, curr)] = cnt / trans_total[prev] * 100
        freq_hora = df.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100)
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        DIA_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        resultados = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            animal_real = actual['Animal']
            hora_real = actual['Hora']
            dia = df.iloc[i]['Dia_Semana']
            markov_scores = {}
            ultimo_animal = prev_state['Animal']
            if ultimo_animal in trans_total and trans_total[ultimo_animal] > 0:
                for a in animales_validos:
                    p = trans_prob.get((ultimo_animal, a), 0)
                    if p > 0:
                        markov_scores[a] = p
            markov_top = sorted(markov_scores, key=markov_scores.get, reverse=True)[:20]
            markov_hit = animal_real in markov_top
            hourly_scores = {}
            if hora_real in freq_hora.index:
                for a, p in freq_hora[hora_real].items():
                    hourly_scores[a] = p
            hourly_top = sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:20]
            hourly_hit = animal_real in hourly_top
            max_m = max(markov_scores.values()) if markov_scores else 1
            combined = {}
            for a in animales_validos:
                mp = markov_scores.get(a, 0) / max_m * 100
                hp = hourly_scores.get(a, 0)
                combined[a] = mp + hp
            combined_top = sorted(combined, key=combined.get, reverse=True)[:20]
            combined_hit = animal_real in combined_top
            resultados.append({
                'dia': dia, 'markov': markov_hit, 'hourly': hourly_hit, 'combined': combined_hit
            })
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            print("No se generaron resultados")
            return
        print(f"\n{'='*70}")
        print(f"  ACIERTOS POR DIA DE LA SEMANA (Top-20)")
        print(f"  Basado en {len(df_res)} sorteos analizados")
        print(f"{'='*70}")
        header = f"{'Dia':<12} {'Markov':<10} {'Hora':<10} {'Combinado':<12} {'Sorteos':<8}"
        print(f"\n{header}")
        print(f"{'-'*52}")
        for dia_en in DIA_ORDER:
            sub = df_res[df_res['dia'] == dia_en]
            if len(sub) == 0:
                continue
            mk = sub['markov'].mean() * 100
            hr = sub['hourly'].mean() * 100
            cb = sub['combined'].mean() * 100
            nombre = DIA_NOMBRE.get(dia_en, dia_en)
            print(f"{nombre:<12} {mk:<10.1f}% {hr:<10.1f}% {cb:<12.1f}% {len(sub):<8}")
        print(f"\n  Resumen global:")
        print(f"  Markov:    {df_res['markov'].mean()*100:.1f}%")
        print(f"  Hora:      {df_res['hourly'].mean()*100:.1f}%")
        print(f"  Combinado: {df_res['combined'].mean()*100:.1f}%")
        print(f"\n  Mejor dia por modelo:")
        for col, label in [('markov','Markov'), ('hourly','Hora'), ('combined','Combinado')]:
            best = df_res.groupby('dia')[col].mean().idxmax()
            best_val = df_res.groupby('dia')[col].mean().max() * 100
            print(f"     {label}: {DIA_NOMBRE.get(best, best)} ({best_val:.1f}%)")
        print(f"\n{'='*70}\n")
        return df_res

    def analizar_aciertos_por_hora(self, datos):
        from collections import defaultdict
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        animales_validos = list(self.animales_carac.keys())
        trans_count = defaultdict(lambda: defaultdict(int))
        trans_total = defaultdict(int)
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                prev = df.iloc[i-1]['Animal']
                curr = df.iloc[i]['Animal']
                trans_count[prev][curr] += 1
                trans_total[prev] += 1
        trans_prob = {}
        for prev in trans_count:
            for curr, cnt in trans_count[prev].items():
                trans_prob[(prev, curr)] = cnt / trans_total[prev] * 100
        freq_hora = df.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100)
        resultados = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            animal_real = actual['Animal']
            hora_real = actual['Hora']
            markov_scores = {}
            ultimo_animal = prev_state['Animal']
            if ultimo_animal in trans_total and trans_total[ultimo_animal] > 0:
                for a in animales_validos:
                    p = trans_prob.get((ultimo_animal, a), 0)
                    if p > 0:
                        markov_scores[a] = p
            markov_top = sorted(markov_scores, key=markov_scores.get, reverse=True)[:20]
            markov_hit = animal_real in markov_top
            hourly_scores = {}
            if hora_real in freq_hora.index:
                for a, p in freq_hora[hora_real].items():
                    hourly_scores[a] = p
            hourly_top = sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:20]
            hourly_hit = animal_real in hourly_top
            max_m = max(markov_scores.values()) if markov_scores else 1
            combined = {}
            for a in animales_validos:
                mp = markov_scores.get(a, 0) / max_m * 100
                hp = hourly_scores.get(a, 0)
                combined[a] = mp + hp
            combined_top = sorted(combined, key=combined.get, reverse=True)[:20]
            combined_hit = animal_real in combined_top
            resultados.append({
                'hora': hora_real, 'markov': markov_hit, 'hourly': hourly_hit, 'combined': combined_hit
            })
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            print("No se generaron resultados")
            return
        HORA_ORDER = ['08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00',
                      '14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00']

        def hora_label(h):
            hh = int(h.split(':')[0])
            period = 'AM' if hh < 12 else 'PM'
            if hh > 12:
                hh -= 12
            if hh == 0:
                hh = 12
            return f'{hh:02d}:00 {period}'

        print(f"\n{'='*70}")
        print(f"  ACIERTOS POR HORA (Top-20)")
        print(f"  Basado en {len(df_res)} sorteos analizados")
        print(f"{'='*70}")
        header = f"{'Hora':<12} {'Markov':<10} {'Hora':<10} {'Combinado':<12} {'Sorteos':<8}"
        print(f"\n{header}")
        print(f"{'-'*52}")
        for h in HORA_ORDER:
            sub = df_res[df_res['hora'] == h]
            if len(sub) == 0:
                continue
            mk = sub['markov'].mean() * 100
            hr = sub['hourly'].mean() * 100
            cb = sub['combined'].mean() * 100
            print(f"{hora_label(h):<12} {mk:<10.1f}% {hr:<10.1f}% {cb:<12.1f}% {len(sub):<8}")
        print(f"\n  Resumen global:")
        print(f"  Markov:    {df_res['markov'].mean()*100:.1f}%")
        print(f"  Hora:      {df_res['hourly'].mean()*100:.1f}%")
        print(f"  Combinado: {df_res['combined'].mean()*100:.1f}%")
        print(f"\n  Mejores horas por modelo:")
        for col, label in [('markov','Markov'), ('hourly','Hora'), ('combined','Combinado')]:
            best = df_res.groupby('hora')[col].mean().idxmax()
            best_val = df_res.groupby('hora')[col].mean().max() * 100
            print(f"     {label}: {hora_label(best)} ({best_val:.1f}%)")
        print(f"\n  Ranking Combinado (Top-3 mejores horas):")
        top3 = df_res.groupby('hora')['combined'].mean().sort_values(ascending=False).head(3)
        for h, v in top3.items():
            print(f"     {hora_label(h)} -> {v*100:.1f}%")
        print(f"\n{'='*70}\n")
        return df_res

    def analizar_patrones_sorteo(self, datos):
        from collections import Counter, defaultdict
        df = datos.copy()
        animales_validos = list(self.animales_carac.keys())
        print(f"\n{'='*70}")
        print(f"  PATRONES DEL SORTEO")
        print(f"{'='*70}")
        df['Grupo'] = df['Animal'].map(self.animal_a_grupo)
        print(f"\nPERFIL DEL DIA (categorias)")
        fechas_completas = df.groupby('Fecha').size()
        fechas_completas = fechas_completas[fechas_completas == 12].index
        df_full = df[df['Fecha'].isin(fechas_completas)]
        print(f"  Basado en {len(fechas_completas)} dias con 12 sorteos:")
        print(f"  {'Categoria':<12} {'% dias':<7} {'Promedio':<9} {'Rango':<10}")
        print(f"  {'-'*38}")
        for grupo in ["MAMIFERO","AVE","ACUATICO","REPTIL","INSECTO"]:
            gc = df_full[df_full['Grupo']==grupo].groupby('Fecha').size()
            pct = len(gc)/len(fechas_completas)*100
            media = gc.mean()
            rango = f"{gc.min()}-{gc.max()}"
            print(f"  {grupo:<12} {pct:<7.0f}% {media:<9.1f} {rango:<10}")
        print(f"\nTRANSICIONES ENTRE CATEGORIAS (A -> B)")
        trans_grupo = Counter()
        for i in range(1, len(df)):
            g_prev = df.iloc[i-1]['Grupo']
            g_curr = df.iloc[i]['Grupo']
            trans_grupo[(g_prev, g_curr)] += 1
        total_tg = sum(trans_grupo.values())
        print(f"{'De':<12} {'A':<12} {'Veces':<7} {'Prob':<7}")
        print(f"{'-'*38}")
        for (a, b), c in trans_grupo.most_common(10):
            print(f"{a:<12} {b:<12} {c:<7} {c/total_tg*100:.1f}%")
        misma_cat = sum(c for (a,b),c in trans_grupo.items() if a==b)
        print(f"\n  Misma categoria seguida: {misma_cat}/{total_tg} ({misma_cat/total_tg*100:.1f}%)")
        print(f"\nESTADO DEL DIA DE HOY")
        hoy = date.today()
        df_hoy = df[df['Fecha'] == hoy].sort_values('Hora')
        if not df_hoy.empty:
            grupos_hoy = set(df_hoy['Grupo'])
            print(f"  Grupos que han salido: {', '.join(sorted(grupos_hoy))}")
            faltan = set(self.config['grupos_animales'].keys()) - grupos_hoy
            if faltan:
                print(f"  Grupos que FALTAN por salir: {', '.join(sorted(faltan))}")
            else:
                print(f"  Ya salieron todos los grupos")
            print(f"  Ultimo sorteo: {df_hoy.iloc[-1]['Animal']} ({df_hoy.iloc[-1]['Grupo']})")
        else:
            print(f"  Aun no hay sorteos registrados hoy ({hoy})")
        print(f"\nANIMALES MAS FRECUENTES POR HORA")
        for hora in sorted(df['Hora'].unique()):
            top3 = df[df['Hora'] == hora]['Animal'].value_counts().head(3)
            top3_str = ', '.join([f"{a} ({c})" for a, c in top3.items()])
            print(f"  {hora:<8} -> {top3_str}")
        ultimos_50 = df.tail(50)['Animal'].value_counts()
        frios = [a for a in animales_validos if a not in ultimos_50.index]
        if frios:
            print(f"\nANIMALES FRIOS (sin aparecer en ultimos 50 sorteos): {len(frios)}")
            print(f"  {', '.join(frios)}")
        else:
            print(f"\nANIMALES FRIOS: Ninguno (todos han salido en ultimos 50)")
        ultimos_30 = df.tail(30)['Animal'].value_counts()
        esperado = 30 / len(animales_validos)
        calientes = ultimos_30[ultimos_30 > esperado * 1.5].head(10)
        if not calientes.empty:
            print(f"\nANIMALES CALIENTES (ultimos 30 sorteos, +50% sobre esperado)")
            print(f"  Esperado: {esperado:.1f} apariciones por animal")
            for a, c in calientes.items():
                print(f"  {a:<14} {c} apariciones ({c/esperado*100-100:.0f}% sobre lo esperado)")
        print(f"\nPARES QUE APARECEN EL MISMO DIA (TOP-10)")
        pares_dia = Counter()
        for fecha, grupo in df.groupby('Fecha'):
            animales_dia = grupo['Animal'].unique()
            for i in range(len(animales_dia)):
                for j in range(i+1, len(animales_dia)):
                    par = tuple(sorted([animales_dia[i], animales_dia[j]]))
                    pares_dia[par] += 1
        for par, cnt in pares_dia.most_common(10):
            print(f"  {par[0]:<14} + {par[1]:<14} -> {cnt} dias")
        print(f"\n{'='*70}\n")

    def generar_matriz_probabilidad(self, datos):
        datos['Animal_Siguiente'] = datos['Animal'].shift(-1)
        datos['Solo_Fecha'] = datos['Timestamp'].dt.date
        datos['Es_Ultimo_Sorteo_del_Dia'] = datos.groupby('Solo_Fecha')['Timestamp'].transform('max') == datos['Timestamp']
        df_transiciones = datos[datos['Es_Ultimo_Sorteo_del_Dia'] == False].copy()
        matriz_conteo = pd.crosstab(df_transiciones['Animal'], df_transiciones['Animal_Siguiente'], normalize=False)
        matriz_probabilidad = matriz_conteo.div(matriz_conteo.sum(axis=1), axis=0) * 100
        return matriz_probabilidad.fillna(0)

    def matriz_probabilidad_transicion(self, datos):
        print("\nTOP-10 POR ANIMAL (Matriz de Transicion Markov)")
        print("   Para cada animal, los 10 mas probables que le siguen:\n")
        matriz = self.generar_matriz_probabilidad(datos.copy())
        for animal in matriz.index:
            top10 = matriz.loc[animal].sort_values(ascending=False).head(10)
            top10 = top10[top10 > 0]
            if top10.empty:
                continue
            print(f"\n  {animal:<14} -> Top 10 siguientes:")
            for i, (a, p) in enumerate(top10.items(), 1):
                print(f"     {i:2d}. {a:<14} ({p:.1f}%)")

    def mejor_prediccion_siguiente(self, datos):
        print("\nPrediccion Siguiente en Tiempo Real (TOP-5 Markov)")
        animal_actual = input("Ingresa el Animal que acaba de salir (ej: PERRO): ").strip().upper()
        matriz_probabilidad = self.generar_matriz_probabilidad(datos.copy())
        if animal_actual not in matriz_probabilidad.index:
            print(f"Error: El animal '{animal_actual}' no se encontro en el historial de transiciones.")
            return
        fila_prediccion = matriz_probabilidad.loc[animal_actual].sort_values(ascending=False)
        top_5 = fila_prediccion.head(5)
        print(f"\nResultado de la Prediccion (TOP 5)")
        print(f"Si acaba de salir {animal_actual}, los 5 mas probables son:")
        resultados = []
        for animal, prob in top_5.items():
            resultados.append({'Animal': animal, 'Probabilidad (%)': f"{prob:.2f}"})
        df_resultados = pd.DataFrame(resultados)
        print(df_resultados.to_string(index=False))
        mejor_animal = top_5.index[0]
        probabilidad_max = top_5.iloc[0]
        print(f"\nMaxima probabilidad individual: {mejor_animal} ({probabilidad_max:.2f}%)")

    def probabilidad_maxima_por_hora(self, datos):
        print("\nAnalisis de Frecuencia Historica por Hora (TOP-10)")
        frecuencia_completa = datos.groupby('Solo_hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        total_sorteos_por_hora = datos.groupby('Solo_hora').size().reset_index(name='Total_Sorteos')
        horas_unicas = frecuencia_completa['Solo_hora'].unique()
        print("\nTop 10 Animales por Cada Hora (Para Apuesta Diaria)")
        for hora in sorted(horas_unicas):
            df_hora = frecuencia_completa[frecuencia_completa['Solo_hora'] == hora].copy()
            total_sorteos = total_sorteos_por_hora[total_sorteos_por_hora['Solo_hora'] == hora]['Total_Sorteos'].iloc[0]
            top_10 = df_hora.sort_values(by='Probabilidad', ascending=False).head(10)
            print(f"\nHORA: {hora} (Total Sorteos: {total_sorteos})")
            print(top_10[['Animal', 'Probabilidad']].to_string(index=False, float_format="%.2f%%"))

    def prediccion_markov_hora(self, datos):
        from collections import defaultdict
        df = datos.copy()
        print("\n" + "=" * 74)
        print("  PREDICCION COMBINADA MARKOV + HORA")
        print("=" * 74)
        print("  Ranking = Prob_Markov + Prob_Historica_Hora")
        print("  Precision estimada: ~44% Top-10 (vs 43% Markov solo)\n")
        trans = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                trans[df.iloc[i-1]['Animal']][df.iloc[i]['Animal']] += 1
        hora_freq = df.groupby('Solo_hora')['Animal'].value_counts(normalize=True)
        ultimo = df.iloc[-1]
        ultimo_animal = ultimo['Animal']
        ultimo_hora = ultimo['Solo_hora']
        print(f"  Ultimo: {ultimo_animal} a las {ultimo_hora}\n")
        if ultimo_animal not in trans:
            print("  Sin datos de transicion para este animal.")
            return
        items = sorted(trans[ultimo_animal].items(), key=lambda x: x[1], reverse=True)
        max_c = items[0][1]
        scored = []
        for animal, cnt in trans[ultimo_animal].items():
            mp = cnt / max_c * 100
            hp = hora_freq.get((ultimo_hora, animal), 0) * 100
            scored.append((mp + hp, mp, hp, animal))
        scored.sort(reverse=True)
        print(f"  {'#':<3} {'Animal':<14} {'Score':<7} {'Markov':<7} {'+Hora':<7}")
        print(f"  {'-'*42}")
        for i, (sc, mp, hp, animal) in enumerate(scored[:20], 1):
            print(f"  {i:<3} {animal:<14} {sc:<7.1f} {mp:<7.1f}% {hp:<7.1f}%")
        print(f"\n  (Mostrando Top-20 de {len(scored)} animales posibles)")
        print(f"\n  {'='*74}")
        print(f"  PREDICCION POR CADA HORA DEL DIA (Top-5 combinado)")
        print(f"  {'='*74}")
        horas = sorted(df['Solo_hora'].unique())
        for hora in horas:
            if hora <= ultimo_hora:
                continue
            h_scored = []
            for animal in [a for _,_,_,a in scored[:20]]:
                hp = hora_freq.get((hora, animal), 0) * 100
                mp = next((c/max_c*100 for a,c in items if a==animal), 0)
                h_scored.append((mp + hp, animal))
            h_scored.sort(reverse=True)
            top5 = ', '.join(f"{a} ({s:.0f})" for s,a in h_scored[:5])
            print(f"  {hora:<10} -> {top5}")

    def validar_modelo_markov(self, datos, porcentaje_entrenamiento=0.8, top_k=5):
        print(f"\nValidacion Cruzada del Modelo de Markov (TOP-{top_k})")
        total_sorteos = len(datos)
        corte = int(total_sorteos * porcentaje_entrenamiento)
        df_entrenamiento = datos.iloc[:corte].copy()
        df_prueba = datos.iloc[corte:].copy()
        print(f"Total de sorteos: {total_sorteos}")
        print(f"Sorteos de Entrenamiento ({porcentaje_entrenamiento*100:.0f}%): {len(df_entrenamiento)}")
        print(f"Sorteos de Prueba ({100-porcentaje_entrenamiento*100:.0f}%): {len(df_prueba)}")
        if len(df_prueba) < 2:
            print("Error: No hay suficientes datos para la prueba.")
            return
        print("\nEntrenando la Matriz con los datos historicos...")
        matriz_entrenada = self.generar_matriz_probabilidad(df_entrenamiento)
        aciertos_top_k = 0
        predicciones_totales = 0
        for i in range(len(df_prueba) - 1):
            if df_prueba.iloc[i]['Fecha'] != df_prueba.iloc[i + 1]['Fecha']:
                continue
            animal_actual = df_prueba.iloc[i]['Animal']
            animal_siguiente_real = df_prueba.iloc[i + 1]['Animal']
            if animal_actual in matriz_entrenada.index:
                predicciones_totales += 1
                fila_prediccion = matriz_entrenada.loc[animal_actual].sort_values(ascending=False)
                top_k_predichos = fila_prediccion.head(top_k).index.tolist()
                if animal_siguiente_real in top_k_predichos:
                    aciertos_top_k += 1
        if predicciones_totales > 0:
            precision_top_k = (aciertos_top_k / predicciones_totales) * 100
            num_clases = len(matriz_entrenada.columns)
            probabilidad_azar = (top_k / num_clases) * 100
            print("\nResultados de la Precision (Validacion)")
            print(f"Predicciones realizadas: {predicciones_totales}")
            print(f"Aciertos del Modelo (Top-{top_k}): {aciertos_top_k}")
            print(f"Precision TOP-{top_k} del Modelo: {precision_top_k:.2f}%")
            print(f"\nNota: La precision esperada al azar ({top_k}/{num_clases}) es de {probabilidad_azar:.2f}%")
            if precision_top_k > probabilidad_azar + 5:
                print("Resultado: El modelo es significativamente mejor que el azar para el TOP-5!")
            elif precision_top_k > probabilidad_azar:
                print("Resultado: El modelo tiene un rendimiento ligeramente superior al azar para el TOP-5.")
            else:
                print("Resultado: El rendimiento esta cerca o por debajo del azar. Considera usar mas datos o un modelo diferente.")
        else:
            print("No se pudieron realizar predicciones validas en el conjunto de prueba.")

    def prediccion_por_hora_especifica(self, datos):
        print("\nPrediccion Historica por Hora Especifica")
        while True:
            hora_str = input("Ingresa la hora del sorteo para predecir (ej: 11:00 AM o 14:00): ").strip()
            try:
                hora_dt = pd.to_datetime(hora_str, format='%H:%M', errors='coerce')
                if pd.isna(hora_dt):
                    hora_dt = pd.to_datetime(hora_str, format='%I:%M %p', errors='coerce')
                if pd.isna(hora_dt):
                    raise ValueError
                solo_hora_buscada = hora_dt.strftime('%I:%M %p')
                break
            except ValueError:
                print("Error: Formato de hora invalido. Usa HH:MM (24h) o HH:MM AM/PM.")
        df_filtrado = datos[datos['Solo_hora'] == solo_hora_buscada].copy()
        if df_filtrado.empty:
            print(f"\nNo se encontraron datos historicos para la hora: {solo_hora_buscada}.")
            print("Asegurate de que la hora este escrita exactamente como aparece en tu historial (ej: 09:00 AM).")
            return
        frecuencia_animal = df_filtrado['Animal'].value_counts().reset_index()
        frecuencia_animal.columns = ['Animal', 'Conteo']
        total_sorteos_hora = len(df_filtrado)
        frecuencia_animal['Probabilidad'] = (frecuencia_animal['Conteo'] / total_sorteos_hora) * 100
        prediccion_maxima = frecuencia_animal.iloc[0]
        print(f"\nResultados para la hora: {solo_hora_buscada} (Historico)")
        print(f"Total de sorteos historicos analizados en esta hora: {total_sorteos_hora}")
        print("-" * 50)
        print(f"Animal con mayor probabilidad: {prediccion_maxima['Animal']}")
        print(f"   Probabilidad: {prediccion_maxima['Probabilidad']:.2f}%")
        print(f"   Veces que salio a esta hora: {prediccion_maxima['Conteo']}")
        print("-" * 50)
        print("\nTop 10 de animales en esta hora:")
        print(frecuencia_animal[['Animal', 'Probabilidad']].head(10).to_string(index=False))

    def agregar_datos_al_excel(self, datos_df):
        nombre_archivo = self.config['excel_file']
        print("\nIngreso de Nuevo Sorteo")
        while True:
            hora_str = input("Ingresa la hora del sorteo (ej: 09:00 AM o 14:00): ").strip()
            try:
                hora_dt = pd.to_datetime(hora_str, format='%H:%M', errors='coerce')
                if pd.isna(hora_dt):
                    hora_dt = pd.to_datetime(hora_str, format='%I:%M %p', errors='coerce')
                if pd.isna(hora_dt):
                    raise ValueError
                hora_final = hora_dt.strftime('%H:%M')
                solo_hora_final = hora_dt.strftime('%I:%M %p')
                break
            except ValueError:
                print("Formato de hora invalido. Usa HH:MM (24h) o HH:MM AM/PM.")
        while True:
            try:
                animal = input("Ingresa el nombre del animal (ej: PERRO): ").strip().upper()
                animal = self.validar_animal(animal)
                break
            except ValueError as e:
                print(f"Error: {e}")
        while True:
            numero_str = input("Ingresa el numero (ej: 01): ").strip()
            try:
                if numero_str.isdigit():
                    numero = int(numero_str)
                    numero = self.validar_numero(numero)
                    break
                else:
                    print("Numero invalido. Debe ser un numero entre 00 y 37.")
            except ValueError as e:
                print(f"Error: {e}")
        fecha_hoy = date.today()
        timestamp = pd.to_datetime(f"{fecha_hoy} {hora_final}")
        nueva_fila = pd.DataFrame([{
            'Fecha': fecha_hoy,
            'Hora': hora_final,
            'Animal': animal,
            'Numero': numero,
            'Timestamp': timestamp,
            'Solo_hora': solo_hora_final
        }])
        datos_df = pd.concat([datos_df, nueva_fila], ignore_index=True)
        try:
            datos_df[['Fecha', 'Hora', 'Animal', 'Numero']].to_excel(nombre_archivo, index=False)
            print("\nSorteo agregado y archivo Excel actualizado exitosamente!")
            print(f"   Animal: {animal}, Numero: {numero}, Hora: {solo_hora_final}")
            return datos_df
        except Exception as e:
            print(f"\nError al guardar en el archivo Excel: {e}")
            return datos_df

    def evaluacion_estrategia_solo_manana(self, datos, hora_corte='13:00:00'):
        print(f"\nEVALUACION ESTRATEGIA SOLO MANANA (Hasta {hora_corte})")
        horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        datos_manana = datos[datos['Hora'].isin(horas_manana)].copy()
        frecuencia_manana = datos_manana.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map_manana = {}
        for hora_24h in frecuencia_manana['Hora'].unique():
            top_10_lista = frecuencia_manana[frecuencia_manana['Hora'] == hora_24h].head(15)['Animal'].tolist()
            top_10_map_manana[hora_24h] = top_10_lista
        print(f"Matriz de frecuencia generada para {len(top_10_map_manana)} horas de manana")
        APUESTA_POR_HORA = 500.0
        GANANCIA_POR_ACIERTO = 150.0
        horas_a_jugar = [h for h in horas_manana if h <= hora_corte]
        total_horas = len(horas_a_jugar)
        GASTO_DIARIO = total_horas * APUESTA_POR_HORA
        datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
        resultados_simulacion = []
        for fecha, df_dia in datos.groupby('Fecha'):
            aciertos_totales = 0
            gasto_dia = 0
            ganancia_bruta_dia = 0
            df_manana_jugar = df_dia[df_dia['Hora'].isin(horas_a_jugar)].copy()
            for _, row in df_manana_jugar.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                if hora_filtro in top_10_map_manana and animal_salio in top_10_map_manana[hora_filtro]:
                    aciertos_totales += 1
            gasto_dia = GASTO_DIARIO
            ganancia_bruta_dia = aciertos_totales * GANANCIA_POR_ACIERTO
            resultados_simulacion.append({
                'Fecha': fecha,
                'Horas_Jugadas': total_horas,
                'Aciertos_Manana': aciertos_totales,
                'Gasto': gasto_dia,
                'Ganancia_Bruta': ganancia_bruta_dia,
                'Ganancia_Neta': ganancia_bruta_dia - gasto_dia
            })
        df_resultados = pd.DataFrame(resultados_simulacion)
        dias_completos = df_resultados[df_resultados['Aciertos_Manana'].notna()]
        total_dias = len(dias_completos)
        gasto_total = df_resultados['Gasto'].sum()
        ganancia_bruta_total = df_resultados['Ganancia_Bruta'].sum()
        ganancia_neta_total = df_resultados['Ganancia_Neta'].sum()
        aciertos_promedio = df_resultados['Aciertos_Manana'].mean()
        print("\n" + "="*70)
        print(f"     RESUMEN ESTRATEGIA SOLO MANANA (Hasta {hora_corte})")
        print("="*70)
        print(f"Dias Analizados: {total_dias}")
        print(f"Horas jugadas por dia: {total_horas} ({', '.join(horas_a_jugar)})")
        print(f"Aciertos promedio por dia: {aciertos_promedio:.2f}")
        print("-" * 70)
        print(f"Gasto Total: {gasto_total:,.2f} Bs")
        print(f"Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
        print(f"GANANCIA/PERDIDA NETA TOTAL: {ganancia_neta_total:,.2f} Bs")
        print("-" * 70)
        if gasto_total > 0:
            roi = (ganancia_neta_total / gasto_total) * 100
            print(f"Retorno de la Inversion (ROI): {roi:,.2f}%")
        if ganancia_neta_total > 0:
            print(f"\nLa estrategia SOLO MANANA genero ganancias!")
            print(f"Ganancia promedio por dia: {ganancia_neta_total/total_dias:,.2f} Bs")
        elif ganancia_neta_total < 0:
            print(f"\nLa estrategia genero perdidas.")
            print(f"Perdida promedio por dia: {ganancia_neta_total/total_dias:,.2f} Bs")
        else:
            print("\nResultado: Punto de Equilibrio.")
        print(f"\nDISTRIBUCION DE ACIERTOS POR DIA:")
        distribucion = df_resultados['Aciertos_Manana'].value_counts().sort_index()
        for aciertos, conteo in distribucion.items():
            porcentaje = (conteo / total_dias) * 100
            print(f"   * {aciertos} aciertos: {conteo} dias ({porcentaje:.1f}%)")
        return df_resultados

    def evaluacion_estrategia_filtrada(self, datos, filtro_ganancia=True):
        print("\nEvaluacion Estrategia DINAMICA FILTRADA")
        print("DEBUG - Columnas en datos de entrada:", list(datos.columns))
        print("DEBUG - Primeras filas:")
        print(datos[['Timestamp', 'Hora', 'Animal']].head(3) if 'Timestamp' in datos.columns else "No hay Timestamp")
        try:
            frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
            print(f"Matriz de frecuencia calculada: {len(frecuencia_completa)} registros")
        except Exception as e:
            print(f"Error calculando frecuencia: {e}")
            return None
        top_10_map = {}
        horas_con_datos = frecuencia_completa['Hora'].unique()
        print(f"Horas con datos: {sorted(horas_con_datos)}")
        for hora_24h in horas_con_datos:
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
            top_10_map[hora_24h] = top_10_lista
        print(f"Top-10 map creado: {len(top_10_map)} horas")
        HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
        GASTO_TARDE = 3000.0
        GANANCIA_POR_ACIERTO = 1500.0
        if 'Fecha' not in datos.columns:
            if 'Timestamp' in datos.columns:
                datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
                print("Columna Fecha creada desde Timestamp")
            else:
                print("ERROR: No hay columna Timestamp ni Fecha")
                return None
        fechas_unicas = datos['Fecha'].nunique()
        print(f"Fechas unicas encontradas: {fechas_unicas}")
        resultados_simulacion = []
        dias_procesados = 0
        for fecha, df_dia in datos.groupby('Fecha'):
            dias_procesados += 1
            print(f"Procesando dia {dias_procesados}: {fecha} - {len(df_dia)} registros")
            aciertos_manana = 0
            df_manana = df_dia[df_dia['Hora'].isin(HORAS_MANANA)].copy()
            print(f"   - Registros en manana: {len(df_manana)}")
            for _, row in df_manana.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    aciertos_manana += 1
            print(f"   - Aciertos en manana: {aciertos_manana}")
            jugar_tarde = (aciertos_manana <= 1)
            aciertos_tarde = 0
            ganancia_bruta_tarde = 0
            gasto_tarde = 0
            df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)].copy()
            print(f"   - Registros en tarde: {len(df_tarde)}")
            if jugar_tarde:
                gasto_tarde = GASTO_TARDE
                for _, row in df_tarde.iterrows():
                    hora_filtro = row['Hora']
                    animal_salio = row['Animal']
                    if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                        aciertos_tarde += 1
                ganancia_bruta_tarde = aciertos_tarde * GANANCIA_POR_ACIERTO
                print(f"   - Aciertos en tarde: {aciertos_tarde}")
            ganancia_neta = ganancia_bruta_tarde - gasto_tarde
            resultados_simulacion.append({
                'Fecha': fecha,
                'Aciertos_Manana': aciertos_manana,
                'Jugar_Tarde': 'SI' if jugar_tarde else 'NO',
                'Aciertos_Tarde': aciertos_tarde,
                'Gasto': gasto_tarde,
                'Ganancia_Bruta': ganancia_bruta_tarde,
                'Ganancia_Neta': ganancia_neta
            })
        print(f"Dias procesados: {dias_procesados}")
        print(f"Resultados en simulacion: {len(resultados_simulacion)}")
        if len(resultados_simulacion) == 0:
            print("ERROR: No se crearon resultados. Posibles causas:")
            print("   - No hay datos en las horas especificadas")
            print("   - Problema con las columnas Hora o Fecha")
            return None
        df_resultados = pd.DataFrame(resultados_simulacion)
        print("Columnas creadas en resultados:", list(df_resultados.columns))
        print("Primeras filas de resultados:")
        print(df_resultados.head(3))
        if filtro_ganancia:
            if 'Jugar_Tarde' not in df_resultados.columns or 'Ganancia_Neta' not in df_resultados.columns:
                print("ERROR: Columnas necesarias no se crearon correctamente")
                print("Columnas disponibles:", list(df_resultados.columns))
                return None
            df_filtrado = df_resultados[
                (df_resultados['Jugar_Tarde'] == 'SI') &
                (df_resultados['Ganancia_Neta'] > 0)
            ].copy()
            print(f"FILTRO APLICADO: Solo dias con GANANCIA NETA POSITIVA")
            print(f"   - Dias antes del filtro: {len(df_resultados)}")
            print(f"   - Dias despues del filtro: {len(df_filtrado)}")
        else:
            if 'Jugar_Tarde' not in df_resultados.columns:
                print("ERROR: Columna 'Jugar_Tarde' no encontrada")
                return None
            df_filtrado = df_resultados[df_resultados['Jugar_Tarde'] == 'SI'].copy()
        if len(df_filtrado) == 0:
            print("No hay dias que cumplan los criterios de filtrado")
            return None
        total_dias_jugados = len(df_filtrado)
        gasto_total = df_filtrado['Gasto'].sum()
        ganancia_bruta_total = df_filtrado['Ganancia_Bruta'].sum()
        ganancia_neta_total = df_filtrado['Ganancia_Neta'].sum()
        print("\n" + "="*70)
        print("        RESUMEN ESTRATEGIA FILTRADA")
        print("="*70)
        print(f"Dias Completos Analizados: {len(df_resultados)}")
        print(f"Dias Jugados (con regla 0 o 1 acierto manana): {len(df_resultados[df_resultados['Jugar_Tarde'] == 'SI'])}")
        print(f"Dias FILTRADOS con Ganancia: {total_dias_jugados}")
        print("-" * 70)
        print(f"Gasto Total: {gasto_total:,.2f} Bs")
        print(f"Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
        print(f"GANANCIA NETA TOTAL: {ganancia_neta_total:,.2f} Bs")
        print("-" * 70)
        if gasto_total > 0:
            roi = (ganancia_neta_total / gasto_total) * 100
            print(f"Retorno de la Inversion (ROI): {roi:,.2f}%")
        if total_dias_jugados > 0:
            print(f"\nDISTRIBUCION DE RESULTADOS:")
            distribucion = df_filtrado['Ganancia_Neta'].value_counts().sort_index(ascending=False)
            for ganancia, conteo in distribucion.items():
                porcentaje = (conteo / total_dias_jugados) * 100
                print(f"   * {ganancia:+,.0f} Bs: {conteo} dias ({porcentaje:.1f}%)")
        dias_con_perdida = len(df_resultados[
            (df_resultados['Jugar_Tarde'] == 'SI') &
            (df_resultados['Ganancia_Neta'] < 0)
        ])
        print(f"\nDias eliminados con perdida: {dias_con_perdida}")
        if len(df_filtrado) > 0:
            print(f"\nTOP 5 MEJORES DIAS (con filtro):")
            top_5 = df_filtrado.nlargest(5, 'Ganancia_Neta')[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']]
            print(top_5.to_string(index=False))
        return df_filtrado

    def analisis_estadistico_avanzado(self, datos):
        print("\nANALISIS ESTADISTICO AVANZADO")
        print("DIAGNOSTICO COMPLETO:")
        print(f"   * Total de registros en datos: {len(datos)}")
        print(f"   * Columnas disponibles: {list(datos.columns)}")
        if len(datos) == 0:
            print("ERROR: El DataFrame esta completamente vacio")
            return None
        columnas_criticas = ['Timestamp', 'Hora', 'Animal']
        for col in columnas_criticas:
            if col not in datos.columns:
                print(f"ERROR: Columna critica '{col}' no encontrada")
                return None
        print(f"Valores en columnas criticas:")
        print(f"   * Timestamp: {datos['Timestamp'].notna().sum()} valores no nulos")
        print(f"   * Hora: {datos['Hora'].notna().sum()} valores no nulos")
        print(f"   * Animal: {datos['Animal'].notna().sum()} valores no nulos")
        print(f"Primeras 3 filas REALES:")
        print(datos[['Timestamp', 'Hora', 'Animal']].head(3))
        print(f"Valores unicos en Hora: {sorted(datos['Hora'].unique())[:10]}")
        try:
            frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
            print(f"Matriz de frecuencia: {len(frecuencia_completa)} registros")
            if len(frecuencia_completa) == 0:
                print("La matriz de frecuencia esta vacia - revisar formato de Hora y Animal")
                return None
            top_10_map = {}
            horas_con_datos = frecuencia_completa['Hora'].unique()
            print(f"Horas con datos en frecuencia: {sorted(horas_con_datos)}")
            for hora_24h in horas_con_datos:
                top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
                top_10_map[hora_24h] = top_10_lista
            HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
            HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
            GASTO_TARDE = 1200.0
            GANANCIA_POR_ACIERTO = 600.0
            datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
            datos['Dia_Semana'] = datos['Timestamp'].dt.day_name()
            datos['Mes'] = datos['Timestamp'].dt.month
            datos['Dia_Mes'] = datos['Timestamp'].dt.day
            print(f"Fechas unicas despues de procesar: {datos['Fecha'].nunique()}")
            resultados = []
            dias_procesados = 0
            for fecha, df_dia in datos.groupby('Fecha'):
                dias_procesados += 1
                aciertos_manana = 0
                df_manana = df_dia[df_dia['Hora'].isin(HORAS_MANANA)]
                for _, row in df_manana.iterrows():
                    hora_filtro = row['Hora']
                    animal_salio = row['Animal']
                    if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                        aciertos_manana += 1
                jugar_tarde = (aciertos_manana <= 1)
                aciertos_tarde = 0
                if jugar_tarde:
                    df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)]
                    for _, row in df_tarde.iterrows():
                        hora_filtro = row['Hora']
                        animal_salio = row['Animal']
                        if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                            aciertos_tarde += 1
                ganancia_neta = (aciertos_tarde * GANANCIA_POR_ACIERTO) - (GASTO_TARDE if jugar_tarde else 0)
                resultados.append({
                    'Fecha': fecha,
                    'Dia_Semana': df_dia['Dia_Semana'].iloc[0],
                    'Mes': df_dia['Mes'].iloc[0],
                    'Dia_Mes': df_dia['Dia_Mes'].iloc[0],
                    'Aciertos_Manana': aciertos_manana,
                    'Jugar_Tarde': jugar_tarde,
                    'Aciertos_Tarde': aciertos_tarde,
                    'Ganancia_Neta': ganancia_neta
                })
            print(f"Dias procesados: {dias_procesados}")
            print(f"Resultados creados: {len(resultados)}")
            if len(resultados) == 0:
                print("No se crearon resultados - posiblemente no hay datos")
                return None
            df_analisis = pd.DataFrame(resultados)
            print("Analisis completado - Columnas creadas:", list(df_analisis.columns))
            df_jugados = df_analisis[df_analisis['Jugar_Tarde'] == True]
            print(f"\nRESUMEN GENERAL:")
            print(f"   * Total dias analizados: {len(df_analisis)}")
            print(f"   * Dias que cumplen regla (0-1 aciertos manana): {len(df_jugados)}")
            if len(df_jugados) > 0:
                print("ANALISIS POR DIA DE LA SEMANA:")
                analisis_dia_semana = df_jugados.groupby('Dia_Semana').agg({
                    'Ganancia_Neta': ['count', 'sum', 'mean'],
                    'Aciertos_Tarde': 'mean'
                }).round(2)
                print(analisis_dia_semana)
            else:
                print("No hay dias que cumplan la condicion para jugar en la tarde")
            return df_analisis
        except Exception as e:
            print(f"Error en analisis estadistico: {e}")
            import traceback
            traceback.print_exc()
            return None

    def patrones_dias_rentables(self, datos):
        print("\nANALISIS DE PATRONES PARA DIAS RENTABLES")
        frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map = {}
        for hora_24h in frecuencia_completa['Hora'].unique():
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
            top_10_map[hora_24h] = top_10_lista
        HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
        GASTO_TARDE = 3000.0
        GANANCIA_POR_ACIERTO = 1500.0
        datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
        datos['Dia_Semana'] = datos['Timestamp'].dt.day_name()
        datos['Mes'] = datos['Timestamp'].dt.month
        datos['Dia_Mes'] = datos['Timestamp'].dt.day
        resultados_detallados = []
        for fecha, df_dia in datos.groupby('Fecha'):
            aciertos_por_hora_manana = {}
            animales_por_hora_manana = {}
            for hora in HORAS_MANANA:
                df_hora = df_dia[df_dia['Hora'] == hora]
                if not df_hora.empty:
                    animal_salio = df_hora['Animal'].iloc[0]
                    acierto = 1 if (hora in top_10_map and animal_salio in top_10_map[hora]) else 0
                    aciertos_por_hora_manana[hora] = acierto
                    animales_por_hora_manana[hora] = animal_salio
            aciertos_manana = sum(aciertos_por_hora_manana.values())
            jugar_tarde = (aciertos_manana <= 1)
            aciertos_tarde = 0
            if jugar_tarde:
                for _, row in df_dia[df_dia['Hora'].isin(HORAS_TARDE)].iterrows():
                    if row['Hora'] in top_10_map and row['Animal'] in top_10_map[row['Hora']]:
                        aciertos_tarde += 1
            ganancia_neta = (aciertos_tarde * GANANCIA_POR_ACIERTO) - (GASTO_TARDE if jugar_tarde else 0)
            resultados_detallados.append({
                'Fecha': fecha,
                'Dia_Semana': df_dia['Dia_Semana'].iloc[0],
                'Mes': df_dia['Mes'].iloc[0],
                'Dia_Mes': df_dia['Dia_Mes'].iloc[0],
                'Aciertos_Manana': aciertos_manana,
                'Aciertos_Tarde': aciertos_tarde,
                'Ganancia_Neta': ganancia_neta,
                'Jugar_Tarde': jugar_tarde,
                'Acierto_8am': aciertos_por_hora_manana.get('08:00:00', 0),
                'Acierto_9am': aciertos_por_hora_manana.get('09:00:00', 0),
                'Acierto_10am': aciertos_por_hora_manana.get('10:00:00', 0),
                'Acierto_11am': aciertos_por_hora_manana.get('11:00:00', 0),
                'Acierto_12pm': aciertos_por_hora_manana.get('12:00:00', 0),
                'Acierto_1pm': aciertos_por_hora_manana.get('13:00:00', 0),
                'Animal_8am': animales_por_hora_manana.get('08:00:00', 'N/A'),
                'Animal_9am': animales_por_hora_manana.get('09:00:00', 'N/A'),
                'Animal_10am': animales_por_hora_manana.get('10:00:00', 'N/A'),
                'Animal_11am': animales_por_hora_manana.get('11:00:00', 'N/A'),
                'Animal_12pm': animales_por_hora_manana.get('12:00:00', 'N/A'),
                'Animal_1pm': animales_por_hora_manana.get('13:00:00', 'N/A')
            })
        df_analisis = pd.DataFrame(resultados_detallados)
        df_jugados = df_analisis[df_analisis['Jugar_Tarde'] == True]
        print("PATRONES ENCONTRADOS:")
        print("="*60)
        print("\n1. RENTABILIDAD POR DIA DE LA SEMANA:")
        analisis_dias = df_jugados.groupby('Dia_Semana').agg({
            'Ganancia_Neta': ['count', 'sum', 'mean'],
            'Aciertos_Tarde': 'mean'
        }).round(2)
        print(analisis_dias)
        print("\n2. PATRONES POR COMPORTAMIENTO EN MANANA:")
        print("Combinaciones que PREDICEN EXITO en la tarde:")
        combinaciones_exitosas = df_jugados[df_jugados['Ganancia_Neta'] > 0]
        if len(combinaciones_exitosas) > 0:
            patron_exito = combinaciones_exitosas[['Acierto_8am', 'Acierto_9am', 'Acierto_10am',
                                                  'Acierto_11am', 'Acierto_12pm', 'Acierto_1pm']].mean()
            print("Aciertos promedio por hora en dias GANADORES:")
            for hora, prob in patron_exito.items():
                print(f"   {hora}: {prob:.2%}")
        print("\nCombinaciones que PREDICEN FRACASO en la tarde:")
        combinaciones_fracaso = df_jugados[df_jugados['Ganancia_Neta'] < 0]
        if len(combinaciones_fracaso) > 0:
            patron_fracaso = combinaciones_fracaso[['Acierto_8am', 'Acierto_9am', 'Acierto_10am',
                                                   'Acierto_11am', 'Acierto_12pm', 'Acierto_1pm']].mean()
            print("Aciertos promedio por hora en dias PERDEDORES:")
            for hora, prob in patron_fracaso.items():
                print(f"   {hora}: {prob:.2%}")
        print("\n3. REGLAS DE DECISION (CUANDO APOSTAR):")
        dias_recomendados = df_jugados.groupby('Dia_Semana')['Ganancia_Neta'].mean()
        dias_recomendados = dias_recomendados[dias_recomendados > 0].index.tolist()
        if dias_recomendados:
            print(f"   PRIORIZAR estos dias: {', '.join(dias_recomendados)}")
        print("\n4. SISTEMA DE ALERTA EN TIEMPO REAL:")
        print("   (Basado en tus datos historicos)")
        print(f"   * Dias totales analizados: {len(df_jugados)}")
        print(f"   * Dias con ganancia: {len(df_jugados[df_jugados['Ganancia_Neta'] > 0])}")
        print(f"   * Dias con perdida: {len(df_jugados[df_jugados['Ganancia_Neta'] < 0])}")
        prob_ganar = len(df_jugados[df_jugados['Ganancia_Neta'] > 0]) / len(df_jugados) if len(df_jugados) > 0 else 0
        print(f"   * Probabilidad de ganar: {prob_ganar:.1%}")
        return df_analisis

    def predictor_dia_actual(self, datos):
        print("\nPREDICCION PARA HOY")
        df_analisis = self.patrones_dias_rentables(datos)
        from datetime import datetime
        hoy = datetime.now().date()
        dia_semana_hoy = hoy.strftime("%A")
        print(f"\nHOY ES: {hoy} ({dia_semana_hoy})")
        df_mismo_dia = df_analisis[df_analisis['Dia_Semana'] == dia_semana_hoy]
        df_mismo_dia_jugado = df_mismo_dia[df_mismo_dia['Jugar_Tarde'] == True]
        if len(df_mismo_dia_jugado) == 0:
            print("No hay suficientes datos historicos para este dia")
            return
        total_dias = len(df_mismo_dia_jugado)
        dias_ganadores = len(df_mismo_dia_jugado[df_mismo_dia_jugado['Ganancia_Neta'] > 0])
        dias_perdedores = len(df_mismo_dia_jugado[df_mismo_dia_jugado['Ganancia_Neta'] < 0])
        probabilidad_ganar = dias_ganadores / total_dias if total_dias > 0 else 0
        ganancia_promedio = df_mismo_dia_jugado['Ganancia_Neta'].mean()
        print(f"ESTADISTICAS PARA LOS {dia_semana_hoy}s:")
        print(f"   * Dias jugados historicamente: {total_dias}")
        print(f"   * Dias con ganancia: {dias_ganadores} ({probabilidad_ganar:.1%})")
        print(f"   * Dias con perdida: {dias_perdedores}")
        print(f"   * Ganancia promedio: {ganancia_promedio:,.0f} Bs")
        print(f"\nRECOMENDACION PARA HOY:")
        if probabilidad_ganar >= 0.6 and ganancia_promedio > 500:
            print("   CONFIANZA ALTA! Es un buen dia para apostar en la tarde")
            print(f"   Probabilidad historica de ganar: {probabilidad_ganar:.1%}")
            print(f"   Ganancia promedio esperada: {ganancia_promedio:,.0f} Bs")
        elif probabilidad_ganar >= 0.4:
            print("   CONFIANZA MEDIA - Considera apostar moderadamente")
            print(f"   Probabilidad historica: {probabilidad_ganar:.1%}")
        else:
            print("   CONFIANZA BAJA - Mejor no apostar hoy")
            print(f"   Probabilidad historica: {probabilidad_ganar:.1%}")
        print(f"\nPATRONES ESPECIFICOS A OBSERVAR HOY:")
        if len(df_mismo_dia_jugado) > 0:
            dias_exitosos = df_mismo_dia_jugado[df_mismo_dia_jugado['Ganancia_Neta'] > 0]
            if len(dias_exitosos) > 0:
                print("   En dias EXITOSOS de este dia, los patrones fueron:")
                for col in ['Acierto_8am', 'Acierto_9am', 'Acierto_10am', 'Acierto_11am', 'Acierto_12pm', 'Acierto_1pm']:
                    if col in dias_exitosos.columns:
                        prob = dias_exitosos[col].mean()
                        print(f"     * {col}: {prob:.0%} de aciertos")
        return probabilidad_ganar, ganancia_promedio

    def ver_ultimos_registros_y_faltantes(self, datos):
        print("\nULTIMOS REGISTROS Y ANALISIS DEL DIA")
        if len(datos) == 0:
            print("No hay registros en la base de datos")
            return
        datos_ordenados = datos.sort_values('Timestamp', ascending=False)
        print("\nULTIMOS 10 REGISTROS (mas recientes primero):")
        print("="*60)
        columnas_mostrar = ['Fecha', 'Hora', 'Animal', 'Numero']
        ultimos_10 = datos_ordenados[columnas_mostrar].head(10)
        print(ultimos_10.to_string(index=False))
        fecha_mas_reciente = datos_ordenados['Fecha'].iloc[0]
        print(f"\nFECHA MAS RECIENTE EN DATOS: {fecha_mas_reciente}")
        from datetime import date
        hoy = date.today()
        datos_hoy = datos[datos['Fecha'] == hoy]
        if len(datos_hoy) > 0:
            print(f"\nREGISTROS DE HOY ({hoy}):")
            print("="*40)
            registros_hoy = datos_hoy[['Hora', 'Animal', 'Numero']].sort_values('Hora')
            print(registros_hoy.to_string(index=False))
            horas_posibles = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00',
                             '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00',
                             '18:00:00', '19:00:00']
            horas_registradas_hoy = datos_hoy['Hora'].unique()
            horas_faltantes = [h for h in horas_posibles if h not in horas_registradas_hoy]
            if horas_faltantes:
                print(f"\nHORAS FALTANTES POR AGREGAR HOY:")
                for hora in horas_faltantes:
                    print(f"   * {hora}")
            else:
                print(f"\nTODAS LAS HORAS DE HOY ESTAN COMPLETAS!")
        else:
            print(f"\nHOY NO HAY REGISTROS ({hoy})")
            print("   Usa la Opcion 1 para agregar el primer sorteo del dia")
        print(f"\nANALISIS DE ANIMALES RECIENTES:")
        print("="*40)
        ultimos_20 = datos_ordenados.head(20)
        animales_recientes = ultimos_20['Animal'].unique()
        print(f"Animales en ultimos 20 sorteos ({len(animales_recientes)} unicos):")
        for i, animal in enumerate(animales_recientes, 1):
            print(f"   {i:2d}. {animal}")
        ultimos_50 = datos_ordenados.head(50)
        frecuencia_reciente = ultimos_50['Animal'].value_counts().head(10)
        print(f"\nTOP 10 ANIMALES MAS FRECUENTES (ultimos 50 sorteos):")
        for animal, conteo in frecuencia_reciente.items():
            porcentaje = (conteo / len(ultimos_50)) * 100
            print(f"   * {animal}: {conteo} veces ({porcentaje:.1f}%)")
        return datos_ordenados.head(10)

    def ver_estado_actual_dia(self, datos):
        from datetime import date
        hoy = date.today()
        print(f"\nESTADO ACTUAL - {hoy}")
        datos_hoy = datos[datos['Fecha'] == hoy]
        if len(datos_hoy) == 0:
            print("No hay registros para hoy")
            print("   Usa la Opcion 1 para agregar sorteos")
            return
        registros_hoy = datos_hoy[['Hora', 'Animal', 'Numero']].sort_values('Hora')
        print(f"\nSORTEOS REGISTRADOS HOY ({len(registros_hoy)}):")
        print(registros_hoy.to_string(index=False))
        horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        horas_tarde = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
        horas_registradas = datos_hoy['Hora'].tolist()
        manana_registradas = [h for h in horas_registradas if h in horas_manana]
        tarde_registradas = [h for h in horas_registradas if h in horas_tarde]
        print(f"\nRESUMEN HORARIO:")
        print(f"   * Manana (8am-1pm): {len(manana_registradas)}/{len(horas_manana)} horas")
        print(f"   * Tarde (2pm-7pm): {len(tarde_registradas)}/{len(horas_tarde)} horas")
        todas_horas = horas_manana + horas_tarde
        horas_faltantes = [h for h in todas_horas if h not in horas_registradas]
        if horas_faltantes:
            print(f"\nHORAS FALTANTES:")
            for hora in horas_faltantes:
                print(f"   * {hora}")
        else:
            print(f"\nDIA COMPLETO! Todas las horas registradas")

    def analizar_rachas_tempranas(self, datos, horas_evaluacion=3, umbral_aciertos=3):
        print(f"\nANALISIS DE RACHAS TEMPRANAS")
        print(f"Buscando: {umbral_aciertos}+ aciertos en primeras {horas_evaluacion} horas")
        if horas_evaluacion == 2:
            horas_a_evaluar = ['08:00:00', '09:00:00']
        elif horas_evaluacion == 3:
            horas_a_evaluar = ['08:00:00', '09:00:00', '10:00:00']
        elif horas_evaluacion == 4:
            horas_a_evaluar = ['08:00:00', '09:00:00', '10:00:00', '11:00:00']
        frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map = {}
        for hora_24h in frecuencia_completa['Hora'].unique():
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(15)['Animal'].tolist()
            top_10_map[hora_24h] = top_10_lista
        resultados = []
        for fecha, df_dia in datos.groupby('Fecha'):
            aciertos_tempranos = 0
            df_temprano = df_dia[df_dia['Hora'].isin(horas_a_evaluar)]
            for _, row in df_temprano.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    aciertos_tempranos += 1
            aciertos_manana = 0
            aciertos_tarde = 0
            horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
            horas_tarde = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
            for _, row in df_dia.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    if hora_filtro in horas_manana:
                        aciertos_manana += 1
                    elif hora_filtro in horas_tarde:
                        aciertos_tarde += 1
            resultados.append({
                'Fecha': fecha,
                'Aciertos_Tempranos': aciertos_tempranos,
                'Aciertos_Manana_Total': aciertos_manana,
                'Aciertos_Tarde_Total': aciertos_tarde,
                'Aciertos_Dia_Completo': aciertos_manana + aciertos_tarde,
                'Tiene_Racha_Temprana': aciertos_tempranos >= umbral_aciertos
            })
        df_analisis = pd.DataFrame(resultados)
        print(f"\nRESULTADOS PARA {umbral_aciertos}+ ACIERTOS EN {horas_evaluacion} HORAS:")
        dias_con_racha = df_analisis[df_analisis['Tiene_Racha_Temprana'] == True]
        dias_sin_racha = df_analisis[df_analisis['Tiene_Racha_Temprana'] == False]
        if len(dias_con_racha) > 0:
            print(f"DIAS CON RACHA TEMPRANA ({len(dias_con_racha)} dias):")
            print(f"   * Aciertos manana promedio: {dias_con_racha['Aciertos_Manana_Total'].mean():.2f}")
            print(f"   * Aciertos tarde promedio: {dias_con_racha['Aciertos_Tarde_Total'].mean():.2f}")
            print(f"   * Aciertos dia completo: {dias_con_racha['Aciertos_Dia_Completo'].mean():.2f}")
            ganancia_promedio_racha = (dias_con_racha['Aciertos_Dia_Completo'].mean() * 580) - (12 * 20)
            print(f"   * Ganancia neta estimada: {ganancia_promedio_racha:+.0f} Bs")
        if len(dias_sin_racha) > 0:
            print(f"\nDIAS SIN RACHA TEMPRANA ({len(dias_sin_racha)} dias):")
            print(f"   * Aciertos manana promedio: {dias_sin_racha['Aciertos_Manana_Total'].mean():.2f}")
            print(f"   * Aciertos tarde promedio: {dias_sin_racha['Aciertos_Tarde_Total'].mean():.2f}")
            print(f"   * Aciertos dia completo: {dias_sin_racha['Aciertos_Dia_Completo'].mean():.2f}")
            ganancia_promedio_sin_racha = (dias_sin_racha['Aciertos_Dia_Completo'].mean() * 580) - (12 * 20)
            print(f"   * Ganancia neta estimada: {ganancia_promedio_sin_racha:+.0f} Bs")
        return df_analisis

    def probar_umbrales_rachas(self, datos):
        print("\nOPTIMIZACION DE UMBRALES DE RACHA")
        combinaciones = [
            (2, 2), (2, 3), (2, 4),
            (3, 2), (3, 3), (3, 4),
            (4, 2), (4, 3), (4, 4)
        ]
        resultados_umbrales = []
        for horas, umbral in combinaciones:
            df_temp = self.analizar_rachas_tempranas(datos, horas_evaluacion=horas, umbral_aciertos=umbral)
            dias_con_racha = df_temp[df_temp['Tiene_Racha_Temprana'] == True]
            if len(dias_con_racha) > 0:
                efectividad = dias_con_racha['Aciertos_Dia_Completo'].mean()
                resultados_umbrales.append({
                    'Horas_Evaluacion': horas,
                    'Umbral_Aciertos': umbral,
                    'Dias_Con_Racha': len(dias_con_racha),
                    'Aciertos_Promedio_Dia': efectividad,
                    'Ganancia_Estimada': (efectividad * 580) - (12 * 20)
                })
        df_umbrales = pd.DataFrame(resultados_umbrales)
        print(f"\nMEJORES COMBINACIONES:")
        mejores = df_umbrales.nlargest(3, 'Ganancia_Estimada')
        print(mejores[['Horas_Evaluacion', 'Umbral_Aciertos', 'Dias_Con_Racha', 'Ganancia_Estimada']].to_string(index=False))
        return df_umbrales

    def main_menu(self, datos):
        opciones = [
            "Ingresar Sorteo del Dia (Actualizar Excel)",
            "Validacion Cruzada de Precision del Modelo (Markov)",
            "Mostrar Matriz de Probabilidad de Transicion",
            "Prediccion Siguiente (Basado en Ultimo Animal - Markov)",
            "Probabilidad Maxima Historica por Hora (Tabla Completa - TOP-10)",
            "Prediccion Historica por Hora Especifica (TOP-10)",
            "ENTRENAR: Random Forest Optimizado (Auto-tuning)",
            "ENTRENAR: XGBoost Optimizado (Auto-tuning)",
            "CARGAR Modelo Pre-entrenado",
            "Evaluar Estrategia Dinamica (Frecuencia Historica)",
            "Evaluar Estrategia Dinamica (Prediccion de IA/ML)",
            "Evaluar Estrategia Solo Manana (Hasta Hora Especifica)",
            "Evaluar Estrategia Dinamica Filtrada (Ganancia Neta Positiva)",
            "Analisis Estadistico Avanzado (Dias Ganadores vs Perdedores)",
            "Analisis de Patrones para Dias Rentables vs No Rentables",
            "Prediccion para Hoy (Es un dia seguro para apostar?)",
            "Ver Ultimos Registros y Analisis del Dia",
            "Ver Estado Rapido del Dia Actual",
            "Analisis de Rachas Tempranas y su Impacto en el Dia Completo",
            "Web Scraper - Actualizar Datos desde loteriadehoy.com",
            "Salir del Programa"
        ]
        matriz_ia_entrenada = None
        modelo_cargado = None
        le_y_cargado = None

        while True:
            opcion_elegida = mostrar_menu("Programa de Prediccion - Menu Principal", opciones)

            if opcion_elegida == 1:
                datos = self.agregar_datos_al_excel(datos)
                datos = self.agregar_caracteristicas_avanzadas(datos.copy())
                self.logger.info("Nuevo sorteo ingresado y caracteristicas actualizadas")
            elif opcion_elegida == 2:
                self.validar_modelo_markov(datos.copy())
            elif opcion_elegida == 3:
                self.matriz_probabilidad_transicion(datos.copy())
            elif opcion_elegida == 4:
                self.mejor_prediccion_siguiente(datos.copy())
            elif opcion_elegida == 5:
                self.probabilidad_maxima_por_hora(datos.copy())
            elif opcion_elegida == 6:
                self.prediccion_por_hora_especifica(datos.copy())
            elif opcion_elegida == 7:
                matriz_ia_entrenada = self.random_forest_optimizado(datos.copy())
                if matriz_ia_entrenada:
                    print("\nMatriz de prediccion Random Forest Optimizado generada.")
            elif opcion_elegida == 8:
                matriz_ia_entrenada = self.xgboost_optimizado(datos.copy())
                if matriz_ia_entrenada:
                    print("\nMatriz de prediccion XGBoost Optimizado generada.")
            elif opcion_elegida == 9:
                print("\nCARGAR MODELO PRE-ENTRENADO")
                print("1. Cargar Random Forest")
                print("2. Cargar XGBoost")
                sub_opcion = input("Selecciona tipo de modelo: ").strip()
                if sub_opcion == '1':
                    modelo_cargado, le_y_cargado, metricas = self.cargar_ultimo_modelo("random_forest")
                    if modelo_cargado:
                        print("Random Forest cargado - Listo para predicciones")
                elif sub_opcion == '2':
                    modelo_cargado, le_y_cargado, metricas = self.cargar_ultimo_modelo("xgboost")
                    if modelo_cargado:
                        print("XGBoost cargado - Listo para predicciones")
            elif opcion_elegida == 10:
                self.evaluacion_estrategia_frecuencia(datos.copy())
            elif opcion_elegida == 11:
                if matriz_ia_entrenada is None and modelo_cargado is None:
                    print("\nADVERTENCIA: Primero debes ejecutar la Opcion 7, 8 o 9.")
                elif modelo_cargado:
                    datos_con_features = self.agregar_caracteristicas_avanzadas(datos.copy())
                    matriz_prediccion = self.predecir_top_k_por_hora(
                        modelo_cargado, le_y_cargado, datos_con_features.copy(), k=20
                    )
                    self.mostrar_matriz_prediccion(matriz_prediccion)
                    self.evaluacion_estrategia_ia(datos.copy(), matriz_prediccion)
                else:
                    self.mostrar_matriz_prediccion(matriz_ia_entrenada)
                    self.evaluacion_estrategia_ia(datos.copy(), matriz_ia_entrenada)
            elif opcion_elegida == 12:
                print("\nESTRATEGIA SOLO MANANA")
                print("1. Jugar hasta las 12:00 (4 horas)")
                print("2. Jugar hasta las 13:00 (5 horas)")
                sub_opcion = input("Selecciona horario: ").strip()
                if sub_opcion == '1':
                    self.evaluacion_estrategia_solo_manana(datos.copy(), '12:00:00')
                elif sub_opcion == '2':
                    self.evaluacion_estrategia_solo_manana(datos.copy(), '13:00:00')
            elif opcion_elegida == 13:
                self.evaluacion_estrategia_filtrada(datos.copy(), filtro_ganancia=True)
            elif opcion_elegida == 14:
                self.analisis_estadistico_avanzado(datos.copy())
            elif opcion_elegida == 15:
                self.patrones_dias_rentables(datos.copy())
            elif opcion_elegida == 16:
                self.predictor_dia_actual(datos.copy())
            elif opcion_elegida == 17:
                self.ver_ultimos_registros_y_faltantes(datos.copy())
            elif opcion_elegida == 18:
                self.ver_estado_actual_dia(datos.copy())
            elif opcion_elegida == 19:
                print("\nOPCIONES DE ANALISIS DE RACHAS:")
                print("1. Analisis con 3+ aciertos en primeras 3 horas")
                print("2. Probar diferentes umbrales")
                sub_opcion = input("Selecciona: ").strip()
                if sub_opcion == '1':
                    self.analizar_rachas_tempranas(datos.copy(), horas_evaluacion=3, umbral_aciertos=3)
                elif sub_opcion == '2':
                    self.probar_umbrales_rachas(datos.copy())
            elif opcion_elegida == 20:
                scraper_map = {
                    "Lotto Activo": "scraper_lotto",
                    "La Granjita": "scraper_la_granjita",
                    "Selva Plus": "scraper_selva_plus",
                }
                scraper_mod = scraper_map.get(self.config['nombre'], "scraper_lotto")
                print(f"\nWEB SCRAPER - {self.config['nombre']}")
                print("1. Scrapear dia especifico")
                print("2. Scrapear rango de fechas")
                print("3. Buscar fechas faltantes y scrapear")
                sub_opcion = input("Selecciona: ").strip()
                if sub_opcion == '1':
                    fecha = input("Fecha (YYYY-MM-DD): ").strip()
                    mod = __import__(scraper_mod)
                    records = mod.scrape_date(fecha)
                    if records:
                        df = pd.DataFrame(records)
                        combined = mod.save_to_excel(df, self.config['excel_file'])
                        datos = combined
                        datos = self.agregar_caracteristicas_avanzadas(datos.copy())
                        print(f"{len(records)} registros agregados de {fecha}")
                    else:
                        print("No se encontraron registros para esa fecha")
                elif sub_opcion == '2':
                    inicio = input("Fecha inicio (YYYY-MM-DD): ").strip()
                    fin = input("Fecha fin (YYYY-MM-DD): ").strip()
                    mod = __import__(scraper_mod)
                    df = mod.scrape_range(inicio, fin)
                    if not df.empty:
                        combined = mod.save_to_excel(df, self.config['excel_file'])
                        datos = combined
                        datos = self.agregar_caracteristicas_avanzadas(datos.copy())
                        print(f"{len(df)} registros agregados")
                    else:
                        print("No se encontraron registros")
                elif sub_opcion == '3':
                    import datetime
                    inicio = input("Fecha inicio (YYYY-MM-DD): ").strip()
                    fin = input("Fecha fin (YYYY-MM-DD): ").strip()
                    import pandas as pd
                    mod = __import__(scraper_mod)
                    existing = pd.read_excel(self.config['excel_file'])
                    existing['Fecha'] = pd.to_datetime(existing['Fecha']).dt.strftime("%Y-%m-%d")
                    existing_dates = set(existing['Fecha'].unique())
                    all_dates = set()
                    current = datetime.datetime.strptime(inicio, "%Y-%m-%d")
                    end_dt = datetime.datetime.strptime(fin, "%Y-%m-%d")
                    while current <= end_dt:
                        all_dates.add(current.strftime("%Y-%m-%d"))
                        current += datetime.timedelta(days=1)
                    missing = sorted(all_dates - existing_dates)
                    print(f"Fechas faltantes: {len(missing)} de {len(all_dates)}")
                    if not missing:
                        print("No hay fechas faltantes!")
                    else:
                        all_records = []
                        for i, date_str in enumerate(missing):
                            records = mod.scrape_date(date_str)
                            all_records.extend(records)
                            if (i + 1) % 10 == 0:
                                print(f"Progreso: {i+1}/{len(missing)} dias")
                            import time
                            time.sleep(1.5)
                        if all_records:
                            df = pd.DataFrame(all_records)
                            combined = mod.save_to_excel(df, self.config['excel_file'])
                            datos = combined
                            datos = self.agregar_caracteristicas_avanzadas(datos.copy())
                            print(f"{len(df)} nuevos registros agregados")
                        else:
                            print("No se encontraron nuevos registros")
            elif opcion_elegida == 21:
                print("\nGracias por usar el programa de prediccion! Saliendo...")
                break

            if opcion_elegida != 1:
                input("\nPresiona Enter para volver al menu...")
