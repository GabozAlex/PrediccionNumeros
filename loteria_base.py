import pandas as pd
import os
import numpy as np
import pickle
import joblib
import json
from datetime import datetime, date, timedelta

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from utils import setup_logging, mostrar_menu, ANIMALES_38, GRUPOS_ANIMALES, HORA_MAP_12_TO_24, ANIMAL_A_NUM_INT, NUM_INT_A_ANIMAL


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
        self.animal_a_num_int = ANIMAL_A_NUM_INT
        self.num_int_a_animal = NUM_INT_A_ANIMAL
        self._cache_markov = {}

    def _transiciones_markov(self, df):
        from collections import defaultdict
        trans_count = defaultdict(lambda: defaultdict(int))
        trans_total = defaultdict(int)
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] == df.iloc[i]['Fecha']:
                prev = df.iloc[i-1]['Num_Int']
                cur = df.iloc[i]['Num_Int']
                trans_count[prev][cur] += 1
                trans_total[prev] += 1
        trans_prob = {}
        for prev, followers in trans_count.items():
            for cur, cnt in followers.items():
                trans_prob[(prev, cur)] = cnt / trans_total[prev] * 100
        return trans_prob, trans_total

    def _frecuencias_hora(self, df, col_hora='Hora'):
        freq = df.groupby(col_hora)['Num_Int'].value_counts(normalize=True).mul(100)
        result = {}
        for (hora, num), prob in freq.items():
            result.setdefault(hora, {})[num] = prob
        return result

    def _mh_ranking(self, markov_scores, hourly_scores, numeros_validos):
        max_m = max(markov_scores.values()) if markov_scores else 1
        combined = {}
        for n in numeros_validos:
            mp = markov_scores.get(n, 0) / max_m * 100
            hp = hourly_scores.get(n, 0)
            combined[n] = mp + hp
        top20 = sorted(combined, key=combined.get, reverse=True)[:25]
        rankings = {}
        for i, n in enumerate(top20, 1):
            rankings[n] = (i, combined[n])
        return rankings, combined

    def _resolver_animal(self, entrada):
        if isinstance(entrada, str):
            entrada = entrada.strip()
        # Try numeric
        try:
            n = int(entrada)
            if 0 <= n <= self.config['max_numero']:
                animal = self.num_int_a_animal.get(n)
                if animal:
                    return animal
        except (ValueError, TypeError):
            pass
        # Try animal name
        animal = str(entrada).upper().strip()
        if animal in self.animales_carac:
            return animal
        raise ValueError(
            f"'{entrada}' no es numero valido (0-{self.config['max_numero']}) ni nombre de animal"
        )

    def verificar_diccionario_animales(self):
        self.logger.info("\nVERIFICANDO LISTA DE ANIMALES...")
        total_animales = len(self.animales_carac)
        self.logger.info(f"  Total de animales: {total_animales}")
        if total_animales != len(self.config['animales']):
            self.logger.error(f"Se esperaban {len(self.config['animales'])} animales, pero hay {total_animales}")
            return False
        self.logger.info(f"Lista correcta - {total_animales} animales")
        self.logger.info("Lista completa de animales:")
        for i, animal in enumerate(sorted(self.animales_carac.keys()), 1):
            self.logger.info(f"   {i:2d}. {animal}")
        return True

    def validar_animal(self, animal):
        return self._resolver_animal(animal)

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
        df['Hora_Sorteo'] = df['Hora'].astype(str).str.strip().str.zfill(8)
        # Features basadas en Num_Int
        df['Num_Int_Prev'] = df['Num_Int'].shift(1)
        df['Dif_Ciclica_N'] = df.apply(
            lambda row: self.calcular_diferencia_ciclica(row['Num_Int'], row['Num_Int_Prev'], max_val=38),
            axis=1
        )
        df['Media_5_N'] = df['Num_Int'].rolling(5, min_periods=1).mean()
        df['Std_5_N'] = df['Num_Int'].rolling(5, min_periods=1).std()
        df['Repite_Num'] = (df['Num_Int'] == df['Num_Int_Prev']).astype(int)
        df['Mismo_Num_3'] = ((df['Num_Int'] == df['Num_Int'].shift(1)) & (df['Num_Int'] == df['Num_Int'].shift(2))).astype(int)

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

        # --- Color y Paridad de la ruleta ---
        def color_numero(num):
            if num == 0 or num == 37:
                return 0  # Verde
            rojos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
            return 1 if num in rojos else 2  # 1=Rojo, 2=Negro

        def paridad_numero(num):
            if num == 0 or num == 37:
                return 0  # Especial
            return 1 if num % 2 == 0 else 2  # 1=Par, 2=Impar

        df['Color_Numero'] = df['Numero'].apply(color_numero)
        df['Color_Previo'] = df['Color_Numero'].shift(1)
        df['Paridad_Numero'] = df['Numero'].apply(paridad_numero)
        df['Paridad_Previo'] = df['Paridad_Numero'].shift(1)

        from collections import defaultdict, deque
        # Expanding window features (no future leakage)
        hour_animal_cnt = defaultdict(lambda: defaultdict(int))
        hour_total_cnt = defaultdict(int)
        dia_animal_cnt = defaultdict(lambda: defaultdict(int))
        dia_total_cnt = defaultdict(int)
        trans_cnt = defaultdict(lambda: defaultdict(int))
        trans_total_cnt = defaultdict(int)
        last_pos = {}
        # Contadores basados en Num_Int
        hour_num_cnt = defaultdict(lambda: defaultdict(int))
        hour_num_total = defaultdict(int)
        last_num_pos = {}
        # Co-ocurrencia en el mismo dia
        cooc_cnt = defaultdict(lambda: defaultdict(int))
        cooc_total = defaultdict(int)
        # Lift por dia de semana
        dia_num_cnt = defaultdict(lambda: defaultdict(int))
        dia_num_total = defaultdict(int)
        global_num_cnt = defaultdict(int)
        global_total = 0
        prob_hora_vals = []
        prob_dia_vals = []
        prob_trans_vals = []
        freq10_vals = []
        distancia_vals = []
        prob_hora_num_vals = []
        gap_num_vals = []
        racha_num_vals = []
        prob_cooc_vals = []
        lift_dia_vals = []
        recent_window = deque(maxlen=10)
        recent_num_window = deque(maxlen=10)
        # Track last fecha for co-occurrence
        last_fecha = None
        last_num_int = None
        for idx in range(len(df)):
            cur_animal = df.iloc[idx]['Animal']
            cur_num = df.iloc[idx]['Num_Int']
            cur_hour = df.iloc[idx]['Solo_hora']
            cur_dia = df.iloc[idx]['Dia_Semana']
            cur_fecha = df.iloc[idx]['Fecha'] if 'Fecha' in df.columns else None
            # Calculate probabilities using PREVIOUS data only (no look-ahead)
            prob_hora_vals.append(
                (hour_animal_cnt[cur_hour][cur_animal] / hour_total_cnt[cur_hour] * 100)
                if hour_total_cnt[cur_hour] > 0 else 0.0
            )
            hour_animal_cnt[cur_hour][cur_animal] += 1
            hour_total_cnt[cur_hour] += 1

            prob_dia_vals.append(
                (dia_animal_cnt[cur_dia][cur_animal] / dia_total_cnt[cur_dia] * 100)
                if dia_total_cnt[cur_dia] > 0 else 0.0
            )
            dia_animal_cnt[cur_dia][cur_animal] += 1
            dia_total_cnt[cur_dia] += 1

            if idx > 0:
                prev = df.iloc[idx-1]['Animal']
                if df.iloc[idx-1]['Fecha'] == df.iloc[idx]['Fecha']:
                    trans_cnt[prev][cur_animal] += 1
                    trans_total_cnt[prev] += 1
                prob_trans_vals.append(
                    (trans_cnt[prev][cur_animal] / trans_total_cnt[prev] * 100)
                    if trans_total_cnt[prev] > 0 else 0.0
                )
            else:
                prob_trans_vals.append(0.0)

            # (Markov×Hora / Markov×Dia removed — keep only global transition probs)

            # Frequency of this animal in last 10 draws (before current)
            freq10_vals.append(recent_window.count(cur_animal))
            recent_window.append(cur_animal)
            # Draws since this animal last appeared before current draw
            if cur_animal in last_pos:
                distancia_vals.append(idx - last_pos[cur_animal] - 1)
            else:
                distancia_vals.append(-1)
            last_pos[cur_animal] = idx
            # --- Features basadas en Num_Int (expanding) ---
            prob_hora_num_vals.append(
                (hour_num_cnt[cur_hour][cur_num] / hour_num_total[cur_hour] * 100)
                if hour_num_total[cur_hour] > 0 else 0.0
            )
            hour_num_cnt[cur_hour][cur_num] += 1
            hour_num_total[cur_hour] += 1
            # Gap (sorteos desde ultima aparicion de este Num_Int)
            if cur_num in last_num_pos:
                gap_num_vals.append(idx - last_num_pos[cur_num] - 1)
            else:
                gap_num_vals.append(-1)
            last_num_pos[cur_num] = idx
            # Racha del mismo Num_Int consecutivo
            if idx > 0 and df.iloc[idx-1]['Num_Int'] == cur_num:
                racha_num_vals.append(racha_num_vals[-1] + 1 if racha_num_vals else 2)
            else:
                racha_num_vals.append(1)
            # Co-ocurrencia en el mismo dia
            if cur_fecha is not None and last_num_int is not None and last_fecha == cur_fecha:
                prob_cooc_vals.append(
                    (cooc_cnt[last_num_int][cur_num] / cooc_total[last_num_int] * 100)
                    if cooc_total[last_num_int] > 0 else 0.0
                )
            else:
                prob_cooc_vals.append(0.0)
            if cur_fecha is not None and last_num_int is not None and last_fecha == cur_fecha:
                cooc_cnt[last_num_int][cur_num] += 1
                cooc_total[last_num_int] += 1
            # Lift por dia de semana
            prob_global_num = (global_num_cnt[cur_num] / global_total * 100) if global_total > 0 else 0.0
            prob_dia_num = (dia_num_cnt[cur_dia][cur_num] / dia_num_total[cur_dia] * 100) if dia_num_total[cur_dia] > 0 else 0.0
            if prob_global_num > 0:
                lift_dia_vals.append((prob_dia_num / prob_global_num - 1) * 100)
            else:
                lift_dia_vals.append(0.0)
            global_num_cnt[cur_num] += 1
            global_total += 1
            dia_num_cnt[cur_dia][cur_num] += 1
            dia_num_total[cur_dia] += 1
            last_fecha = cur_fecha
            last_num_int = cur_num
        df['Prob_Hist_Hora'] = prob_hora_vals
        df['Prob_Hist_Dia'] = prob_dia_vals
        df['Prob_Trans_Markov'] = prob_trans_vals
        df['Frecuencia_10'] = freq10_vals
        df['Sorteos_Desde_Aparicion'] = distancia_vals
        df['Prob_Num_Hora'] = prob_hora_num_vals
        df['Gap_Num'] = gap_num_vals
        df['Racha_Num'] = racha_num_vals
        df['Prob_Cooc'] = prob_cooc_vals
        df['Lift_Dia'] = lift_dia_vals

        try:
            self.logger.info(f"Caracteristicas avanzadas anadidas: {len(df.columns)} features totales")
        except Exception:
            print(f"Caracteristicas avanzadas anadidas: {len(df.columns)} features totales")
        return df

    def preparar_datos_ml_completo(self, datos):
        df_ml = datos.copy()
        animales_validos = list(self.animales_carac.keys())
        df_ml = df_ml[df_ml['Animal'].isin(animales_validos)].copy()
        df_ml = df_ml[df_ml['Num_Int'].between(0, 37)].copy()
        if len(df_ml) < 50:
            self.logger.warning(f"Solo {len(df_ml)} registros validos. Se recomiendan al menos 50.")
        # Filter to period where all Num_Int values (0-37) are present
        if 'Fecha' in df_ml.columns:
            sorted_dates = sorted(df_ml['Fecha'].unique())
            all_nums = set(range(38))
            seen_nums = set()
            min_date = sorted_dates[0]
            for d in sorted_dates:
                df_dia = df_ml[df_ml['Fecha'] == d]
                seen_nums.update(df_dia['Num_Int'].unique())
                if seen_nums == all_nums:
                    min_date = d
                    break
            before = len(df_ml)
            df_ml = df_ml[df_ml['Fecha'] >= min_date].copy()
            after = len(df_ml)
            if before != after:
                self.logger.info(f"Filtrados datos desde {min_date}: {before} -> {after} registros (quitados {before-after} historicos)")
        numeric_features = ['Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num',
                            'Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                            'Frecuencia_10', 'Sorteos_Desde_Aparicion',
                            'Color_Previo', 'Paridad_Previo',
                            'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                            'Prob_Cooc', 'Lift_Dia']
        categorical_features = ['Hora_Sorteo']
        if 'Hora_Sorteo' not in df_ml.columns:
            df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip().str.zfill(8)
        # Target: Num_Int (0-37) instead of encoded animal names
        le_y = LabelEncoder()
        le_y.fit(sorted(range(38)))
        Y = df_ml['Num_Int']
        self.logger.info(f"Target: Num_Int (clases 0-37, {Y.nunique()} valores unicos)")
        available_features = []
        for feature in numeric_features + categorical_features:
            if feature in df_ml.columns:
                available_features.append(feature)
        X = df_ml[available_features].copy()
        filas_antes = len(X)
        X = X.dropna()
        filas_despues = len(X)
        if filas_antes != filas_despues:
            self.logger.info(f"Eliminadas {filas_antes - filas_despues} filas con valores NaN")
        Y = Y.loc[X.index]
        self.logger.info(f"Datos ML preparados: {len(X)} muestras, {len(available_features)} caracteristicas")
        self.logger.debug(f"   Caracteristicas: {available_features}")
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
        self.logger.info(f"Entrenando {modelo_nombre} con {len(X)} sorteos")
        tscv = TimeSeriesSplit(n_splits=5)
        pipeline = self.crear_pipeline_ml(modelo, numeric_features, categorical_features)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            pipeline.fit(X_train, Y_train)
            accuracy = pipeline.score(X_test, Y_test)
            self.logger.info(f"Precision de la validacion temporal: {accuracy:.2%}")
        return pipeline

    def predecir_top_k_por_hora(self, pipeline, le_y, df_ml, k=25):
        matriz_prediccion_ia = {}
        if 'Hora_Sorteo' not in df_ml.columns:
            df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip().str.zfill(8)
        horas_sorteo = sorted(df_ml['Hora_Sorteo'].unique())
        numeric_candidates = ['Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num',
                              'Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion',
                              'Color_Previo', 'Paridad_Previo',
                              'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Prob_Cooc', 'Lift_Dia']
        available_numeric = [f for f in numeric_candidates if f in df_ml.columns]
        all_features = available_numeric + ['Hora_Sorteo']
        self.logger.info(f"Generando Matriz de Prediccion TOP-{k} de la IA ({len(all_features)} features)...")
        for hora in horas_sorteo:
            df_hora = df_ml[df_ml['Hora_Sorteo'] == hora].iloc[[-1]].copy()
            if df_hora.isnull().any().any() or df_hora.empty:
                continue
            X_query = df_hora[all_features]
            try:
                y_proba = pipeline.predict_proba(X_query)[0]
                indices_top_k = np.argsort(y_proba)[::-1][:k]
                # indices son directamente los Num_Int predecidos (0-37)
                matriz_prediccion_ia[hora] = indices_top_k.tolist()
            except Exception as e:
                self.logger.warning(f"Error en hora {hora}: {e}")
                continue
        self.logger.info(f"Matriz generada con {len(matriz_prediccion_ia)} horas")
        return matriz_prediccion_ia

    def _top25_cv_scorer(self, estimator, X, y):
        try:
            y_proba = estimator.predict_proba(X)
            top20 = np.argsort(y_proba, axis=1)[:, -25:]
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
            scoring=self._top25_cv_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1,
            error_score=0.0
        )
        random_search.fit(X, Y)
        self.logger.info(f"Mejores parametros RF: {random_search.best_params_}")
        self.logger.info(f"Mejor score RF (Top-25): {random_search.best_score_:.4f}")
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
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        pipeline_base = self.crear_pipeline_ml(modelo_base, numeric_features, categorical_features)
        tscv = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            pipeline_base,
            param_distributions=param_dist,
            n_iter=10,
            cv=tscv,
            scoring=self._top25_cv_scorer,
            n_jobs=1,
            random_state=42,
            verbose=1,
            error_score=0.0
        )
        random_search.fit(X, Y)
        self.logger.info(f"Mejores parametros XGB: {random_search.best_params_}")
        self.logger.info(f"Mejor score XGB (Top-25): {random_search.best_score_:.4f}")
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
        top25_accuracies = []
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            try:
                modelo_optimizado.fit(X_train, Y_train)
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} fit failed: {e}")
                continue
            try:
                accuracy = modelo_optimizado.score(X_test, Y_test)
                y_proba = modelo_optimizado.predict_proba(X_test)
                top25_acc = self.calcular_precision_top_k(Y_test.values, y_proba, k=25)
                accuracies.append(accuracy)
                top25_accuracies.append(top25_acc)
                self.logger.info(f"Fold {fold+1}: Accuracy = {accuracy:.2%}, Top-25 = {top25_acc:.2%}")
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} evaluation failed: {e}")
        tiempo_entrenamiento = datetime.now() - start_time
        avg_accuracy = np.mean(accuracies)
        avg_top25 = np.mean(top25_accuracies)
        self.logger.info(f"RESULTADOS {modelo_nombre}:")
        self.logger.info(f"   Accuracy Promedio: {avg_accuracy:.2%}")
        self.logger.info(f"   Top-25 Accuracy: {avg_top25:.2%}")
        self.logger.info(f"   Tiempo entrenamiento: {tiempo_entrenamiento}")
        self.logger.info(f"   Mejor Fold: {max(accuracies):.2%}")
        self.logger.info(f"Entrenamiento completado: {avg_accuracy:.2%} accuracy, {avg_top25:.2%} top-25")
        return modelo_optimizado

    def guardar_modelo(self, modelo, le_y, metricas, nombre_modelo):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_dir = f"{self.config['modelos_dir']}/{nombre_modelo}_{timestamp}"
        if not os.path.exists(self.config['modelos_dir']):
            os.makedirs(self.config['modelos_dir'])
        if not os.path.exists(modelo_dir):
            os.makedirs(modelo_dir)
        modelo_path = f"{modelo_dir}/modelo.joblib"
        joblib.dump(modelo, modelo_path)
        le_path = f"{modelo_dir}/label_encoder.joblib"
        joblib.dump(le_y, le_path)
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
            modelo_path = f"{modelo_dir}/modelo.joblib"
            le_path = f"{modelo_dir}/label_encoder.joblib"
            if os.path.exists(modelo_path) and os.path.exists(le_path):
                modelo = joblib.load(modelo_path)
                le_y = joblib.load(le_path)
            else:
                # Fallback to old pickle files for backward compatibility
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
                    self.logger.info(f"Modelo cargado: {ultimo_modelo_dir}")
                    self.logger.info(f"   Fecha entrenamiento: {metricas.get('fecha_entrenamiento', 'N/A')}")
                    self.logger.info(f"   Muestras: {metricas.get('num_muestras', 'N/A')}")
                    acc = metricas.get('accuracy_promedio', None)
                    if acc is not None:
                        self.logger.info(f"   Accuracy: {acc:.2%}")
                    else:
                        self.logger.info(f"   Accuracy: N/A")
                return modelo, le_y, metricas
        self.logger.warning(f"No se encontraron modelos de tipo: {tipo_modelo}")
        return None, None, None

    def random_forest_optimizado(self, datos):
        try:
            self.logger.info("Ejecutando Random Forest Optimizado")
            datos_con_features = self.agregar_caracteristicas_avanzadas(datos.copy())
            X, Y, le_y, numeric_features, categorical_features, available_features = self.preparar_datos_ml_completo(datos_con_features)
            if len(X) < 50:
                self.logger.warning(f"Datos insuficientes: {len(X)} muestras (minimo 50 recomendado)")
                self.logger.info("Se recomiendan al menos 50 muestras para optimizacion")
                return None
            modelo_optimizado = self.entrenar_modelo_con_optimizacion(
                X, Y, 'rf', numeric_features, categorical_features
            )
            matriz_prediccion = self.predecir_top_k_por_hora(
                modelo_optimizado, le_y, datos_con_features.copy(), k=25
            )
            metricas = {
                'accuracy_promedio': np.mean([modelo_optimizado.score(X, Y)]),
                'num_muestras': len(X),
                'caracteristicas': available_features,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            modelo_dir = self.guardar_modelo(modelo_optimizado, le_y, metricas, "random_forest")
            self.logger.info(f"Modelo optimizado guardado en: {modelo_dir}")
            return matriz_prediccion
        except Exception as e:
            self.logger.error(f"Error en Random Forest optimizado: {e}")
            self.logger.exception(e)
            return None

    def xgboost_optimizado(self, datos):
        try:
            self.logger.info("Ejecutando XGBoost Optimizado")
            datos_con_features = self.agregar_caracteristicas_avanzadas(datos.copy())
            X, Y, le_y, numeric_features, categorical_features, available_features = self.preparar_datos_ml_completo(datos_con_features)
            if len(X) < 50:
                self.logger.warning(f"Datos insuficientes: {len(X)} muestras")
                self.logger.info("Se recomiendan al menos 50 muestras para optimizacion")
                return None
            modelo_optimizado = self.entrenar_modelo_con_optimizacion(
                X, Y, 'xgb', numeric_features, categorical_features
            )
            matriz_prediccion = self.predecir_top_k_por_hora(
                modelo_optimizado, le_y, datos_con_features.copy(), k=25
            )
            metricas = {
                'accuracy_promedio': np.mean([modelo_optimizado.score(X, Y)]),
                'num_muestras': len(X),
                'caracteristicas': available_features,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            modelo_dir = self.guardar_modelo(modelo_optimizado, le_y, metricas, "xgboost")
            self.logger.info(f"Modelo optimizado guardado en: {modelo_dir}")
            return matriz_prediccion
        except Exception as e:
            self.logger.error(f"Error en XGBoost optimizado: {e}")
            self.logger.exception(e)
            return None

    def evaluacion_estrategia_frecuencia(self, datos):
        self.logger.info("Evaluacion Estrategia DINAMICA (BASE: Frecuencia Historica por Num_Int)")
        frecuencia_completa = datos.groupby('Hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map = {}
        for hora_24h in frecuencia_completa['Hora'].unique():
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(25)['Num_Int'].tolist()
            top_10_map[hora_24h] = top_10_lista
        self.logger.info("Lista Top-25 generada para todas las horas.")
        self.simular_estrategia(datos, top_10_map)

    def evaluacion_estrategia_ia(self, datos, matriz_prediccion_ia):
        self.logger.info("Evaluacion Estrategia DINAMICA (OPTIMIZADA: Prediccion de IA)")
        top_10_map = matriz_prediccion_ia
        self.logger.info(f"Matriz de prediccion cargada con {len(top_10_map)} horas.")
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
                num_salio = row['Num_Int']
                if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
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
                    num_salio = row['Num_Int']
                    if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
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
        self.logger.info("\n" + "="*70)
        self.logger.info("        RESUMEN DE LA EVALUACION DE LA ESTRATEGIA DINAMICA")
        self.logger.info("="*70)
        self.logger.info(f"Dias Completos Analizados: {total_dias}")
        self.logger.info(f"Dias Jugados: {total_dias_jugados}")
        self.logger.info("-" * 70)
        self.logger.info(f"Gasto Total: {gasto_total:,.2f} Bs")
        self.logger.info(f"Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
        self.logger.info(f"GANANCIA/PERDIDA NETA TOTAL: {ganancia_neta_total:,.2f} Bs")
        self.logger.info("-" * 70)
        if gasto_total > 0:
            roi = (ganancia_neta_total / gasto_total) * 100
            self.logger.info(f"Retorno de la Inversion (ROI): {roi:,.2f}%")
        if ganancia_neta_total > 0:
            self.logger.info("La estrategia genero ganancias.")
        elif ganancia_neta_total < 0:
            self.logger.warning("La estrategia genero perdidas.")
        else:
            self.logger.info("Resultado: Punto de Equilibrio.")
        self.logger.info("Auditoria Diaria de Dias Jugados ---")
        df_jugados = df_resultados[df_resultados['Jugar_Tarde'] == 'SI']
        if not df_jugados.empty:
            top_10_mejor = df_jugados.sort_values(by='Ganancia_Neta', ascending=False).head(10)
            top_10_peor = df_jugados.sort_values(by='Ganancia_Neta', ascending=True).head(10)
            self.logger.info("TOP 10 Dias con Mayor Ganancia Neta:")
            self.logger.info("\n" + top_10_mejor[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']].to_string(index=False))
            self.logger.info("TOP 10 Dias con Mayor Perdida Neta:")
            self.logger.info("\n" + top_10_peor[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']].to_string(index=False))
        else:
            print("No hubo dias suficientes.")
        return df_resultados

    def mostrar_matriz_prediccion(self, matriz_prediccion):
        self.logger.info("MATRIZ DE PREDICCION - TOP 20 POR HORA")
        for hora, nums in sorted(matriz_prediccion.items()):
            lines = [f"Hora {hora}:"]
            for i, n in enumerate(nums[:25], 1):
                animal = self.num_int_a_animal.get(n, "?")
                lines.append(f"    {i:2d}. {n:2d} ({animal})")
            self.logger.info("\n" + "\n".join(lines))

    def prediccion_hoy_ensemble(self, datos, modelo=None, le_y=None, k=25):
        df = datos.copy()
        ultimo = df.iloc[-1]
        ultimo_num = int(ultimo['Num_Int'])
        trans_prob, trans_total = self._transiciones_markov(df)
        prob_hora = self._frecuencias_hora(df, 'Hora')
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion',
                              'Color_Previo', 'Paridad_Previo',
                              'Prob_Cooc', 'Lift_Dia', 'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num']
        available_numeric = [f for f in numeric_candidates if f in df.columns]
        all_nums = list(range(38))
        horas_del_dia = sorted(df['Hora'].unique())
        print("\n" + "=" * 74)
        print("  PREDICCION COMBINADA PARA HOY (Ensemble)")
        print("=" * 74)
        print(f"  Ultimo sorteo: {ultimo_num:2d} ({self.num_int_a_animal.get(ultimo_num, '?')}) a las {ultimo['Hora']}")
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
                        indices_top_k = np.argsort(y_proba)[::-1][:25]
                        ml_top10_por_hora[hora_24h] = set(indices_top_k)
                except Exception:
                    pass
        for hora_24h in horas_del_dia:
            h_stripped = hora_24h.split(':')[0] + ':' + hora_24h.split(':')[1]
            try:
                hora_12h = datetime.strptime(h_stripped, '%H:%M').strftime('%I:%M %p').lstrip('0')
            except Exception:
                hora_12h = hora_24h
            markov_scores = {}
            if ultimo_num in trans_total and trans_total[ultimo_num] > 0:
                for n in all_nums:
                    p = trans_prob.get((ultimo_num, n), 0)
                    if p > 0:
                        markov_scores[n] = p
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
                            n = i
                            ml_scores[n] = prob * 100
                except Exception:
                    ml_ok = False
            all_nums_set = set(list(markov_scores.keys()) + list(hourly_scores.keys()) + list(ml_scores.keys()))
            if not all_nums_set:
                all_nums_set = set(all_nums)
            max_m = max(markov_scores.values()) if markov_scores else 1
            max_h = max(hourly_scores.values()) if hourly_scores else 1
            max_ml = max(ml_scores.values()) if ml_scores else 1
            w_m, w_h, w_ml = (0.35, 0.35, 0.30) if ml_ok else (0.50, 0.50, 0)
            ensemble_scores = []
            ml_top10 = ml_top10_por_hora.get(hora_24h, set())
            for n in all_nums_set:
                m = markov_scores.get(n, 0)
                h = hourly_scores.get(n, 0)
                ml = ml_scores.get(n, 0)
                m_norm = m / max_m if max_m > 0 else 0
                h_norm = h / max_h if max_h > 0 else 0
                ml_norm = ml / max_ml if max_ml > 0 else 0
                models_cnt = (1 if m > 0 else 0) + (1 if h > 0 else 0) + (1 if n in ml_top10 else 0)
                score = m_norm * w_m + h_norm * w_h + ml_norm * w_ml
                if models_cnt >= 2 and ml_ok:
                    score *= 1.15
                ensemble_scores.append((n, score, m, h, ml, models_cnt))
            ensemble_scores.sort(key=lambda x: x[1], reverse=True)
            topk = ensemble_scores[:k]
            resultados[hora_24h] = topk
            print(f"\nHora {hora_12h}")
            print(f"   {'#':<3} {'Num(Animal)':<14} {'Ensemble':<9} {'Markov':<7} {'Hist':<7} {'ML':<8} {'M':<3}")
            print(f"   {'-'*53}")
            for i, (n, ens, m, h, ml, mc) in enumerate(topk, 1):
                ms = f"{m:.1f}%" if m > 0 else "-"
                hs = f"{h:.1f}%" if h > 0 else "-"
                mls = f"{ml:.1f}%" if ml > 0 else "-"
                conf = "*" * mc if mc >= 2 else ""
                animal = self.num_int_a_animal.get(n, "?")
                print(f"   {i:<3} {n:2d}({animal:<10}) {ens*100:<9.1f} {ms:<7} {hs:<7} {mls:<8} {conf:<3}")
        pred_matrix = {h: [a[0] for a in r] for h, r in resultados.items()}
        return pred_matrix

    def prediccion_completa_hoy(self, datos, modelo_rf=None, le_rf=None, modelo_xgb=None, le_xgb=None):
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
        trans_prob, trans_total = self._transiciones_markov(df)
        prob_hora = self._frecuencias_hora(df, 'Hora')
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion',
                              'Color_Previo', 'Paridad_Previo',
                              'Prob_Cooc', 'Lift_Dia', 'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num']
        available_numeric = [f for f in numeric_candidates if f in df.columns]
        ultimo_num = int(ultimo['Num_Int'])
        markov_scores = {}
        if ultimo_num in trans_total:
            for n in range(38):
                p = trans_prob.get((ultimo_num, n), 0)
                if p > 0: markov_scores[n] = p
        markov_top = sorted(markov_scores, key=markov_scores.get, reverse=True)[:25]
        hourly_scores = prob_hora.get(hora_target, {})
        hourly_top = sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:25]
        rf_list = []
        xgb_list = []
        if modelo_rf and available_numeric:
            try:
                df_hora = df[df['Hora'] == hora_target].iloc[[-1]].copy()
                df_hora['Hora_Sorteo'] = hora_target
                X_q = df_hora[available_numeric + ['Hora_Sorteo']]
                if not X_q.isnull().any().any():
                    yp = modelo_rf.predict_proba(X_q)[0]
                    rf_list = le_rf.inverse_transform(np.argsort(yp)[::-1][:25]).tolist()
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
                    xgb_list = le_xgb.inverse_transform(np.argsort(yp)[::-1][:25]).tolist()
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
        for i, (a, cnt, _) in enumerate(scored[:25], 1):
            mods = []
            pos = []
            if a in markov_top: mods.append("M"); pos.append(f"M#{markov_top.index(a)+1}")
            if a in hourly_top: mods.append("H"); pos.append(f"H#{hourly_top.index(a)+1}")
            if a in rf_list: mods.append("RF"); pos.append(f"RF#{rf_list.index(a)+1}")
            if a in xgb_list: mods.append("X"); pos.append(f"X#{xgb_list.index(a)+1}")
            print(f"  {i:<3} {a:<12} {cnt}/4{'':<6} {','.join(mods):<12} {','.join(pos):<24}")
        print(f"\n  Jugada recomendada: Top-3 con mayor consenso")
        return None

    def prediccion_modelo_simple(self, datos, animal, hora_str, modelo, le_y, nombre_modelo):
        """Retorna top-25 [(num_int, animal_name, prob_pct)] dados animal, hora y modelo.
        animal: nombre de animal o numero entero 0-37."""
        d = datos.copy()
        animal_resuelto = self._resolver_animal(animal)
        num = self.animal_a_num_int.get(animal_resuelto)
        if num is None:
            return []
        d.iloc[-1, d.columns.get_loc('Num_Int')] = num
        d.iloc[-1, d.columns.get_loc('Numero')] = num
        d.iloc[-1, d.columns.get_loc('Animal')] = animal
        d2 = self.agregar_caracteristicas_avanzadas(d)
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion',
                              'Color_Previo', 'Paridad_Previo',
                              'Prob_Cooc', 'Lift_Dia', 'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num']
        available_numeric = [f for f in numeric_candidates if f in d2.columns]
        df_hora = d2[d2['Hora'] == hora_str].iloc[[-1]].copy()
        if df_hora.empty:
            return []
        df_hora['Hora_Sorteo'] = hora_str
        X = df_hora[available_numeric + ['Hora_Sorteo']]
        if X.isnull().any().any():
            return []
        yp = modelo.predict_proba(X)[0]
        top_idx = np.argsort(yp)[::-1][:25]
        top_nums = le_y.inverse_transform(top_idx)
        result = []
        for idx, n in zip(top_idx, top_nums):
            n_int = int(n)
            a = self.num_int_a_animal.get(n_int, '?')
            result.append((n_int, a, yp[idx] * 100))
        return result

    def imprimir_prediccion_modelo(self, datos, animal, hora_str, modelo, le_y, nombre_modelo):
        """Imprime top-25 de un modelo (RF o XGB) para animal y hora dados."""
        items = self.prediccion_modelo_simple(datos, animal, hora_str, modelo, le_y, nombre_modelo)
        if not items:
            print(f"  No se pudo generar prediccion {nombre_modelo} para {animal}")
            return
        try:
            hora_12h = datetime.strptime(hora_str, '%H:%M:%S').strftime('%I:%M %p').lstrip('0')
        except Exception:
            hora_12h = hora_str
        print(f"\n{'='*60}")
        print(f"  PREDICCION {nombre_modelo}: {animal} -> {hora_12h}")
        print(f"{'='*60}")
        print(f"  {'#':>3} {'Num(Animal)':<18} {'Prob':>6}")
        print(f"  {'-'*30}")
        for i, (n, a, p) in enumerate(items, 1):
            print(f"  {i:3d} {str(n):>2} ({a:<14}) {p:>5.1f}%")
        print(f"{'='*60}\n")

    def evaluar_predicciones_historicas(self, datos, modelo_rf=None, le_rf=None, modelo_xgb=None, le_xgb=None, n_ultimos=500):
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        n_eval = min(n_ultimos, len(df) - 5)
        df_eval = df.tail(n_eval).reset_index(drop=True)
        if 'Hora_Sorteo' not in df_eval.columns:
            df_eval['Hora_Sorteo'] = df_eval['Hora'].astype(str).str.strip().str.zfill(8)
        numeric_candidates = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                              'Frecuencia_10', 'Sorteos_Desde_Aparicion',
                              'Color_Previo', 'Paridad_Previo',
                              'Prob_Cooc', 'Lift_Dia', 'Racha_Num', 'Mismo_Num_3', 'Media_5_N', 'Std_5_N',
                              'Dif_Ciclica_N', 'Prob_Num_Hora', 'Gap_Num', 'Repite_Num']
        available_numeric = [f for f in numeric_candidates if f in df_eval.columns]
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        freq_hora = self._frecuencias_hora(df, 'Solo_hora')
        global_freq = df['Num_Int'].value_counts(normalize=True).mul(100)
        global_top25 = set(global_freq.head(25).index)
        resultados = []
        rf_count = 0
        xgb_count = 0
        numeros_validos = list(range(38))
        for i in range(4, len(df_eval) - 1):
            if i + 1 >= len(df_eval):
                break
            if df_eval.iloc[i]['Fecha'] != df_eval.iloc[i + 1]['Fecha']:
                continue
            prev_state = df_eval.iloc[i]
            actual = df_eval.iloc[i + 1]
            num_real = actual['Num_Int']
            hora_real = actual['Solo_hora']
            fecha = actual['Fecha']
            markov_scores = {}
            ultimo_num = prev_state['Num_Int']
            if ultimo_num in trans_total and trans_total[ultimo_num] > 0:
                for n in numeros_validos:
                    p = trans_prob.get((ultimo_num, n), 0)
                    if p > 0:
                        markov_scores[n] = p
            markov_top = sorted(markov_scores, key=markov_scores.get, reverse=True)[:25]
            markov_rank = markov_top.index(num_real) + 1 if num_real in markov_top else None
            markov_full = sorted(markov_scores, key=markov_scores.get, reverse=True)
            markov_all_rank = markov_full.index(num_real) + 1 if num_real in markov_full else None
            hourly_scores = {}
            if hora_real in freq_hora:
                for n, p in freq_hora[hora_real].items():
                    hourly_scores[n] = p
            hourly_top = sorted(hourly_scores, key=hourly_scores.get, reverse=True)[:25]
            hourly_rank = hourly_top.index(num_real) + 1 if num_real in hourly_top else None
            hourly_full = sorted(hourly_scores, key=hourly_scores.get, reverse=True)
            hourly_all_rank = hourly_full.index(num_real) + 1 if num_real in hourly_full else None
            combined_mh_scores = {}
            for n in markov_scores:
                hp = hourly_scores.get(n, 0)
                combined_mh_scores[n] = markov_scores[n] + hp
            combined_mh_top = sorted(combined_mh_scores, key=combined_mh_scores.get, reverse=True)[:25]
            combined_mh_rank = combined_mh_top.index(num_real) + 1 if num_real in combined_mh_top else None
            hp_h = prev_state['Hora']
            hn_h = actual['Hora']
            pareja_h = (hp_h, hn_h)
            markov_hora_scores = {}
            if pareja_h in trans_h and ultimo_num in total_h.get(pareja_h, {}):
                tot_hh = total_h[pareja_h][ultimo_num]
                if tot_hh > 0:
                    markov_hora_scores = {n2: cnt / tot_hh * 100 for n2, cnt in trans_h[pareja_h][ultimo_num].items()}
            # Fill to 25 with global Markov probabilities (by highest prob, not numeric order)
            if len(markov_hora_scores) < 25 and ultimo_num in trans_total and trans_total[ultimo_num] > 0:
                candidates = [(n2, trans_prob.get((ultimo_num, n2), 0)) for n2 in range(38)
                              if n2 not in markov_hora_scores and trans_prob.get((ultimo_num, n2), 0) > 0]
                candidates.sort(key=lambda x: x[1], reverse=True)
                for n2, p in candidates:
                    markov_hora_scores[n2] = p
                    if len(markov_hora_scores) >= 25:
                        break
            if len(markov_hora_scores) < 25:
                for n2, pct in global_freq.items():
                    if n2 not in markov_hora_scores:
                        markov_hora_scores[n2] = pct
                        if len(markov_hora_scores) >= 25:
                            break
            markov_hora_top = sorted(markov_hora_scores, key=markov_hora_scores.get, reverse=True)[:25]
            markov_hora_rank = markov_hora_top.index(num_real) + 1 if num_real in markov_hora_top else None
            m_hh = total_h.get(pareja_h, {}).get(ultimo_num, 0)
            w_hh = min(0.9, max(0.1, m_hh / 50))
            todos_hh = set(markov_scores) | set(markov_hora_scores)
            global_hora_scores = {}
            for n2 in todos_hh:
                global_hora_scores[n2] = markov_scores.get(n2, 0) * (1 - w_hh) + markov_hora_scores.get(n2, 0) * w_hh
            global_hora_top = sorted(global_hora_scores, key=global_hora_scores.get, reverse=True)[:25]
            global_hora_rank = global_hora_top.index(num_real) + 1 if num_real in global_hora_top else None
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
                        rf_top = le_rf.inverse_transform(indices[:25]).tolist()
                        rf_all = le_rf.inverse_transform(indices).tolist()
                        rf_rank = rf_top.index(num_real) + 1 if num_real in rf_top else None
                        rf_all_rank = rf_all.index(num_real) + 1 if num_real in rf_all else None
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
                        xgb_top = le_xgb.inverse_transform(indices[:25]).tolist()
                        xgb_all = le_xgb.inverse_transform(indices).tolist()
                        xgb_rank = xgb_top.index(num_real) + 1 if num_real in xgb_top else None
                        xgb_all_rank = xgb_all.index(num_real) + 1 if num_real in xgb_all else None
                except Exception:
                    pass
            combined_rf_xgb_scores = {}
            if rf_top and xgb_top:
                rf_set = set(rf_top)
                xgb_set = set(xgb_top)
                for n in set(rf_top + xgb_top):
                    score = 0
                    if n in rf_top:
                        score += 20 - rf_top.index(n)
                    if n in xgb_top:
                        score += 20 - xgb_top.index(n)
                    combined_rf_xgb_scores[n] = score
                combined_rf_xgb_top = sorted(combined_rf_xgb_scores, key=combined_rf_xgb_scores.get, reverse=True)[:25]
                combined_rf_xgb_rank = combined_rf_xgb_top.index(num_real) + 1 if num_real in combined_rf_xgb_top else None
            else:
                combined_rf_xgb_top = []
                combined_rf_xgb_rank = None
            global_hit = num_real in global_top25
            top1 = xgb_top[0] if xgb_top else rf_top[0] if rf_top else combined_mh_top[0] if combined_mh_top else markov_top[0] if markov_top else hourly_top[0] if hourly_top else "?"
            acertado = (xgb_top and num_real in xgb_top) or (rf_top and num_real in rf_top) or (combined_mh_top and num_real in combined_mh_top) or (markov_top and num_real in markov_top) or (hourly_top and num_real in hourly_top)
            resultados.append({
                'fecha': fecha, 'hora': hora_real, 'real': num_real,
                'predicho': top1, 'acertado': acertado,
                'markov_rank': markov_rank, 'markov_all_rank': markov_all_rank,
                'hourly_rank': hourly_rank, 'hourly_all_rank': hourly_all_rank,
                'combined_mh_rank': combined_mh_rank,
                'markov_hora_rank': markov_hora_rank,
                'global_hora_rank': global_hora_rank,
                'rf_rank': rf_rank, 'rf_all_rank': rf_all_rank,
                'xgb_rank': xgb_rank, 'xgb_all_rank': xgb_all_rank,
                'combined_rf_xgb_rank': combined_rf_xgb_rank,
                'global_hit': global_hit,
            })
        print(f"\n{'='*90}")
        print(f"  EVALUACION AUTOMATICA: Prediccion vs Realidad")
        print(f"  Ultimos {len(resultados)} sorteos analizados")
        print(f"{'='*90}")
        print(f"{'Fecha':<12} {'Hora':<6} {'Predicho':<13} {'Real':<13} {'Hit':<4} {'Rk-M':<5} {'Rk-MkH':<7} {'Rk-GxH':<7} {'Rk-H':<5} {'Rk-C(M+H)':<10} {'Rk-RF':<6} {'Rk-XGB':<7} {'C(R+X)':<7}")
        print(f"{'-'*107}")
        for r in resultados[-20:]:
            ac = "OK" if r['acertado'] else "NO"
            rk_m = str(r['markov_rank']) if r['markov_rank'] else "-"
            rk_h = str(r['hourly_rank']) if r['hourly_rank'] else "-"
            rk_c = str(r['combined_mh_rank']) if r['combined_mh_rank'] else "-"
            rk_rf = str(r['rf_rank']) if r['rf_rank'] else "-"
            rk_x = str(r['xgb_rank']) if r['xgb_rank'] else "-"
            rk_crx = str(r['combined_rf_xgb_rank']) if r['combined_rf_xgb_rank'] else "-"
            rk_mkh = str(r['markov_hora_rank']) if r['markov_hora_rank'] else "-"
            rk_gxh = str(r['global_hora_rank']) if r['global_hora_rank'] else "-"
            pred_str = f"{r['predicho']:2d}({self.num_int_a_animal.get(r['predicho'], '?'):<10})" if isinstance(r['predicho'], int) else str(r['predicho'])
            real_str = f"{r['real']:2d}({self.num_int_a_animal.get(r['real'], '?'):<10})" if isinstance(r['real'], int) else str(r['real'])
            print(f"{str(r['fecha']):<12} {r['hora']:<6} {pred_str:<13} {real_str:<13} {ac:<4} {rk_m:<5} {rk_mkh:<7} {rk_gxh:<7} {rk_h:<5} {rk_c:<10} {rk_rf:<6} {rk_x:<7} {rk_crx:<7}")
        total = len(resultados)
        if total > 0:
            markov_hits = sum(1 for r in resultados if r['markov_rank'] is not None)
            hourly_hits = sum(1 for r in resultados if r['hourly_rank'] is not None)
            combined_mh_hits = sum(1 for r in resultados if r['combined_mh_rank'] is not None)
            rf_hits = sum(1 for r in resultados if r['rf_rank'] is not None)
            xgb_hits = sum(1 for r in resultados if r['xgb_rank'] is not None)
            combined_rx_hits = sum(1 for r in resultados if r['combined_rf_xgb_rank'] is not None)
            markov_hora_hits = sum(1 for r in resultados if r['markov_hora_rank'] is not None)
            global_hora_hits = sum(1 for r in resultados if r['global_hora_rank'] is not None)
            print(f"\nPRECISION TOP-25 por modelo:")
            print(f"  Markov (M):              {markov_hits}/{total} = {markov_hits/total*100:.1f}%")
            print(f"  Markov x Hora (MkH):     {markov_hora_hits}/{total} = {markov_hora_hits/total*100:.1f}%")
            print(f"  Global x Hora (GxH):     {global_hora_hits}/{total} = {global_hora_hits/total*100:.1f}%")
            print(f"  Hist. Hora (H):          {hourly_hits}/{total} = {hourly_hits/total*100:.1f}%")
            print(f"  M + H:                   {combined_mh_hits}/{total} = {combined_mh_hits/total*100:.1f}%")
            if rf_count > 0:
                print(f"  Random Forest:           {rf_hits}/{rf_count} = {rf_hits/rf_count*100:.1f}%")
            if xgb_count > 0:
                print(f"  XGBoost:                 {xgb_hits}/{xgb_count} = {xgb_hits/xgb_count*100:.1f}%")
            if rf_count > 0 and xgb_count > 0:
                print(f"  RF + XGB:                {combined_rx_hits}/{total} = {combined_rx_hits/total*100:.1f}%")
            global_hits = sum(1 for r in resultados if r['global_hit'])
            print(f"  Global Top-25:           {global_hits}/{total} = {global_hits/total*100:.1f}%")
        HORA_ORDER = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                      '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']
        MODEL_RANK_MAP = {'Markov': 'markov_rank', 'MkHora': 'markov_hora_rank', 'GxHora': 'global_hora_rank',
                          'Hora': 'hourly_rank', 'M+H': 'combined_mh_rank',
                          'RF': 'rf_rank', 'XGB': 'xgb_rank', 'RF+XGB': 'combined_rf_xgb_rank',
                          'Global': 'global_hit'}
        visible_models = ['Markov', 'MkHora', 'GxHora', 'Hora', 'M+H', 'Global']
        if rf_count > 0:
            visible_models.append('RF')
        if xgb_count > 0:
            visible_models.append('XGB')
        if rf_count > 0 and xgb_count > 0:
            visible_models.append('RF+XGB')
        def _fmt(pct):
            return f"{pct:.1f}%" if pct > 50 else "——"
        print(f"\n{'='*90}")
        print(f"  ACIERTOS TOP-25 POR HORA")
        print(f"{'='*90}")
        header = f"{'Hora':<10}" + "".join(f"{m:<10}" for m in visible_models) + f"{'Sorteos':<8}"
        print(header)
        print(f"{'-'*len(header)}")
        for h in HORA_ORDER:
            sub = [r for r in resultados if r['hora'] == h]
            if not sub:
                continue
            n = len(sub)
            cells = [f"{h[:5]:<10}"]
            for m in visible_models:
                rank_key = MODEL_RANK_MAP[m]
                if m == 'Global':
                    hits = sum(1 for r in sub if r['global_hit'])
                else:
                    hits = sum(1 for r in sub if r[rank_key] is not None)
                cells.append(f"{_fmt(hits/n*100):<10}")
            cells.append(f"{n:<8}")
            print("".join(cells))
        print(f"{'='*90}\n")
        return resultados

    def _add_markov_hora_hit(self, df, i, prev_state, actual, num_real, trans_h, total_h, trans_prob=None, trans_total=None):
        hp = prev_state['Hora']
        hn = actual['Hora']
        pareja = (hp, hn)
        markov_hora_scores = {}
        ultimo_num = prev_state['Num_Int']
        if pareja in trans_h and ultimo_num in total_h.get(pareja, {}):
            tot_h = total_h[pareja][ultimo_num]
            if tot_h > 0:
                for n2, cnt in trans_h[pareja][ultimo_num].items():
                    markov_hora_scores[n2] = cnt / tot_h * 100
        if not markov_hora_scores:
            return False
        # Fill to 25 with global Markov probabilities (by highest prob, not numeric order)
        if len(markov_hora_scores) < 25 and trans_prob is not None and trans_total is not None and ultimo_num in trans_total and trans_total[ultimo_num] > 0:
            candidates = [(n2, trans_prob.get((ultimo_num, n2), 0)) for n2 in range(38)
                          if n2 not in markov_hora_scores and trans_prob.get((ultimo_num, n2), 0) > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)
            for n2, p in candidates:
                markov_hora_scores[n2] = p
                if len(markov_hora_scores) >= 25:
                    break
        if len(markov_hora_scores) < 25:
            global_freq = df['Num_Int'].value_counts(normalize=True).mul(100)
            for n2, pct in global_freq.items():
                if n2 not in markov_hora_scores:
                    markov_hora_scores[n2] = pct
                    if len(markov_hora_scores) >= 25:
                        break
        markov_hora_top = sorted(markov_hora_scores, key=markov_hora_scores.get, reverse=True)[:25]
        return num_real in markov_hora_top

    def _eval_gxhora_hit(self, prev_state, actual, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=38):
        """Retorna True si num_real esta en top_k de GxHora para la transicion."""
        ultimo_num = int(prev_state['Num_Int'])
        hp = prev_state['Hora']
        hn = actual['Hora']
        pareja_h = (hp, hn)

        markov_scores = {}
        if ultimo_num in trans_total and trans_total[ultimo_num] > 0:
            for n in range(38):
                p = trans_prob.get((ultimo_num, n), 0)
                if p > 0:
                    markov_scores[n] = p

        markov_hora_scores = {}
        if pareja_h in trans_h and ultimo_num in total_h.get(pareja_h, {}):
            tot_hh = total_h[pareja_h][ultimo_num]
            if tot_hh > 0:
                for n2, cnt in trans_h[pareja_h][ultimo_num].items():
                    markov_hora_scores[n2] = cnt / tot_hh * 100
        if not markov_hora_scores and not markov_scores:
            return False
        # Fill MkHora to top_k (by highest global prob, not numeric order)
        if len(markov_hora_scores) < top_k and ultimo_num in trans_total and trans_total[ultimo_num] > 0:
            candidates = []
            for n2 in range(38):
                if n2 not in markov_hora_scores:
                    p = trans_prob.get((ultimo_num, n2), 0)
                    if p > 0:
                        candidates.append((n2, p))
            candidates.sort(key=lambda x: x[1], reverse=True)
            for n2, p in candidates:
                markov_hora_scores[n2] = p
                if len(markov_hora_scores) >= top_k:
                    break
        if len(markov_hora_scores) < top_k:
            gf = df['Num_Int'].value_counts(normalize=True).mul(100)
            for n2, pct in gf.items():
                if n2 not in markov_hora_scores:
                    markov_hora_scores[n2] = pct
                    if len(markov_hora_scores) >= top_k:
                        break

        m_hh = total_h.get(pareja_h, {}).get(ultimo_num, 0)
        w_hh = min(0.9, max(0.1, m_hh / 50))
        todos = set(markov_scores) | set(markov_hora_scores)
        gxh = {}
        for n2 in todos:
            gxh[n2] = markov_scores.get(n2, 0) * (1 - w_hh) + markov_hora_scores.get(n2, 0) * w_hh
        top = sorted(gxh, key=gxh.get, reverse=True)[:top_k]
        return num_real in top

    def get_matriz_markov_por_dia(self, datos):
        df = datos.copy()
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        matrices = {}
        for dia in df['Dia_Semana'].unique():
            sub = df[df['Dia_Semana'] == dia]
            if len(sub) < 5:
                continue
            matrices[dia] = self._transiciones_markov(sub)
        return matrices

    def prediccion_markov_dia_semana(self, datos, top_k=38, animal=None, hora=None, incluir_trasnocho=False):
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        matrices = self.get_matriz_markov_por_dia(df)
        from datetime import timedelta
        fechas = pd.to_datetime(df['Fecha'].astype(str))
        ultima_fecha = fechas.iloc[-1]

        if animal is not None:
            animal_resuelto = self._resolver_animal(animal)
            ultimo_num = self.animal_a_num_int.get(animal_resuelto)
            if ultimo_num is None:
                print(f"\n  Animal '{animal}' no valido")
                return
            num_animal = animal_resuelto
            if hora is None:
                hora_str = "desconocida"
                dia_hoy = ultima_fecha.strftime('%A')
            else:
                hora_str = hora
                dt_h = pd.to_datetime(hora_str, format="%I:%M %p")
                solo_hora = dt_h.strftime("%H:%M:%S")
                # Detect trasnocho: 7PM -> 8AM next day
                if incluir_trasnocho and solo_hora == '19:00:00':
                    dia_hoy = (ultima_fecha + timedelta(days=1)).strftime('%A')
                    print(f"  (trasnocho: 7PM -> 8AM del dia siguiente)")
                else:
                    dia_hoy = ultima_fecha.strftime('%A')
            print(f"\n{'='*70}")
            print(f"  PREDICCION MARKOV POR DIA DE SEMANA")
            print(f"  Dia: {dia_hoy} | Animal: {num_animal} (#{ultimo_num}) | Hora: {hora_str}")
            print(f"{'='*70}")
        else:
            ultimo = df.iloc[-1]
            dia_hoy = ultima_fecha.strftime('%A')
            print(f"\n{'='*70}")
            print(f"  PREDICCION MARKOV POR DIA DE SEMANA")
            print(f"  Dia: {dia_hoy} | Ultimo sorteo: {ultimo['Solo_hora']} - {ultimo['Animal']} (#{ultimo['Num_Int']})")
            print(f"{'='*70}")
            ultimo_num = int(ultimo['Num_Int'])
        if dia_hoy not in matrices:
            print(f"\n  No hay datos historicos para {dia_hoy}")
            return
        trans_prob, trans_total = matrices[dia_hoy]
        scores = {}
        if ultimo_num in trans_total and trans_total[ultimo_num] > 0:
            for n in range(38):
                p = trans_prob.get((ultimo_num, n), 0)
                if p > 0:
                    scores[n] = p
        if not scores:
            print(f"\n  No hay transiciones registradas para el numero {ultimo_num} en {dia_hoy}")
            return
        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        print(f"\n  {'#':<3} {'Num':<6} {'Animal':<16} {'Prob':<8}")
        print(f"  {'-'*35}")
        for i, n in enumerate(top, 1):
            nom = self.num_int_a_animal.get(n, '?')
            print(f"  {i:<3} {n:<6} {nom:<16} {scores[n]:<8.2f}%")
        print(f"\n{'='*70}\n")
        return top

    def evaluar_markov_dia_semana(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10: print("Pocos datos"); return
        matrices = self.get_matriz_markov_por_dia(df)
        hits, horas, dias = [], [], []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']: continue
            prev, act = df.iloc[i-1], df.iloc[i]
            num_real = act['Num_Int']
            dia = pd.to_datetime(act['Fecha']).strftime('%A')
            ul = int(prev['Num_Int'])
            hit = False
            if dia in matrices:
                tp_d, tt_d = matrices[dia]
                scores = {}
                if ul in tt_d and tt_d[ul] > 0:
                    for n in range(38):
                        p = tp_d.get((ul, n), 0)
                        if p > 0: scores[n] = p
                top = sorted(scores, key=scores.get, reverse=True)[:top_k]
                hit = num_real in top
            hits.append(hit)
            horas.append(act['Solo_hora'])
            dias.append(dia)
        self._print_model_stats(hits, horas, dias, 'Markov x Dia', top_k)
        return hits

    def _numeros_por_dia(self, df):
        """Retorna dict dia -> set de numeros que han aparecido en ese dia."""
        dfc = df.copy()
        dfc['Dia_Semana'] = pd.to_datetime(dfc['Fecha'].astype(str)).dt.day_name()
        return {d: set(g['Num_Int'].unique()) for d, g in dfc.groupby('Dia_Semana')}

    def prediccion_gxhora_filtro_dia(self, datos, top_k=38):
        """Predice con GxHora filtrando solo numeros que han salido en el dia actual."""
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        nums_por_dia = self._numeros_por_dia(df)
        ultimo = df.iloc[-1]
        dia = pd.to_datetime(ultimo['Fecha']).strftime('%A')
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        candidatos = nums_por_dia.get(dia, set(range(38)))
        print(f"\n{'='*70}")
        print(f"  PREDICCION GxHora + FILTRO POR DIA")
        print(f"  Dia: {DIA_NOMBRE.get(dia, dia)} | Candidatos: {len(candidatos)}/38")
        print(f"  Ultimo: {ultimo['Solo_hora']} - {ultimo['Animal']} (#{ultimo['Num_Int']})")
        print(f"{'='*70}")
        scores = {}
        for n in candidatos:
            scores[n] = 0
        ultimo_num = int(ultimo['Num_Int'])
        hp = ultimo['Hora']
        hn = None
        for o, d in self.get_parejas_horarias():
            if o == ultimo['Solo_hora']:
                hn = d
                break
        if hn is None:
            print("\n  No hay hora siguiente definida")
            return
        pareja_h = (hp, hn)
        mk_scores = {}
        if ultimo_num in trans_total and trans_total[ultimo_num] > 0:
            for n in range(38):
                p = trans_prob.get((ultimo_num, n), 0)
                if p > 0:
                    mk_scores[n] = p
        mh_scores = {}
        if pareja_h in trans_h and ultimo_num in total_h.get(pareja_h, {}):
            tot = total_h[pareja_h][ultimo_num]
            if tot > 0:
                for n2, cnt in trans_h[pareja_h][ultimo_num].items():
                    mh_scores[n2] = cnt / tot * 100
        if len(mh_scores) < top_k and ultimo_num in trans_total and trans_total[ultimo_num] > 0:
            candidates = [(n2, trans_prob.get((ultimo_num, n2), 0)) for n2 in range(38)
                          if n2 not in mh_scores and trans_prob.get((ultimo_num, n2), 0) > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)
            for n2, p in candidates:
                mh_scores[n2] = p
                if len(mh_scores) >= top_k:
                    break
        if len(mh_scores) < top_k:
            gf = df['Num_Int'].value_counts(normalize=True).mul(100)
            for n2, pct in gf.items():
                if n2 not in mh_scores:
                    mh_scores[n2] = pct
                    if len(mh_scores) >= top_k:
                        break
        m_hh = total_h.get(pareja_h, {}).get(ultimo_num, 0)
        w_hh = min(0.9, max(0.1, m_hh / 50))
        todos = set(mk_scores) | set(mh_scores)
        gxh = {}
        for n2 in todos:
            gxh[n2] = mk_scores.get(n2, 0) * (1 - w_hh) + mh_scores.get(n2, 0) * w_hh
        top = sorted([(n, gxh.get(n, 0)) for n in candidatos if n in gxh], key=lambda x: x[1], reverse=True)[:top_k]
        print(f"\n  {'#':<3} {'Num':<6} {'Animal':<16} {'Prob':<8}")
        print(f"  {'-'*35}")
        if not top:
            print("  (sin datos para esta combinacion)")
        for i, (n, p) in enumerate(top, 1):
            nom = self.num_int_a_animal.get(n, '?')
            print(f"  {i:<3} {n:<6} {nom:<16} {p:<8.2f}%")
        print(f"\n  Nota: se filtraron {38 - len(candidatos)} numeros que nunca salieron en {DIA_NOMBRE.get(dia, dia)}")
        print(f"\n{'='*70}\n")
        return top

    def evaluar_gxhora_filtro_dia(self, datos, top_k=38):
        """Backtesting: compara GxHora normal vs GxHora con filtro por dia."""
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        nums_por_dia = self._numeros_por_dia(df)
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        res_gxh = []
        res_filt = []
        res_dias = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev = df.iloc[i-1]
            act = df.iloc[i]
            num_real = act['Num_Int']
            dia = act['Dia_Semana']
            gxh_hit = self._eval_gxhora_hit(prev, act, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            candidatos = nums_por_dia.get(dia, set(range(38)))
            res_gxh.append(gxh_hit)
            res_filt.append(gxh_hit if num_real in candidatos else False)
            res_dias.append(dia)
        print(f"\n{'='*70}")
        print(f"  EVALUACION GxHora + FILTRO POR DIA (Top-{top_k})")
        print(f"  Basado en {len(res_gxh)} transiciones")
        print(f"{'='*70}")
        print(f"\n  {'Modelo':<25} {'% Acierto':<12} {'Candidatos':<12}")
        print(f"  {'-'*49}")
        print(f"  {'GxHora normal':<25} {np.mean(res_gxh)*100:<12.2f}% {38:<12}")
        prom_filt = np.mean(res_filt)
        prom_cands = np.mean([len(nums_por_dia.get(d, set(range(38)))) for d in res_dias])
        print(f"  {'GxHora + filtro dia':<25} {prom_filt*100:<12.2f}% {prom_cands:<12.0f}")
        print(f"\n  Por dia:")
        for dia_en in DIA_NOMBRE:
            d_res = [res_filt[j] for j in range(len(res_filt)) if res_dias[j] == dia_en]
            if not d_res:
                continue
            print(f"    {DIA_NOMBRE[dia_en]:<12} {np.mean(d_res)*100:.1f}% ({len(d_res)} muestras)")
        diff = prom_filt - np.mean(res_gxh)
        print(f"\n  Diferencia con GxHora normal: {diff*100:+.2f}%")
        print(f"  Candidatos promedio: {prom_cands:.0f}/38 ({100-prom_cands/38*100:.0f}% reduccion)")
        print(f"\n{'='*70}\n")
        return {'gxh': np.mean(res_gxh), 'filt': prom_filt}

    def prediccion_markov_hora_dia(self, datos, top_k=38):
        """Predice combinando Markov x Hora + dia de semana."""
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h_g, total_h_g = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        ultimo = df.iloc[-1]
        dia = pd.to_datetime(ultimo['Fecha']).strftime('%A')
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        sub = df[pd.to_datetime(df['Fecha'].astype(str)).dt.strftime('%A') == dia]
        mh_scores = {}
        if len(sub) >= 5:
            sub_prep = self.preparar_datos_markov(sub)
            _, _, trans_h_d, total_h_d = self.construir_matrices_markov(sub_prep, incluir_trasnocho=False)
            hp = ultimo['Hora']
            hn = None
            for o, d in self.get_parejas_horarias():
                if o == ultimo['Solo_hora']:
                    hn = d
                    break
            if hn:
                pareja = (hp, hn)
                ul = int(ultimo['Num_Int'])
                if pareja in trans_h_d and ul in total_h_d.get(pareja, {}):
                    tot = total_h_d[pareja][ul]
                    if tot > 0:
                        for n2, cnt in trans_h_d[pareja][ul].items():
                            mh_scores[n2] = cnt / tot * 100
        print(f"\n{'='*70}")
        print(f"  PREDICCION MkHora + DIA")
        print(f"  Dia: {DIA_NOMBRE.get(dia, dia)} | Muestras del dia: {len(sub)}")
        print(f"  Ultimo: {ultimo['Solo_hora']} - {ultimo['Animal']} (#{ultimo['Num_Int']})")
        print(f"{'='*70}")
        if mh_scores:
            top = sorted(mh_scores, key=mh_scores.get, reverse=True)[:top_k]
            print(f"\n  {'#':<3} {'Num':<6} {'Animal':<16} {'Prob':<8}")
            print(f"  {'-'*35}")
            for i, n in enumerate(top, 1):
                nom = self.num_int_a_animal.get(n, '?')
                print(f"  {i:<3} {n:<6} {nom:<16} {mh_scores[n]:<8.2f}%")
        else:
            print(f"\n  Sin datos de MkHora para este dia. Usando MkHora global...")
            scores = {}
            hp = ultimo['Hora']
            hn = None
            for o, d in self.get_parejas_horarias():
                if o == ultimo['Solo_hora']:
                    hn = d
                    break
            if hn:
                pareja = (hp, hn)
                ul = int(ultimo['Num_Int'])
                if pareja in trans_h_g and ul in total_h_g.get(pareja, {}):
                    tot = total_h_g[pareja][ul]
                    if tot > 0:
                        for n2, cnt in trans_h_g[pareja][ul].items():
                            scores[n2] = cnt / tot * 100
                if len(scores) < top_k and ul in trans_total and trans_total[ul] > 0:
                    candidates = [(n2, trans_prob.get((ul, n2), 0)) for n2 in range(38)
                                  if n2 not in scores and trans_prob.get((ul, n2), 0) > 0]
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    for n2, p in candidates:
                        scores[n2] = p
                        if len(scores) >= top_k:
                            break
                if len(scores) < top_k:
                    gf = df['Num_Int'].value_counts(normalize=True).mul(100)
                    for n2, pct in gf.items():
                        if n2 not in scores:
                            scores[n2] = pct
                            if len(scores) >= top_k:
                                break
                top = sorted(scores, key=scores.get, reverse=True)[:top_k]
                print(f"\n  {'#':<3} {'Num':<6} {'Animal':<16} {'Prob':<8}")
                print(f"  {'-'*35}")
                for i, n in enumerate(top, 1):
                    nom = self.num_int_a_animal.get(n, '?')
                    print(f"  {i:<3} {n:<6} {nom:<16} {scores[n]:<8.2f}%")
        print(f"\n{'='*70}\n")

    def evaluar_markov_hora_dia(self, datos, top_k=38):
        """Backtesting: compara MkHora normal vs MkHora+Dia."""
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h_g, total_h_g = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        global_freq = df['Num_Int'].value_counts(normalize=True).mul(100)
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        matrices_dia = {}
        for dia in df['Dia_Semana'].unique():
            sub = df[df['Dia_Semana'] == dia]
            if len(sub) >= 5:
                sub_prep = self.preparar_datos_markov(sub)
                _, _, th, toh = self.construir_matrices_markov(sub_prep, incluir_trasnocho=False)
                matrices_dia[dia] = (th, toh)
        res_mkh = []
        res_mkhd = []
        res_dias = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev = df.iloc[i-1]
            act = df.iloc[i]
            num_real = act['Num_Int']
            dia = act['Dia_Semana']
            hp = prev['Hora']
            hn = act['Hora']
            pareja = (hp, hn)
            ul = int(prev['Num_Int'])
            mkh_hit = self._add_markov_hora_hit(df, i, prev, act, num_real, trans_h_g, total_h_g, trans_prob, trans_total)
            res_mkh.append(mkh_hit)
            res_dias.append(dia)
            mkhd_hit = False
            if dia in matrices_dia:
                th_d, toh_d = matrices_dia[dia]
                scores = {}
                if pareja in th_d and ul in toh_d.get(pareja, {}):
                    tot = toh_d[pareja][ul]
                    if tot > 0:
                        for n2, cnt in th_d[pareja][ul].items():
                            scores[n2] = cnt / tot * 100
                if len(scores) < top_k and ul in trans_total and trans_total[ul] > 0:
                    candidates = [(n2, trans_prob.get((ul, n2), 0)) for n2 in range(38)
                                  if n2 not in scores and trans_prob.get((ul, n2), 0) > 0]
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    for n2, p in candidates:
                        scores[n2] = p
                        if len(scores) >= top_k:
                            break
                if len(scores) < top_k:
                    for n2, pct in global_freq.items():
                        if n2 not in scores:
                            scores[n2] = pct
                            if len(scores) >= top_k:
                                break
                top_mkhd = sorted(scores, key=scores.get, reverse=True)[:top_k]
                mkhd_hit = num_real in top_mkhd
            res_mkhd.append(mkhd_hit)
        print(f"\n{'='*70}")
        print(f"  EVALUACION MkHora + DIA (Top-{top_k})")
        print(f"  Basado en {len(res_mkh)} transiciones")
        print(f"{'='*70}")
        print(f"\n  {'Modelo':<20} {'% Acierto':<12}")
        print(f"  {'-'*32}")
        print(f"  {'MkHora normal':<20} {np.mean(res_mkh)*100:<12.2f}%")
        print(f"  {'MkHora + Dia':<20} {np.mean(res_mkhd)*100:<12.2f}%")
        diff = np.mean(res_mkhd) - np.mean(res_mkh)
        print(f"  {'Diferencia':<20} {diff*100:<+12.2f}%")
        print(f"\n  Por dia:")
        for dia_en in DIA_NOMBRE:
            d_res_mkh = [res_mkh[j] for j in range(len(res_mkh)) if res_dias[j] == dia_en]
            d_res_mkhd = [res_mkhd[j] for j in range(len(res_mkhd)) if res_dias[j] == dia_en]
            if not d_res_mkh:
                continue
            print(f"    {DIA_NOMBRE[dia_en]:<12} MkHora={np.mean(d_res_mkh)*100:.1f}%  MkHora+Dia={np.mean(d_res_mkhd)*100:.1f}% ({len(d_res_mkh)} muestras)")
        print(f"\n{'='*70}\n")
        return {'mkh': np.mean(res_mkh), 'mkhd': np.mean(res_mkhd)}

    def _print_model_stats(self, hits, horas, dias, nombre, top_k):
        """Helper: imprime formato compacto de evaluacion con mejor/peor."""
        print(f"\n{'='*70}")
        print(f"  EVALUACION: {nombre} (Top-{top_k})")
        print(f"  Basado en {len(hits)} transiciones")
        print(f"{'='*70}")
        print(f"\n  {'Global:':<12} {np.mean(hits)*100:<.2f}%")
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        HORA_ORDER = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                      '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']
        print(f"\n  Por hora:")
        print(f"  {'Hora':<12} {'% Acierto':<12} {'Muestras':<8}")
        print(f"  {'-'*32}")
        pcts_h = {}
        for h in HORA_ORDER:
            sub = [hits[j] for j in range(len(hits)) if horas[j] == h]
            if not sub: continue
            pct = np.mean(sub) * 100
            pcts_h[h] = pct
            print(f"  {h:<12} {pct:<12.1f}% {len(sub):<8}")
        mejor_h = max(pcts_h, key=pcts_h.get) if pcts_h else ''
        peor_h = min(pcts_h, key=pcts_h.get) if pcts_h else ''
        print(f"\n  Mejor: {mejor_h} ({pcts_h.get(mejor_h, 0):.1f}%)")
        print(f"  Peor:  {peor_h} ({pcts_h.get(peor_h, 0):.1f}%)")
        print(f"\n  Por dia:")
        print(f"  {'Dia':<12} {'% Acierto':<12} {'Muestras':<8}")
        print(f"  {'-'*32}")
        pcts_d = {}
        for d in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
            sub = [hits[j] for j in range(len(hits)) if dias[j] == d]
            if not sub: continue
            pct = np.mean(sub) * 100
            pcts_d[d] = pct
            print(f"  {DIA_NOMBRE[d]:<12} {pct:<12.1f}% {len(sub):<8}")
        mejor_d = max(pcts_d, key=pcts_d.get) if pcts_d else ''
        peor_d = min(pcts_d, key=pcts_d.get) if pcts_d else ''
        print(f"\n  Mejor: {DIA_NOMBRE.get(mejor_d, mejor_d)} ({pcts_d.get(mejor_d, 0):.1f}%)")
        print(f"  Peor:  {DIA_NOMBRE.get(peor_d, peor_d)} ({pcts_d.get(peor_d, 0):.1f}%)")
        riesgo_h = [f"{h}({pcts_h[h]:.1f}%)" for h in sorted(pcts_h, key=pcts_h.get) if pcts_h[h] < 75]
        riesgo_d = [f"{DIA_NOMBRE[d]}({pcts_d[d]:.1f}%)" for d in sorted(pcts_d, key=pcts_d.get) if pcts_d[d] < 75]
        if riesgo_h:
            print(f"\n  Horas de riesgo (< 75%): {'; '.join(riesgo_h)}")
        if riesgo_d:
            print(f"  Dias de riesgo (< 75%): {'; '.join(riesgo_d)}")
        print(f"\n{'='*70}\n")

    def evaluar_markov_global(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10: print("Pocos datos"); return
        trans_prob, trans_total = self._transiciones_markov(df)
        hits, horas, dias = [], [], []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']: continue
            prev, act = df.iloc[i-1], df.iloc[i]
            num_real = act['Num_Int']
            ul = int(prev['Num_Int'])
            scores = {}
            if ul in trans_total and trans_total[ul] > 0:
                for n in range(38):
                    p = trans_prob.get((ul, n), 0)
                    if p > 0: scores[n] = p
            top = sorted(scores, key=scores.get, reverse=True)[:top_k]
            hits.append(num_real in top)
            horas.append(act['Solo_hora'])
            dias.append(pd.to_datetime(act['Fecha']).strftime('%A'))
        self._print_model_stats(hits, horas, dias, 'Markov Global', top_k)
        return hits

    def evaluar_frecuencia_hora(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10: print("Pocos datos"); return
        freq_hora = self._frecuencias_hora(df, 'Solo_hora')
        hits, horas, dias = [], [], []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']: continue
            act = df.iloc[i]
            num_real = act['Num_Int']
            hora_real = act['Solo_hora']
            scores = {}
            if hora_real in freq_hora:
                for n, p in freq_hora[hora_real].items():
                    scores[n] = p
            top = sorted(scores, key=scores.get, reverse=True)[:top_k]
            hits.append(num_real in top)
            horas.append(hora_real)
            dias.append(pd.to_datetime(act['Fecha']).strftime('%A'))
        self._print_model_stats(hits, horas, dias, 'Frecuencia x Hora', top_k)
        return hits

    def evaluar_markov_hora(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10: print("Pocos datos"); return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        hits, horas, dias = [], [], []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']: continue
            prev, act = df.iloc[i-1], df.iloc[i]
            num_real = act['Num_Int']
            hit = self._add_markov_hora_hit(df, i, prev, act, num_real, trans_h, total_h, trans_prob, trans_total)
            hits.append(hit)
            horas.append(act['Solo_hora'])
            dias.append(pd.to_datetime(act['Fecha']).strftime('%A'))
        self._print_model_stats(hits, horas, dias, 'Markov x Hora (MkHora)', top_k)
        return hits

    def evaluar_gxhora(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10: print("Pocos datos"); return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        hits, horas, dias = [], [], []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']: continue
            prev, act = df.iloc[i-1], df.iloc[i]
            num_real = act['Num_Int']
            hit = self._eval_gxhora_hit(prev, act, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            hits.append(hit)
            horas.append(act['Solo_hora'])
            dias.append(pd.to_datetime(act['Fecha']).strftime('%A'))
        self._print_model_stats(hits, horas, dias, 'GxHora', top_k)
        return hits

    def evaluar_top5_completo(self, datos, top_k=38):
        """Tabla detallada con los 5 modelos (sin ML)."""
        df = datos.copy()
        if len(df) < 10: print("Pocos datos"); return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        freq_hora = self._frecuencias_hora(df, 'Solo_hora')
        matrices = self.get_matriz_markov_por_dia(df)
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        rows = []
        resumen = { 'M':[], 'MkH':[], 'GxH':[], 'H':[], 'MxD':[] }
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']: continue
            prev, act = df.iloc[i-1], df.iloc[i]
            num_real = int(act['Num_Int'])
            fecha, hora_r = act['Fecha'], act['Solo_hora']
            ul = int(prev['Num_Int'])
            dia = pd.to_datetime(fecha).strftime('%A')

            # Markov global ranks
            mk_scores = {}
            if ul in trans_total and trans_total[ul] > 0:
                for n in range(38):
                    p = trans_prob.get((ul, n), 0)
                    if p > 0: mk_scores[n] = p
            mk_full = sorted(mk_scores, key=mk_scores.get, reverse=True)
            mk_top = mk_full[:top_k]
            rk_m = (mk_top.index(num_real) + 1) if num_real in mk_top else '-'

            # Frecuencia x Hora
            h_scores = {}
            if hora_r in freq_hora:
                for n, p in freq_hora[hora_r].items():
                    h_scores[n] = p
            h_full = sorted(h_scores, key=h_scores.get, reverse=True)
            h_top = h_full[:top_k]
            rk_h = (h_top.index(num_real) + 1) if num_real in h_top else '-'

            # MkHora
            hp, hn = prev['Hora'], act['Hora']
            pareja_h = (hp, hn)
            mh_scores = {}
            if pareja_h in trans_h and ul in total_h.get(pareja_h, {}):
                tot = total_h[pareja_h][ul]
                if tot > 0:
                    for n2, cnt in trans_h[pareja_h][ul].items():
                        mh_scores[n2] = cnt / tot * 100
            if len(mh_scores) < top_k and ul in trans_total and trans_total[ul] > 0:
                candidates = [(n2, trans_prob.get((ul, n2), 0)) for n2 in range(38)
                              if n2 not in mh_scores and trans_prob.get((ul, n2), 0) > 0]
                candidates.sort(key=lambda x: x[1], reverse=True)
                for n2, p in candidates:
                    mh_scores[n2] = p
                    if len(mh_scores) >= top_k: break
            if len(mh_scores) < top_k:
                gf = df['Num_Int'].value_counts(normalize=True).mul(100)
                for n2, pct in gf.items():
                    if n2 not in mh_scores:
                        mh_scores[n2] = pct
                        if len(mh_scores) >= top_k: break
            mh_full = sorted(mh_scores, key=mh_scores.get, reverse=True)
            mh_top = mh_full[:top_k]
            rk_mkh = (mh_top.index(num_real) + 1) if num_real in mh_top else '-'

            # GxHora
            todos = set(mk_scores) | set(mh_scores)
            gxh_scores = {}
            m_hh = total_h.get(pareja_h, {}).get(ul, 0)
            w_hh = min(0.9, max(0.1, m_hh / 50))
            for n2 in todos:
                gxh_scores[n2] = mk_scores.get(n2, 0) * (1 - w_hh) + mh_scores.get(n2, 0) * w_hh
            gxh_full = sorted(gxh_scores, key=gxh_scores.get, reverse=True)
            gxh_top = gxh_full[:top_k]
            rk_gxh = (gxh_top.index(num_real) + 1) if num_real in gxh_top else '-'

            # Markov x Dia
            rk_mxd = '-'
            if dia in matrices:
                tp_d, tt_d = matrices[dia]
                d_scores = {}
                if ul in tt_d and tt_d[ul] > 0:
                    for n in range(38):
                        p = tp_d.get((ul, n), 0)
                        if p > 0: d_scores[n] = p
                d_full = sorted(d_scores, key=d_scores.get, reverse=True)
                d_top = d_full[:top_k]
                rk_mxd = (d_top.index(num_real) + 1) if num_real in d_top else '-'

            predicho = mk_top[0] if mk_top else '?'
            pred_str = f"{predicho}({self.num_int_a_animal.get(predicho, '?'):<12})" if isinstance(predicho, int) else f"{'?':>2}({'?':<12})"
            real_str = f"{num_real:>2}({self.num_int_a_animal.get(num_real, '?'):<12})"
            any_hit = (rk_m != '-' or rk_mkh != '-' or rk_gxh != '-' or rk_h != '-' or rk_mxd != '-')
            hit_str = "OK" if any_hit else "NO"
            rows.append((fecha, hora_r, pred_str, real_str, hit_str,
                        str(rk_m), str(rk_mkh), str(rk_gxh), str(rk_h), str(rk_mxd)))
            # Fill summary for each model
            resumen['M'].append(rk_m != '-')
            resumen['MkH'].append(rk_mkh != '-')
            resumen['GxH'].append(rk_gxh != '-')
            resumen['H'].append(rk_h != '-')
            resumen['MxD'].append(rk_mxd != '-')

        print(f"\n{'='*110}")
        print(f"  EVALUACION COMPLETA — 5 MODELOS (Top-{top_k})")
        print(f"  {len(rows)} transiciones analizadas")
        print(f"{'='*110}")
        header = f"{'Fecha':<12} {'Hora':<10} {'Predicho':<19} {'Real':<19} {'Hit':<5} {'Rk-M':<6} {'Rk-MkH':<8} {'Rk-GxH':<8} {'Rk-H':<6} {'Rk-MxD':<8}"
        print(f"\n{header}")
        print("-" * len(header))
        for r in rows:
            print(f"{r[0]:<12} {r[1]:<10} {r[2]:<19} {r[3]:<19} {r[4]:<5} {r[5]:<6} {r[6]:<8} {r[7]:<8} {r[8]:<6} {r[9]:<8}")
        print(f"\n{'='*110}")
        print(f"\n  PRECISION TOP-{top_k} por modelo:")
        print(f"  {'-'*40}")
        labels = [('Markov (M)', 'M'), ('Markov x Hora (MkH)', 'MkH'), ('GxHora (GxH)', 'GxH'),
                  ('Frec. Hora (H)', 'H'), ('Markov x Dia (MxD)', 'MxD')]
        for label, key in labels:
            total = len(resumen[key])
            aciertos = sum(resumen[key])
            if total > 0:
                print(f"  {label:<25} {aciertos:>4}/{total} = {aciertos/total*100:.1f}%")
        print(f"\n{'='*110}\n")
        return rows

    def analizar_aciertos_por_dia_semana(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        DIA_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        resultados = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            num_real = actual['Num_Int']
            hit = self._eval_gxhora_hit(prev_state, actual, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            resultados.append({'dia': df.iloc[i]['Dia_Semana'], 'gxh': hit})
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            print("No se generaron resultados")
            return
        print(f"\n{'='*60}")
        print(f"  ACIERTOS GxHora POR DIA DE LA SEMANA (Top-{top_k})")
        print(f"  Basado en {len(df_res)} sorteos analizados")
        print(f"{'='*60}")
        print(f"\n{'Dia':<12} {'% Acierto':<12} {'Sorteos':<8}")
        print(f"{'-'*32}")
        for dia_en in DIA_ORDER:
            sub = df_res[df_res['dia'] == dia_en]
            if len(sub) == 0:
                continue
            pct = sub['gxh'].mean() * 100
            print(f"{DIA_NOMBRE[dia_en]:<12} {pct:<12.1f}% {len(sub):<8}")
        print(f"\n  Global: {df_res['gxh'].mean()*100:.1f}%")
        best_dia = df_res.groupby('dia')['gxh'].mean().idxmax()
        best_val = df_res.groupby('dia')['gxh'].mean().max() * 100
        print(f"  Mejor dia: {DIA_NOMBRE[best_dia]} ({best_val:.1f}%)")
        print(f"\n{'='*60}\n")
        return df_res

    def analizar_aciertos_por_hora(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        HORA_ORDER = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                      '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']
        resultados = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            num_real = actual['Num_Int']
            hora_real = actual['Solo_hora']
            hit = self._eval_gxhora_hit(prev_state, actual, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            resultados.append({'hora': hora_real, 'gxh': hit})
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            print("No se generaron resultados")
            return
        print(f"\n{'='*60}")
        print(f"  ACIERTOS GxHora POR HORA (Top-{top_k})")
        print(f"  Basado en {len(df_res)} sorteos analizados")
        print(f"{'='*60}")
        print(f"\n{'Hora':<12} {'% Acierto':<12} {'Sorteos':<8}")
        print(f"{'-'*32}")
        for h in HORA_ORDER:
            sub = df_res[df_res['hora'] == h]
            if len(sub) == 0:
                continue
            pct = sub['gxh'].mean() * 100
            print(f"{h:<12} {pct:<12.1f}% {len(sub):<8}")
        print(f"\n  Global: {df_res['gxh'].mean()*100:.1f}%")
        best_h = df_res.groupby('hora')['gxh'].mean().idxmax()
        best_val = df_res.groupby('hora')['gxh'].mean().max() * 100
        print(f"  Mejor hora: {best_h} ({best_val:.1f}%)")
        print(f"\n{'='*60}\n")
        return df_res

    def analizar_fallos_por_dia_semana(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        DIA_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        resultados = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            num_real = actual['Num_Int']
            fallo = not self._eval_gxhora_hit(prev_state, actual, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            resultados.append({'dia': df.iloc[i]['Dia_Semana'], 'gxh': fallo})
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            print("No se generaron resultados")
            return
        print(f"\n{'='*60}")
        print(f"  FALLOS GxHora POR DIA DE LA SEMANA (Top-{top_k})")
        print(f"  Basado en {len(df_res)} sorteos analizados")
        print(f"{'='*60}")
        print(f"\n{'Dia':<12} {'% Fallo':<12} {'Sorteos':<8}")
        print(f"{'-'*32}")
        for dia_en in DIA_ORDER:
            sub = df_res[df_res['dia'] == dia_en]
            if len(sub) == 0:
                continue
            pct = sub['gxh'].mean() * 100
            print(f"{DIA_NOMBRE[dia_en]:<12} {pct:<12.1f}% {len(sub):<8}")
        print(f"\n  Global: {df_res['gxh'].mean()*100:.1f}%")
        print(f"\n{'='*60}\n")
        return df_res

    def analizar_fallos_por_hora(self, datos, top_k=38):
        df = datos.copy()
        if len(df) < 10:
            print("Pocos datos")
            return
        trans_prob, trans_total = self._transiciones_markov(df)
        d_prep = self.preparar_datos_markov(df)
        _, _, trans_h, total_h = self.construir_matrices_markov(d_prep, incluir_trasnocho=False)
        HORA_ORDER = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                      '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']
        resultados = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            num_real = actual['Num_Int']
            hora_real = actual['Solo_hora']
            fallo = not self._eval_gxhora_hit(prev_state, actual, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            resultados.append({'hora': hora_real, 'gxh': fallo})
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            print("No se generaron resultados")
            return
        print(f"\n{'='*60}")
        print(f"  FALLOS GxHora POR HORA (Top-{top_k})")
        print(f"  Basado en {len(df_res)} sorteos analizados")
        print(f"{'='*60}")
        print(f"\n{'Hora':<12} {'% Fallo':<12} {'Sorteos':<8}")
        print(f"{'-'*32}")
        for h in HORA_ORDER:
            sub = df_res[df_res['hora'] == h]
            if len(sub) == 0:
                continue
            pct = sub['gxh'].mean() * 100
            print(f"{h:<12} {pct:<12.1f}% {len(sub):<8}")
        print(f"\n  Global: {df_res['gxh'].mean()*100:.1f}%")
        print(f"\n{'='*60}\n")
        return df_res
        df['Dia_Semana'] = pd.to_datetime(df['Fecha'].astype(str)).dt.day_name()
        DIA_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        DIA_NOMBRE = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miercoles','Thursday':'Jueves','Friday':'Viernes','Saturday':'Sabado','Sunday':'Domingo'}
        HORA_ORDER = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                      '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']
        matriz = {}
        totales = {}
        for dia_en in DIA_ORDER:
            matriz[dia_en] = {h: 0 for h in HORA_ORDER}
            totales[dia_en] = {h: 0 for h in HORA_ORDER}
        for i in range(1, len(df)):
            if df.iloc[i-1]['Fecha'] != df.iloc[i]['Fecha']:
                continue
            prev_state = df.iloc[i-1]
            actual = df.iloc[i]
            num_real = actual['Num_Int']
            hora_real = actual['Solo_hora']
            dia = df.iloc[i]['Dia_Semana']
            fallo = not self._eval_gxhora_hit(prev_state, actual, num_real, trans_prob, trans_total, trans_h, total_h, df, top_k=top_k)
            matriz[dia][hora_real] += (1 if fallo else 0)
            totales[dia][hora_real] += 1
        print(f"\n{'='*80}")
        print(f"  MATRIZ DE FALLOS GxHora POR DIA Y HORA (Top-{top_k})")
        print(f"{'='*80}")
        header = f"{'Dia/Hora':<14}" + "".join(f"{h[-8:]:<9}" for h in HORA_ORDER) + f"{'Total':<9}{'Prom':<9}"
        print(f"\n{header}")
        print("-" * len(header))
        todas_tasas = []
        for dia_en in DIA_ORDER:
            nombre = DIA_NOMBRE.get(dia_en, dia_en)
            row = f"{nombre:<14}"
            total_f = 0
            total_c = 0
            for h in HORA_ORDER:
                c = totales[dia_en][h]
                f = matriz[dia_en][h]
                total_f += f
                total_c += c
                tasa = (f / c * 100) if c > 0 else 0
                todas_tasas.append(tasa)
                if tasa >= 15:
                    row += f"{tasa:6.1f}%** "
                else:
                    row += f"{tasa:<9.1f}%"
            prom = (total_f / total_c * 100) if total_c > 0 else 0
            row += f"{total_f:<9}{prom:<9.1f}%"
            print(row)
        prom_gral = np.mean(todas_tasas) if todas_tasas else 0
        print(f"\n  Promedio general de fallo: {prom_gral:.1f}%")
        print(f"  ** = tasa >= 15% (zona de riesgo)")
        print(f"\n{'='*80}\n")
        return matriz

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
        print(f"\nNUMEROS MAS FRECUENTES POR HORA")
        for hora in sorted(df['Hora'].unique()):
            top3 = df[df['Hora'] == hora]['Num_Int'].value_counts().head(3)
            top3_str = ', '.join([f"{int(a):2d}({self.num_int_a_animal.get(int(a), '?')}) ({c})" for a, c in top3.items()])
            print(f"  {hora:<8} -> {top3_str}")
        ultimos_50 = df.tail(50)['Num_Int'].value_counts()
        frios = [n for n in range(38) if n not in ultimos_50.index]
        if frios:
            print(f"\nNUMEROS FRIOS (sin aparecer en ultimos 50 sorteos): {len(frios)}")
            frios_str = ', '.join([f"{n:2d}({self.num_int_a_animal.get(n, '?')})" for n in frios])
            print(f"  {frios_str}")
        else:
            print(f"\nNUMEROS FRIOS: Ninguno (todos han salido en ultimos 50)")
        ultimos_30 = df.tail(30)['Num_Int'].value_counts()
        esperado = 30 / 38
        calientes = ultimos_30[ultimos_30 > esperado * 1.5].head(10)
        if not calientes.empty:
            print(f"\nNUMEROS CALIENTES (ultimos 30 sorteos, +50% sobre esperado)")
            print(f"  Esperado: {esperado:.1f} apariciones por numero")
            for n, c in calientes.items():
                n = int(n)
                a = self.num_int_a_animal.get(n, "?")
                print(f"  {n:2d}({a:<10}) {c} apariciones ({c/esperado*100-100:.0f}% sobre lo esperado)")
        print(f"\nPARES DE NUMEROS QUE APARECEN EL MISMO DIA (TOP-10)")
        pares_dia = Counter()
        for fecha, grupo in df.groupby('Fecha'):
            nums_dia = sorted(grupo['Num_Int'].unique())
            for i in range(len(nums_dia)):
                for j in range(i+1, len(nums_dia)):
                    par = (int(nums_dia[i]), int(nums_dia[j]))
                    pares_dia[par] += 1
        for par, cnt in pares_dia.most_common(10):
            a1 = self.num_int_a_animal.get(par[0], "?")
            a2 = self.num_int_a_animal.get(par[1], "?")
            print(f"  {par[0]:2d}({a1:<10}) + {par[1]:2d}({a2:<10}) -> {cnt} dias")
        print(f"\n{'='*70}\n")

    def analizar_coocurrencias(self, datos, top_k=38):
        from collections import Counter
        from itertools import combinations
        df = datos.copy()
        print(f"\n{'='*70}")
        print(f"  CO-OCURRENCIAS DE NUMEROS EN EL MISMO DIA")
        print(f"{'='*70}")
        fechas_completas = df.groupby('Fecha').size()
        fechas_completas = fechas_completas[fechas_completas == 12]
        df_full = df[df['Fecha'].isin(fechas_completas.index)]
        n_dias = len(fechas_completas)
        print(f"  Basado en {n_dias} dias con 12 sorteos completos\n")

        pares_dia = Counter()
        tripletas_dia = Counter()
        for fecha, grupo in df_full.groupby('Fecha'):
            nums = sorted(grupo['Num_Int'].unique())
            for combo in combinations(nums, 2):
                pares_dia[combo] += 1
            for combo in combinations(nums, 3):
                tripletas_dia[combo] += 1

        print(f"  PARES MAS FRECUENTES (Top-{top_k})")
        print(f"  {'#':>3} {'Par':<20} {'Dias':>5} {'%':>6}")
        print(f"  {'-'*35}")
        for i, (par, cnt) in enumerate(pares_dia.most_common(top_k), 1):
            a1 = self.num_int_a_animal.get(int(par[0]), "?")
            a2 = self.num_int_a_animal.get(int(par[1]), "?")
            str_ = f"{int(par[0]):2d}({a1:.6s}) + {int(par[1]):2d}({a2:.6s})"
            print(f"  {i:3d} {str_:<20} {cnt:5d} {cnt/n_dias*100:5.1f}%")

        print(f"\n  TRIPLETAS MAS FRECUENTES (Top-{top_k})")
        print(f"  {'#':>3} {'Tripleta':<35} {'Dias':>5} {'%':>6}")
        print(f"  {'-'*46}")
        for i, (tri, cnt) in enumerate(tripletas_dia.most_common(top_k), 1):
            a1 = self.num_int_a_animal.get(int(tri[0]), "?")
            a2 = self.num_int_a_animal.get(int(tri[1]), "?")
            a3 = self.num_int_a_animal.get(int(tri[2]), "?")
            str_ = f"{int(tri[0]):2d}({a1:.6s}) + {int(tri[1]):2d}({a2:.6s}) + {int(tri[2]):2d}({a3:.6s})"
            print(f"  {i:3d} {str_:<35} {cnt:5d} {cnt/n_dias*100:5.1f}%")

        print(f"\n{'='*70}\n")

    def analizar_coocurrencias_por_rango(self, datos, top_k=38):
        from collections import Counter
        from itertools import combinations
        df = datos.copy()
        print(f"\n{'='*70}")
        print(f"  CO-OCURRENCIAS POR RANGO HORARIO")
        print(f"{'='*70}")

        def asignar_rango(hora):
            h = int(hora.split(':')[0])
            return 'MANANA' if h < 14 else 'TARDE'

        df['Rango'] = df['Hora'].apply(asignar_rango)
        fechas_completas = df.groupby('Fecha').size()
        fechas_completas = fechas_completas[fechas_completas == 12].index
        df_full = df[df['Fecha'].isin(fechas_completas)]
        n_dias = len(fechas_completas)
        print(f"  Basado en {n_dias} dias con 12 sorteos completos\n")

        for rango in ['MANANA', 'TARDE']:
            pares = Counter()
            for fecha, grupo in df_full[df_full['Rango'] == rango].groupby('Fecha'):
                nums = sorted(grupo['Num_Int'].unique())
                for combo in combinations(nums, 2):
                    pares[combo] += 1
            print(f"  --- {rango} (6 sorteos) ---")
            print(f"  {'#':>3} {'Par':<20} {'Dias':>5} {'%':>6}")
            print(f"  {'-'*35}")
            for i, (par, cnt) in enumerate(pares.most_common(top_k), 1):
                a1 = self.num_int_a_animal.get(int(par[0]), "?")
                a2 = self.num_int_a_animal.get(int(par[1]), "?")
                str_ = f"{int(par[0]):2d}({a1:.6s}) + {int(par[1]):2d}({a2:.6s})"
                print(f"  {i:3d} {str_:<20} {cnt:5d} {cnt/n_dias*100:5.1f}%")
            print()

        print(f"\n{'='*70}\n")

    def analizar_frecuencia_por_dia_semana(self, datos, top_k=15):
        from collections import Counter
        df = datos.copy()
        print(f"\n{'='*70}")
        print(f"  FRECUENCIA POR DIA DE LA SEMANA (Num_Int)")
        print(f"{'='*70}")
        df['DiaSemana'] = pd.to_datetime(df['Fecha']).dt.day_name()
        dias_es = {'Monday': 'LUNES', 'Tuesday': 'MARTES', 'Wednesday': 'MIERCOLES',
                   'Thursday': 'JUEVES', 'Friday': 'VIERNES', 'Saturday': 'SABADO',
                   'Sunday': 'DOMINGO'}
        df['DiaSemana'] = df['DiaSemana'].map(dias_es)
        total_global = Counter(df['Num_Int'])
        total_sorteos = len(df)
        prob_global = {n: c / total_sorteos for n, c in total_global.items()}
        for dia in ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'DOMINGO']:
            sub = df[df['DiaSemana'] == dia]
            if sub.empty:
                continue
            n_sorteos = len(sub)
            n_dias = sub['Fecha'].nunique()
            freq = sub['Num_Int'].value_counts().head(top_k)
            print(f"\n  --- {dia} ({n_dias} dias, {n_sorteos} sorteos) ---")
            print(f"  {'#':>3} {'Num':>4} {'Animal':<14} {'Veces':>5} {'% dia':>6} {'Lift':>7}")
            print(f"  {'-'*45}")
            for i, (n, cnt) in enumerate(freq.items(), 1):
                n = int(n)
                a = self.num_int_a_animal.get(n, "?")
                pct = cnt / n_sorteos * 100
                esperado = prob_global.get(n, 0) * n_sorteos
                lift = (cnt / esperado - 1) * 100 if esperado > 0 else 0
                lift_str = f"+{lift:.0f}%" if lift > 0 else f"{lift:.0f}%"
                print(f"  {i:3d} {n:4d} {a:<14} {cnt:5d} {pct:5.1f}% {lift_str:>7}")
        print(f"\n{'='*70}\n")

    def generar_matriz_probabilidad(self, datos):
        datos['Num_Int_Siguiente'] = datos['Num_Int'].shift(-1)
        datos['Solo_Fecha'] = datos['Timestamp'].dt.date
        datos['Es_Ultimo_Sorteo_del_Dia'] = datos.groupby('Solo_Fecha')['Timestamp'].transform('max') == datos['Timestamp']
        df_transiciones = datos[datos['Es_Ultimo_Sorteo_del_Dia'] == False].copy()
        matriz_conteo = pd.crosstab(df_transiciones['Num_Int'], df_transiciones['Num_Int_Siguiente'], normalize=False)
        matriz_probabilidad = matriz_conteo.div(matriz_conteo.sum(axis=1), axis=0) * 100
        return matriz_probabilidad.fillna(0)

    def matriz_probabilidad_transicion(self, datos):
        print("\nTOP-10 POR NUMERO (Matriz de Transicion Markov)")
        print("   Para cada Num_Int, los 10 mas probables que le siguen:\n")
        matriz = self.generar_matriz_probabilidad(datos.copy())
        for n in matriz.index:
            top10 = matriz.loc[n].sort_values(ascending=False).head(25)
            top10 = top10[top10 > 0]
            if top10.empty:
                continue
            animal_n = self.num_int_a_animal.get(int(n), "?")
            print(f"\n  {int(n):2d} ({animal_n:<14}) -> Top 10 siguientes:")
            for i, (n2, p) in enumerate(top10.items(), 1):
                a2 = self.num_int_a_animal.get(int(n2), "?")
                print(f"     {i:2d}. {int(n2):2d} ({a2:<14}) ({p:.1f}%)")

    def mejor_prediccion_siguiente(self, datos):
        print("\nPrediccion Siguiente en Tiempo Real (TOP-5 Markov)")
        try:
            entrada = input("Ingresa el animal o numero (0-37, ej: 27 o PERRO): ").strip()
            animal = self._resolver_animal(entrada)
            num_actual = self.animal_a_num_int.get(animal)
        except ValueError as e:
            print(f"Error: {e}")
            return
        matriz_probabilidad = self.generar_matriz_probabilidad(datos.copy())
        if num_actual not in matriz_probabilidad.index:
            print(f"Error: El numero '{num_actual}' no se encontro en el historial de transiciones.")
            return
        fila_prediccion = matriz_probabilidad.loc[num_actual].sort_values(ascending=False)
        top_5 = fila_prediccion.head(5)
        animal_n = self.num_int_a_animal.get(num_actual, "?")
        print(f"\nResultado de la Prediccion (TOP 5)")
        print(f"Si acaba de salir {num_actual:2d} ({animal_n}), los 5 mas probables son:")
        resultados = []
        for n2, prob in top_5.items():
            a2 = self.num_int_a_animal.get(int(n2), "?")
            resultados.append({'Num_Int': int(n2), 'Animal': a2, 'Probabilidad (%)': f"{prob:.2f}"})
        df_resultados = pd.DataFrame(resultados)
        print(df_resultados.to_string(index=False))
        mejor_num = int(top_5.index[0])
        mejor_animal = self.num_int_a_animal.get(mejor_num, "?")
        probabilidad_max = top_5.iloc[0]
        print(f"\nMaxima probabilidad individual: {mejor_num:2d} ({mejor_animal}) ({probabilidad_max:.2f}%)")

    def probabilidad_maxima_por_hora(self, datos):
        print("\nAnalisis de Frecuencia Historica por Hora (TOP-10)")
        frecuencia_completa = datos.groupby('Solo_hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        total_sorteos_por_hora = datos.groupby('Solo_hora').size().reset_index(name='Total_Sorteos')
        horas_unicas = frecuencia_completa['Solo_hora'].unique()
        print("\nTop 10 Numeros por Cada Hora")
        for hora in sorted(horas_unicas):
            df_hora = frecuencia_completa[frecuencia_completa['Solo_hora'] == hora].copy()
            total_sorteos = total_sorteos_por_hora[total_sorteos_por_hora['Solo_hora'] == hora]['Total_Sorteos'].iloc[0]
            top_10 = df_hora.sort_values(by='Probabilidad', ascending=False).head(25)
            print(f"\nHORA: {hora} (Total Sorteos: {total_sorteos})")
            rows = []
            for _, r in top_10.iterrows():
                a = self.num_int_a_animal.get(int(r['Num_Int']), "?")
                rows.append(f"{int(r['Num_Int']):2d}({a})")
            print('  ' + ', '.join(rows))

    def prediccion_markov_hora(self, datos):
        # Return structured prediction data (no printing) so callers can render as needed.
        return self.generar_prediccion_markov(datos)

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
            num_actual = df_prueba.iloc[i]['Num_Int']
            num_siguiente_real = df_prueba.iloc[i + 1]['Num_Int']
            if num_actual in matriz_entrenada.index:
                predicciones_totales += 1
                fila_prediccion = matriz_entrenada.loc[num_actual].sort_values(ascending=False)
                top_k_predichos = fila_prediccion.head(top_k).index.tolist()
                if num_siguiente_real in top_k_predichos:
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
                print("Resultado: El rendimiento esta cerca o por debajo del azar.")
        else:
            print("No se pudieron realizar predicciones validas en el conjunto de prueba.")

    def generar_prediccion_markov(self, datos, top_k=25):
        """Genera la predicción Markov+Hora en estructura de datos.

        Retorna un dict con claves:
          - 'top': lista de items {rank,num,animal,score,markov,hora_pct}
          - 'por_hora': OrderedDict hora->lista de {num,animal,score}
          - 'ultimo': {num,animal,hora}
        """
        df = datos.copy()
        trans_prob, trans_total = self._transiciones_markov(df)
        hora_freq = self._frecuencias_hora(df, 'Solo_hora')
        ultimo = df.iloc[-1]
        ultimo_num = int(ultimo['Num_Int']) if 'Num_Int' in ultimo else None
        ultimo_animal = self.num_int_a_animal.get(ultimo_num, "?")
        ultimo_hora = ultimo['Solo_hora'] if 'Solo_hora' in ultimo else None

        # prepare scored list
        scored = []
        for (prev_n, n), mp in trans_prob.items():
            if prev_n == ultimo_num:
                hp = hora_freq.get(ultimo_hora, {}).get(n, 0)
                scored.append((mp + hp, mp, hp, n))
        scored.sort(reverse=True)

        top = []
        for i, (sc, mp, hp, n) in enumerate(scored[:top_k], 1):
            a = self.num_int_a_animal.get(n, "?")
            top.append({
                'rank': i,
                'num': n,
                'animal': a,
                'score': round(sc, 1),
                'markov': round(mp, 1),
                'hora_pct': round(hp, 1),
            })

        # hours of interest (08:00 AM .. 07:00 PM)
        horas = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]

        por_hora = {}

        # helper to parse hour strings robustly into time objects
        def _parse_time(hstr):
            try:
                t = pd.to_datetime(hstr, format='%I:%M %p', errors='coerce')
                if pd.isna(t):
                    t = pd.to_datetime(hstr, format='%H:%M', errors='coerce')
                return None if pd.isna(t) else t.time()
            except Exception:
                return None

        ultimo_time = _parse_time(ultimo_hora) if ultimo_hora is not None else None

        for hora in horas:
            h_scored = []
            for _, _, _, n in scored[:top_k]:
                hp = hora_freq.get(hora, {}).get(n, 0)
                mp = trans_prob.get((ultimo_num, n), 0)
                h_scored.append((mp + hp, n))
            h_scored.sort(reverse=True)
            lista = []
            for s, n in h_scored[:38]:
                lista.append({'num': n, 'animal': self.num_int_a_animal.get(n, '?'), 'score': round(s, 1)})
            por_hora[hora] = lista

        return {'top': top, 'por_hora': por_hora, 'ultimo': {'num': ultimo_num, 'animal': ultimo_animal, 'hora': ultimo_hora}}

    def generar_prediccion_markov_segundo_orden(self, datos, top_k=25):
        """Markov de segundo orden: usa los últimos 2 números para predecir el siguiente."""
        if len(datos) < 3:
            print("Datos insuficientes (min 3 registros).")
            return
        ultimo = datos.iloc[-1]
        penultimo = datos.iloc[-2]
        num1 = int(penultimo['Num_Int'])
        num2 = int(ultimo['Num_Int'])
        a1 = self.num_int_a_animal.get(num1, "?")
        a2 = self.num_int_a_animal.get(num2, "?")
        items = self.get_matriz_segundo_orden(datos, num1, num2, top_k=38)
        if not items:
            print(f"\nNo hay transiciones registradas para el par ({num1}-{num2}).")
            return
        total_muestras = sum(cnt for _, _, cnt in items)
        print(f"\n{'='*60}")
        print(f"  PREDICCIÓN MARKOV DE SEGUNDO ORDEN")
        print(f"{'='*60}")
        print(f"  Últimos: {num1} ({a1}) → {num2} ({a2})")
        print(f"  Muestras totales para este par: {total_muestras}")
        print(f"{'─'*60}")
        print(f"  {'#':<5} {'Siguiente':<12} {'Animal':<16} {'Prob':<9} {'Muestras':<10}")
        print(f"  {'─'*50}")
        for i, (n, prob, cnt) in enumerate(items[:top_k], 1):
            a = self.num_int_a_animal.get(n, "?")
            print(f"  {i:<5} {n:<12} {a:<16} {prob:<8.1f}% {cnt:<5}/{total_muestras}")
        print(f"{'='*60}\n")
        return items[:top_k]

    # ── Análisis estadístico ──────────────────────────────────────────

    def _get_accuracy_top_k(self, datos, top_k=25, counts=None, totals=None):
        """Calcula accuracy de Markov Top-K en todo el dataset.
        Si se pasan counts/totals precomputados, los reusa (más rápido para permutaciones)."""
        if len(datos) < 10:
            return 0.0
        if counts is None or totals is None:
            trans_prob, trans_total = self._transiciones_markov(datos)
        else:
            trans_prob = {}
            for prev, followers in counts.items():
                for cur, cnt in followers.items():
                    trans_prob[(prev, cur)] = cnt / totals[prev] * 100
            trans_total = totals
        hits = 0
        total = 0
        for i in range(1, len(datos)):
            if datos.iloc[i-1]['Fecha'] == datos.iloc[i]['Fecha']:
                prev_num = int(datos.iloc[i-1]['Num_Int'])
                real_num = int(datos.iloc[i]['Num_Int'])
                if prev_num in trans_total:
                    scores = [(n, trans_prob.get((prev_num, n), 0)) for n in range(38)]
                    scores.sort(key=lambda x: -x[1])
                    top_k_nums = [n for n, _ in scores[:top_k]]
                    if real_num in top_k_nums:
                        hits += 1
                    total += 1
        return hits / total if total > 0 else 0.0

    def test_significancia_markov(self, datos, permutaciones=50, top_k=25):
        """Prueba de permutación: ¿el modelo Markov es significativamente mejor que azar?
        Baraja los números aleatoriamente y compara el accuracy real vs el de datos revueltos.
        Si p-valor < 0.05, el modelo tiene señal real.
        Ahora más rápido: precomputa transiciones y reusa en permutaciones.
        """
        import random as _random
        from collections import defaultdict
        print(f"\n{'='*70}")
        print("  TEST DE SIGNIFICANCIA ESTADÍSTICA (PERMUTACIÓN)")
        print(f"{'='*70}")
        print(f"  Dataset: {len(datos)} registros")
        print(f"  Top-K evaluado: {top_k}")
        print(f"  Permutaciones: {permutaciones}")
        print(f"{'─'*70}")

        # Precompute transition structure from original data
        counts = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        for i in range(1, len(datos)):
            if datos.iloc[i-1]['Fecha'] == datos.iloc[i]['Fecha']:
                prev = int(datos.iloc[i-1]['Num_Int'])
                cur = int(datos.iloc[i]['Num_Int'])
                counts[prev][cur] += 1
                totals[prev] += 1

        real_acc = self._get_accuracy_top_k(datos, top_k, counts, totals)
        aleatorio_esperado = top_k / 38
        print(f"\n  Accuracy real:           {real_acc:.4f} ({real_acc*100:.2f}%)")
        print(f"  Azar teórico (K/38):     {aleatorio_esperado:.4f} ({aleatorio_esperado*100:.2f}%)")

        nums_originales = datos['Num_Int'].values.copy()
        count_mejores = 0
        print(f"\n  Ejecutando {permutaciones} permutaciones...", end=' ', flush=True)
        for p in range(permutaciones):
            shuffled_nums = _random.sample(list(nums_originales), len(nums_originales))
            # Recompute counts with shuffled numbers (faster than full _transiciones_markov)
            shuf_counts = defaultdict(lambda: defaultdict(int))
            shuf_totals = defaultdict(int)
            for i in range(1, len(datos)):
                if datos.iloc[i-1]['Fecha'] == datos.iloc[i]['Fecha']:
                    prev = shuffled_nums[i-1]
                    cur = shuffled_nums[i]
                    shuf_counts[prev][cur] += 1
                    shuf_totals[prev] += 1
            acc = self._get_accuracy_top_k(datos, top_k, shuf_counts, shuf_totals)
            if acc >= real_acc:
                count_mejores += 1
            if (p + 1) % 10 == 0:
                print(f"{p+1}..", end='', flush=True)
        print(" hecho.")

        p_valor = count_mejores / permutaciones
        print(f"\n  Veces que el azar superó al real: {count_mejores}/{permutaciones}")
        print(f"  P-valor: {p_valor:.4f}")
        print(f"{'─'*70}")
        if p_valor < 0.01:
            print("  RESULTADO: *** Señal MUY significativa (p<0.01) ***")
        elif p_valor < 0.05:
            print("  RESULTADO: ** Señal significativa (p<0.05) **")
        elif p_valor < 0.10:
            print("  RESULTADO: * Señal marginal (p<0.10) *")
        else:
            print("  RESULTADO: Sin señal significativa (p>=0.10)")
            print("  El modelo NO es mejor que el azar.")
        print(f"{'='*70}\n")
        return {'real_acc': real_acc, 'p_valor': p_valor, 'permutaciones': permutaciones}

    def walk_forward_validation(self, datos, top_k=25):
        """Validación walk-forward: entrena con datos pasados, predice el siguiente, avanza una ventana.
        Optimizada con matriz incremental.
        """
        if len(datos) < 20:
            print("Datos insuficientes para walk-forward (min 20).")
            return
        print(f"\n{'='*70}")
        print("  VALIDACIÓN WALK-FORWARD (VENTANA MÓVIL)")
        print(f"{'='*70}")
        print(f"  Dataset: {len(datos)} registros")
        print(f"  Top-K: {top_k}")
        print(f"{'─'*70}")

        from collections import defaultdict
        # Build transition matrix incrementally
        counts = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        corte = int(len(datos) * 0.7)
        # Seed matrix with first `corte` records
        for i in range(1, corte):
            if datos.iloc[i-1]['Fecha'] == datos.iloc[i]['Fecha']:
                prev = int(datos.iloc[i-1]['Num_Int'])
                cur = int(datos.iloc[i]['Num_Int'])
                counts[prev][cur] += 1
                totals[prev] += 1

        hits = 0
        total = 0
        last_reported = 0
        for i in range(corte, len(datos) - 1):
            test_row = datos.iloc[i]
            next_real = int(datos.iloc[i + 1]['Num_Int'])
            # Skip cross-day transitions
            if test_row['Fecha'] != datos.iloc[i + 1]['Fecha']:
                # Still add transition to matrix for future predictions
                prev = int(datos.iloc[i-1]['Num_Int']) if i > 0 else int(test_row['Num_Int'])
                cur = int(test_row['Num_Int'])
                counts[prev][cur] += 1
                totals[prev] += 1
                continue

            prev_num = int(test_row['Num_Int'])
            if prev_num in totals:
                scores = [(n, counts[prev_num].get(n, 0) / totals[prev_num] * 100) for n in range(38)]
                scores.sort(key=lambda x: -x[1])
                top_k_nums = [n for n, _ in scores[:top_k]]
                if next_real in top_k_nums:
                    hits += 1
                total += 1

            # Add this transition to matrix for future steps
            prev = int(test_row['Num_Int'])
            cur = next_real
            counts[prev][cur] += 1
            totals[prev] += 1

            # Progress every 10% 
            pct = ((i - corte + 1) * 100) // (len(datos) - 1 - corte)
            if pct // 10 > last_reported // 10:
                print(f"  Progreso: {pct}%...", flush=True)
                last_reported = pct
        accuracy = hits / total if total > 0 else 0.0
        aleatorio = top_k / 38
        print(f"\n  Predicciones realizadas: {total}")
        print(f"  Aciertos (Top-{top_k}):    {hits}")
        print(f"  Accuracy walk-forward:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Azar teórico (K/38):     {aleatorio:.4f} ({aleatorio*100:.2f}%)")
        if accuracy > aleatorio:
            print(f"  Diferencia:              +{(accuracy-aleatorio)*100:.2f} puntos porcentuales")
        else:
            print(f"  Diferencia:              {(accuracy-aleatorio)*100:.2f} puntos porcentuales (por debajo del azar)")
        print(f"{'='*70}\n")
        return {'accuracy': accuracy, 'total': total, 'hits': hits}

    def vs_aleatorio(self, datos, top_k=25):
        """Comparación directa: accuracy del modelo real vs accuracy con datos revueltos."""
        print(f"\n{'='*70}")
        print("  COMPARACIÓN: MODELO REAL vs DATOS ALEATORIOS")
        print(f"{'='*70}")
        print(f"  Dataset: {len(datos)} registros")
        print(f"  Top-K: {top_k}")
        print(f"{'─'*70}")

        real_acc = self._get_accuracy_top_k(datos, top_k)

        import random as _random
        nums = datos['Num_Int'].values.copy()
        shuffled = datos.copy()
        shuffled['Num_Int'] = _random.sample(list(nums), len(nums))
        shuffled_acc = self._get_accuracy_top_k(shuffled, top_k)

        aleatorio = top_k / 38
        print(f"\n  {'Métrica':<30} {'Modelo Real':<18} {'Datos Revueltos':<18} {'Azor Teórico':<15}")
        print(f"  {'─'*78}")
        print(f"  {'Accuracy':<30} {real_acc:<18.4f} {shuffled_acc:<18.4f} {aleatorio:<15.4f}")
        print(f"  {'Porcentaje':<30} {real_acc*100:<18.2f}% {shuffled_acc*100:<18.2f}% {aleatorio*100:<15.2f}%")
        print()
        if real_acc > shuffled_acc:
            print(f"  ✓ El modelo real es mejor que datos revueltos")
            print(f"    Diferencia: +{(real_acc - shuffled_acc)*100:.2f} puntos")
        else:
            print(f"  ✗ El modelo real NO es mejor que datos revueltos")
            print(f"    (el ruido es igual o mayor que la supuesta señal)")
        print(f"{'='*70}\n")
        return {'real': real_acc, 'revuelto': shuffled_acc, 'aleatorio': aleatorio}

    def intervalos_confianza(self, datos, num_int, top_k=25):
        """Muestra para un número dado el Top-K de transiciones con intervalo de confianza binomial al 95%."""
        trans_prob, trans_total = self._transiciones_markov(datos)
        if num_int not in trans_total:
            print(f"\nEl número {num_int} no tiene transiciones registradas.")
            return

        total_trans = trans_total[num_int]
        print(f"\n{'='*70}")
        print(f"  INTERVALOS DE CONFIANZA (95%) — Desde número {num_int}")
        print(f"  Animal: {self.num_int_a_animal.get(num_int, '?')}")
        print(f"  Total transiciones desde este número: {total_trans}")
        print(f"{'='*70}")
        print(f"  {'#':<4} {'Siguiente':<12} {'Animal':<16} {'Prob':<9} {'IC 95%':<18} {'Muestras':<10}")
        print(f"  {'─'*65}")

        scores = [(n, trans_prob.get((num_int, n), 0)) for n in range(38)]
        scores.sort(key=lambda x: -x[1])

        for i, (n, prob) in enumerate(scores[:top_k], 1):
            # Count actual occurrences
            count = 0
            for j in range(len(datos) - 1):
                if (datos.iloc[j]['Num_Int'] == num_int and
                    datos.iloc[j]['Fecha'] == datos.iloc[j+1]['Fecha'] and
                    datos.iloc[j+1]['Num_Int'] == n):
                    count += 1
            # Wilson score interval (good for small n)
            # Using normal approximation for simplicity
            z = 1.96
            p_hat = count / total_trans if total_trans > 0 else 0
            se = np.sqrt(p_hat * (1 - p_hat) / total_trans) if total_trans > 0 else 0
            ci_low = max(0, (p_hat - z * se)) * 100
            ci_high = min(100, (p_hat + z * se)) * 100
            a = self.num_int_a_animal.get(n, "?")
            print(f"  {i:<4} {n:<12} {a:<16} {prob:<9.1f} [{ci_low:<7.1f}% - {ci_high:<5.1f}%]  {count:<3}/{total_trans}")

        print(f"{'='*70}\n")
        return scores[:top_k]

    def mostrar_transiciones_reales(self, datos, entrada, top_k=25):
        """Para un animal o número dado, muestra las transiciones reales más probables."""
        # Resolve entrada to numero
        try:
            num_int = int(entrada)
        except ValueError:
            entrada_up = entrada.strip().upper()
            from utils import ANIMAL_A_NUM_INT
            num_int = ANIMAL_A_NUM_INT.get(entrada_up)
            if num_int is None:
                print(f"Animal '{entrada}' no reconocido.")
                return

        trans_prob, trans_total = self._transiciones_markov(datos)
        if num_int not in trans_total:
            print(f"\nNo hay suficientes transiciones desde {entrada}.")
            return

        total_trans = trans_total[num_int]
        animal_origen = self.num_int_a_animal.get(num_int, "?")
        print(f"\n{'='*70}")
        print(f"  TRANSICIONES REALES — {entrada} ({animal_origen})")
        print(f"  Total de transiciones: {total_trans}")
        print(f"{'='*70}")
        print(f"  {'#':<4} {'Siguiente':<12} {'Animal':<16} {'Probabilidad':<14} {'Frecuencia':<10}")
        print(f"  {'─'*55}")

        scores = [(n, trans_prob.get((num_int, n), 0)) for n in range(38)]
        scores.sort(key=lambda x: -x[1])

        for i, (n, prob) in enumerate(scores[:top_k], 1):
            count = 0
            for j in range(len(datos) - 1):
                if (datos.iloc[j]['Num_Int'] == num_int and
                    datos.iloc[j]['Fecha'] == datos.iloc[j+1]['Fecha'] and
                    datos.iloc[j+1]['Num_Int'] == n):
                    count += 1
            a = self.num_int_a_animal.get(n, "?")
            bar = '█' * int(prob / 2) + '░' * (10 - int(prob / 2))
            print(f"  {i:<4} {n:<12} {a:<16} {prob:<6.1f}% {bar:<12} {count}/{total_trans}")

        print(f"\n  (K = {top_k})")
        print(f"{'='*70}\n")
        return scores[:top_k]

    def prediccion_por_hora_especifica(self, datos):
        print("\nPrediccion Historica por Hora Especifica")
        while True:
            hora_str = input("Ingresa la hora del sorteo (ej: 11:00 AM o 14:00): ").strip()
            try:
                hora_dt = pd.to_datetime(hora_str, format='%H:%M', errors='coerce')
                if pd.isna(hora_dt):
                    hora_dt = pd.to_datetime(hora_str, format='%I:%M %p', errors='coerce')
                if pd.isna(hora_dt):
                    raise ValueError
                solo_hora_buscada = hora_dt.strftime('%I:%M %p')
                break
            except ValueError:
                print("Error: Formato de hora invalido.")
        df_filtrado = datos[datos['Solo_hora'] == solo_hora_buscada].copy()
        if df_filtrado.empty:
            print(f"\nNo se encontraron datos historicos para la hora: {solo_hora_buscada}.")
            return
        frecuencia_num = df_filtrado['Num_Int'].value_counts().reset_index()
        frecuencia_num.columns = ['Num_Int', 'Conteo']
        total_sorteos_hora = len(df_filtrado)
        frecuencia_num['Probabilidad'] = (frecuencia_num['Conteo'] / total_sorteos_hora) * 100
        prediccion_maxima = frecuencia_num.iloc[0]
        mejor_animal = self.num_int_a_animal.get(int(prediccion_maxima['Num_Int']), "?")
        print(f"\nResultados para la hora: {solo_hora_buscada} (Historico)")
        print(f"Total de sorteos historicos analizados: {total_sorteos_hora}")
        print("-" * 50)
        print(f"Numero con mayor probabilidad: {int(prediccion_maxima['Num_Int']):2d} ({mejor_animal})")
        print(f"   Probabilidad: {prediccion_maxima['Probabilidad']:.2f}%")
        print(f"   Veces que salio: {prediccion_maxima['Conteo']}")
        print("-" * 50)
        print("\nTop 10 numeros en esta hora:")
        for _, r in frecuencia_num.head(25).iterrows():
            a = self.num_int_a_animal.get(int(r['Num_Int']), "?")
            print(f"   {int(r['Num_Int']):2d} ({a:<12}) - {r['Probabilidad']:.2f}%")

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
                print("Formato de hora invalido.")
        while True:
            try:
                num_str = input("Ingresa el Num_Int (0-37, ej: 27): ").strip()
                num_int = int(num_str)
                if num_int not in range(38):
                    raise ValueError
                animal = self.num_int_a_animal.get(num_int, "")
                if not animal:
                    print("Numero sin animal asignado")
                    continue
                numero_ext = int(datos_df[datos_df['Num_Int']==num_int]['Numero'].iloc[0]) if num_int in datos_df['Num_Int'].values else num_int
                break
            except ValueError:
                print("Numero invalido (0-37).")
        fecha_hoy = date.today()
        timestamp = pd.to_datetime(f"{fecha_hoy} {hora_final}")
        nueva_fila = pd.DataFrame([{
            'Fecha': fecha_hoy,
            'Hora': hora_final,
            'Animal': animal,
            'Numero': numero_ext,
            'Num_Int': num_int,
            'Timestamp': timestamp,
            'Solo_hora': solo_hora_final
        }])
        datos_df = pd.concat([datos_df, nueva_fila], ignore_index=True)
        try:
            datos_df[['Fecha', 'Hora', 'Animal', 'Numero']].to_excel(nombre_archivo, index=False)
            print(f"\nSorteo agregado: Num_Int={num_int} ({animal}), Hora={solo_hora_final}")
            return datos_df
        except Exception as e:
            print(f"\nError al guardar: {e}")
            return datos_df

    def evaluacion_estrategia_solo_manana(self, datos, hora_corte='13:00:00'):
        print(f"\nEVALUACION ESTRATEGIA SOLO MANANA (Hasta {hora_corte})")
        horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        datos_manana = datos[datos['Hora'].isin(horas_manana)].copy()
        frecuencia_manana = datos_manana.groupby('Hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map_manana = {}
        for hora_24h in frecuencia_manana['Hora'].unique():
            top_10_lista = frecuencia_manana[frecuencia_manana['Hora'] == hora_24h].head(25)['Num_Int'].tolist()
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
                num_salio = row['Num_Int']
                if hora_filtro in top_10_map_manana and num_salio in top_10_map_manana[hora_filtro]:
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
        print(datos[['Timestamp', 'Hora', 'Num_Int']].head(3) if 'Timestamp' in datos.columns else "No hay Timestamp")
        try:
            frecuencia_completa = datos.groupby('Hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
            print(f"Matriz de frecuencia calculada: {len(frecuencia_completa)} registros")
        except Exception as e:
            print(f"Error calculando frecuencia: {e}")
            return None
        top_10_map = {}
        horas_con_datos = frecuencia_completa['Hora'].unique()
        print(f"Horas con datos: {sorted(horas_con_datos)}")
        for hora_24h in horas_con_datos:
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(25)['Num_Int'].tolist()
            top_10_map[hora_24h] = top_10_lista
        print(f"Top-25 map creado: {len(top_10_map)} horas")
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
                num_salio = row['Num_Int']
                if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
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
                    num_salio = row['Num_Int']
                    if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
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
        columnas_criticas = ['Timestamp', 'Hora', 'Num_Int']
        for col in columnas_criticas:
            if col not in datos.columns:
                print(f"ERROR: Columna critica '{col}' no encontrada")
                return None
        print(f"Valores en columnas criticas:")
        print(f"   * Timestamp: {datos['Timestamp'].notna().sum()} valores no nulos")
        print(f"   * Hora: {datos['Hora'].notna().sum()} valores no nulos")
        print(f"   * Num_Int: {datos['Num_Int'].notna().sum()} valores no nulos")
        print(f"Primeras 3 filas REALES:")
        print(datos[['Timestamp', 'Hora', 'Num_Int']].head(3))
        print(f"Valores unicos en Hora: {sorted(datos['Hora'].unique())[:10]}")
        try:
            frecuencia_completa = datos.groupby('Hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
            print(f"Matriz de frecuencia: {len(frecuencia_completa)} registros")
            if len(frecuencia_completa) == 0:
                print("La matriz de frecuencia esta vacia - revisar formato de Hora y Num_Int")
                return None
            top_10_map = {}
            horas_con_datos = frecuencia_completa['Hora'].unique()
            print(f"Horas con datos en frecuencia: {sorted(horas_con_datos)}")
            for hora_24h in horas_con_datos:
                top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(25)['Num_Int'].tolist()
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
                    num_salio = int(row['Num_Int'])
                    if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
                        aciertos_manana += 1
                jugar_tarde = (aciertos_manana <= 1)
                aciertos_tarde = 0
                if jugar_tarde:
                    df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)]
                    for _, row in df_tarde.iterrows():
                        hora_filtro = row['Hora']
                        num_salio = int(row['Num_Int'])
                        if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
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
        frecuencia_completa = datos.groupby('Hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map = {}
        for hora_24h in frecuencia_completa['Hora'].unique():
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(25)['Num_Int'].tolist()
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
                    num_salio = int(df_hora['Num_Int'].iloc[0])
                    acierto = 1 if (hora in top_10_map and num_salio in top_10_map[hora]) else 0
                    aciertos_por_hora_manana[hora] = acierto
                    animales_por_hora_manana[hora] = num_salio
            aciertos_manana = sum(aciertos_por_hora_manana.values())
            jugar_tarde = (aciertos_manana <= 1)
            aciertos_tarde = 0
            if jugar_tarde:
                for _, row in df_dia[df_dia['Hora'].isin(HORAS_TARDE)].iterrows():
                    if row['Hora'] in top_10_map and row['Num_Int'] in top_10_map[row['Hora']]:
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
                'Num_8am': animales_por_hora_manana.get('08:00:00', -1),
                'Num_9am': animales_por_hora_manana.get('09:00:00', -1),
                'Num_10am': animales_por_hora_manana.get('10:00:00', -1),
                'Num_11am': animales_por_hora_manana.get('11:00:00', -1),
                'Num_12pm': animales_por_hora_manana.get('12:00:00', -1),
                'Num_1pm': animales_por_hora_manana.get('13:00:00', -1)
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
        columnas_mostrar = ['Fecha', 'Hora', 'Num_Int', 'Animal', 'Numero']
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
            registros_hoy = datos_hoy[['Hora', 'Num_Int', 'Animal', 'Numero']].sort_values('Hora')
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
        print(f"\nANALISIS DE NUMEROS RECIENTES:")
        print("="*40)
        ultimos_20 = datos_ordenados.head(20)
        numeros_recientes = ultimos_20['Num_Int'].unique()
        print(f"Numeros en ultimos 20 sorteos ({len(numeros_recientes)} unicos):")
        for i, n in enumerate(numeros_recientes, 1):
            n = int(n)
            a = self.num_int_a_animal.get(n, "?")
            print(f"   {i:2d}. {n:2d} ({a})")
        ultimos_84 = datos_ordenados.head(84)
        frecuencia_reciente = ultimos_84['Num_Int'].value_counts().head(25)
        print(f"\nTOP 25 NUMEROS MAS FRECUENTES (ultimos 84 sorteos):")
        for i, (n, conteo) in enumerate(frecuencia_reciente.items(), 1):
            n = int(n)
            a = self.num_int_a_animal.get(n, "?")
            porcentaje = (conteo / len(ultimos_84)) * 100
            print(f"   {i:2d}. {n:2d}({a:<10}) {conteo:>2} veces ({porcentaje:>5.1f}%)")
        return datos_ordenados.head(25)

    def ver_estado_actual_dia(self, datos):
        from datetime import date
        hoy = date.today()
        print(f"\nESTADO ACTUAL - {hoy}")
        datos_hoy = datos[datos['Fecha'] == hoy]
        if len(datos_hoy) == 0:
            print("No hay registros para hoy")
            print("   Usa la Opcion 1 para agregar sorteos")
            return
        registros_hoy = datos_hoy[['Hora', 'Num_Int', 'Animal', 'Numero']].sort_values('Hora')
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
        frecuencia_completa = datos.groupby('Hora')['Num_Int'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        top_10_map = {}
        for hora_24h in frecuencia_completa['Hora'].unique():
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(25)['Num_Int'].tolist()
            top_10_map[hora_24h] = top_10_lista
        resultados = []
        for fecha, df_dia in datos.groupby('Fecha'):
            aciertos_tempranos = 0
            df_temprano = df_dia[df_dia['Hora'].isin(horas_a_evaluar)]
            for _, row in df_temprano.iterrows():
                hora_filtro = row['Hora']
                num_salio = int(row['Num_Int'])
                if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
                    aciertos_tempranos += 1
            aciertos_manana = 0
            aciertos_tarde = 0
            horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
            horas_tarde = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
            for _, row in df_dia.iterrows():
                hora_filtro = row['Hora']
                num_salio = int(row['Num_Int'])
                if hora_filtro in top_10_map and num_salio in top_10_map[hora_filtro]:
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

    def prediccion_dia_completo(self, datos):
        df = datos.copy()
        if len(df) < 5:
            print("Pocos datos")
            return
        hoy = df.iloc[-1]['Fecha']
        if isinstance(hoy, datetime):
            hoy = hoy.date()
        df_ayer = df[pd.to_datetime(df['Fecha']).dt.date < hoy]
        if df_ayer.empty:
            print("No hay datos del dia anterior")
            return
        ultimo_ayer = df_ayer.iloc[-1]
        ultimo_num = int(ultimo_ayer['Num_Int'])
        ultimo_animal = self.num_int_a_animal.get(ultimo_num, "?")
        print(f"\n{'='*80}")
        print(f"  PREDICCION DEL DIA COMPLETO ({hoy})")
        print(f"  Basada en el ultimo numero del dia anterior: {ultimo_num:2d} ({ultimo_animal}) a las {ultimo_ayer['Solo_hora']}")
        print(f"{'='*80}")
        trans_prob, trans_total = self._transiciones_markov(df)
        hora_freq = self._frecuencias_hora(df, 'Solo_hora')
        horas = ['08:00 AM','09:00 AM','10:00 AM','11:00 AM','12:00 PM','01:00 PM',
                 '02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM']
        for hora in horas:
            m_scores = {}
            for n in range(38):
                p = trans_prob.get((ultimo_num, n), 0)
                if p > 0:
                    m_scores[n] = p
            h_scores = hora_freq.get(hora, {})
            mh = []
            for n in m_scores:
                hp = h_scores.get(n, 0)
                a = self.num_int_a_animal.get(n, "?")
                mh.append((m_scores[n] + hp, m_scores[n], hp, n, a))
            mh.sort(reverse=True)
            print(f"\n  HORA: {hora}")
            print(f"  {'#':<3} {'Num':<4} {'Animal':<12} {'Markov':<7} {'Hora':<7} {'M+H':<7}")
            print(f"  {'-'*40}")
            for i, (sc, mp, hp, n, a) in enumerate(mh[:5], 1):
                print(f"  {i:<3} {n:<4} {a:<12} {mp:<5.1f}% {hp:<5.2f}% {sc:<6.2f}%")

    def top_25_general(self, datos):
        df = datos.copy()
        if len(df) < 5:
            print("Pocos datos")
            return
        freq = df['Num_Int'].value_counts(normalize=True).mul(100)
        print(f"\n{'='*70}")
        print(f"  TOP-25 GENERAL (frecuencia global por Num_Int)")
        print(f"  Basado en {len(df)} sorteos")
        print(f"{'='*70}")
        print(f"  {'#':<3} {'Num':<4} {'Animal':<12} {'Frecuencia':<10} {'#Sorteos':<8}")
        print(f"  {'-'*37}")
        for i, (n, p) in enumerate(freq.head(25).items(), 1):
            n = int(n)
            a = self.num_int_a_animal.get(n, "?")
            cnt = int(p * len(df) / 100)
            print(f"  {i:<3} {n:<4} {a:<12} {p:<7.2f}%  {cnt:<8}")

    # -----------------------------------------------------------
    # MATRICES DE MARKOV: GLOBAL y POR HORA
    # -----------------------------------------------------------

    def get_parejas_horarias(self):
        """Devuelve las 11 transiciones horarias del dia."""
        return [
            ('08:00:00', '09:00:00'), ('09:00:00', '10:00:00'),
            ('10:00:00', '11:00:00'), ('11:00:00', '12:00:00'),
            ('12:00:00', '13:00:00'), ('13:00:00', '14:00:00'),
            ('14:00:00', '15:00:00'), ('15:00:00', '16:00:00'),
            ('16:00:00', '17:00:00'), ('17:00:00', '18:00:00'),
            ('18:00:00', '19:00:00'),
        ]

    def _markov_cache_key(self, datos, incluir_trasnocho):
        return (len(datos), datos['Num_Int'].sum(), datos['Fecha'].nunique(), incluir_trasnocho)

    def construir_matrices_markov(self, datos, incluir_trasnocho=False):
        """Construye y retorna (trans_count_global, total_global, trans_count_hora, total_hora).
        Descubre pares horarios sobre la marcha (un solo pase sobre datos).
        Las matrices se cachean mientras los datos no cambien."""
        key = self._markov_cache_key(datos, incluir_trasnocho)
        cached = self._cache_markov.get(key)
        if cached is not None:
            return cached

        from collections import defaultdict
        trans_g = defaultdict(lambda: defaultdict(int))
        total_g = defaultdict(int)
        trans_h = {}
        total_h = {}

        for fecha, df in datos.groupby('Fecha'):
            df = df.sort_values('Hora')
            for i in range(1, len(df)):
                a, b = df.iloc[i-1]['Num_Int'], df.iloc[i]['Num_Int']
                hp, hn = df.iloc[i-1]['Hora'], df.iloc[i]['Hora']
                trans_g[a][b] += 1
                total_g[a] += 1
                th = trans_h.get((hp, hn))
                if th is None:
                    trans_h[(hp, hn)] = defaultdict(lambda: defaultdict(int))
                    total_h[(hp, hn)] = defaultdict(int)
                    th = trans_h[(hp, hn)]
                th[a][b] += 1
                total_h[(hp, hn)][a] += 1

        if incluir_trasnocho:
            trasnocho_key = ('19:00:00', '08:00:00')
            if trasnocho_key not in trans_h:
                trans_h[trasnocho_key] = defaultdict(lambda: defaultdict(int))
                total_h[trasnocho_key] = defaultdict(int)
            fechas = sorted(datos['Fecha'].unique())
            for f_hoy, f_man in zip(fechas, fechas[1:]):
                hoy = datos[datos['Fecha'] == f_hoy].sort_values('Hora')
                man = datos[datos['Fecha'] == f_man].sort_values('Hora')
                if hoy.empty or man.empty:
                    continue
                ultimo, primero = hoy.iloc[-1], man.iloc[0]
                if ultimo['Hora'] == '19:00:00' and primero['Hora'] == '08:00:00':
                    a, b = ultimo['Num_Int'], primero['Num_Int']
                    trans_g[a][b] += 1
                    total_g[a] += 1
                    trans_h[trasnocho_key][a][b] += 1
                    total_h[trasnocho_key][a] += 1

        self._cache_markov[key] = (trans_g, total_g, trans_h, total_h)
        return trans_g, total_g, trans_h, total_h

    def preparar_datos_markov(self, datos):
        """Asegura que datos tenga las columnas necesarias para construir matrices de Markov."""
        df = datos.copy()
        if 'Num_Int' not in df.columns:
            raise ValueError("Datos deben tener columna 'Num_Int'")
        if 'Fecha' not in df.columns:
            if 'Timestamp' in df.columns:
                df['Fecha'] = pd.to_datetime(df['Timestamp']).dt.date
            else:
                raise ValueError("Datos deben tener columna 'Fecha' o 'Timestamp'")
        if 'Hora' not in df.columns:
            if 'Timestamp' in df.columns:
                df['Hora'] = df['Timestamp'].dt.strftime('%H:%M:%S')
            else:
                raise ValueError("Datos deben tener columna 'Hora' o 'Timestamp'")
        if df['Num_Int'].dtype not in ('int64', 'int32', int, float):
            df['Num_Int'] = df['Num_Int'].astype(int)
        df['Hora'] = df['Hora'].astype(str).str.strip().str.zfill(8)
        return df

    def get_matriz_global(self, datos, top_k=38, incluir_trasnocho=False):
        """
        Retorna dict[numero_int] = [(siguiente_num, prob %, muestras), ...] top_k de la matriz global.
        """
        datos = self.preparar_datos_markov(datos)
        trans_g, total_g, _, _ = self.construir_matrices_markov(datos, incluir_trasnocho=incluir_trasnocho)
        resultado = {}
        for n in range(38):
            if n not in total_g or total_g[n] == 0:
                resultado[n] = []
                continue
            total = total_g[n]
            scores = [(a2, cnt / total * 100, cnt) for a2, cnt in trans_g[n].items()]
            scores.sort(key=lambda x: -x[1])
            resultado[n] = scores[:top_k]
        return resultado

    def get_matriz_global_por_animal(self, datos, top_k=38, incluir_trasnocho=False):
        """Compat: wrappea get_matriz_global con claves de animal."""
        m = self.get_matriz_global(datos, top_k=top_k, incluir_trasnocho=incluir_trasnocho)
        return {self.num_int_a_animal.get(k, str(k)): v for k, v in m.items()}

    def get_matriz_hora(self, datos, hora_origen, hora_destino, top_k=38, incluir_trasnocho=False):
        """
        Retorna dict[numero_int] = [(siguiente_num, prob %, muestras), ...] para una transicion horaria.
        """
        _, _, trans_h, total_h = self.construir_matrices_markov(datos, incluir_trasnocho=incluir_trasnocho)
        pareja = (hora_origen, hora_destino)
        resultado = {}
        for n in range(38):
            if n not in total_h.get(pareja, {}) or total_h[pareja][n] == 0:
                resultado[n] = []
                continue
            total = total_h[pareja][n]
            scores = [(a2, cnt / total * 100, cnt) for a2, cnt in trans_h[pareja][n].items()]
            scores.sort(key=lambda x: -x[1])
            resultado[n] = scores[:top_k]
        return resultado

    def get_matriz_hora_por_animal(self, datos, hora_origen, hora_destino, top_k=38, incluir_trasnocho=False):
        """Compat: wrappea get_matriz_hora con claves de animal."""
        m = self.get_matriz_hora(datos, hora_origen, hora_destino, top_k=top_k, incluir_trasnocho=incluir_trasnocho)
        return {self.num_int_a_animal.get(k, str(k)): v for k, v in m.items()}

    def get_matriz_segundo_orden(self, datos, num1, num2, top_k=38):
        """Retorna [(siguiente_num, prob%, muestras)] top_k usando Markov de 2do orden sobre Num_Int."""
        from collections import defaultdict
        d = self.preparar_datos_markov(datos)
        conteo = defaultdict(lambda: defaultdict(int))
        total = defaultdict(int)
        for _, grupo in d.groupby('Fecha'):
            grupo = grupo.sort_values('Hora')
            for i in range(2, len(grupo)):
                a, b, c = grupo.iloc[i-2]['Num_Int'], grupo.iloc[i-1]['Num_Int'], grupo.iloc[i]['Num_Int']
                conteo[(a, b)][c] += 1
                total[(a, b)] += 1
        par = (num1, num2)
        if par not in total or total[par] == 0:
            return []
        t = total[par]
        items = [(c, cnt / t * 100, cnt) for c, cnt in conteo[par].items()]
        items.sort(key=lambda x: -x[1])
        return items[:top_k]

    def get_matriz_coocurrencia(self, datos, top_k=38):
        """
        Retorna dict[numero_int] = [(siguiente_num, prob %, muestras), ...] top_k
        basado en co-ocurrencia: numeros que suelen salir en el mismo dia.
        """
        from collections import defaultdict
        d = datos.copy()
        if 'Fecha' not in d.columns:
            return {}
        coex = defaultdict(lambda: defaultdict(int))
        total = defaultdict(int)
        for _, grupo in d.groupby('Fecha'):
            nums = set(grupo['Num_Int'].unique())
            for n in nums:
                total[n] += 1
                for n2 in nums:
                    if n2 != n:
                        coex[n][n2] += 1
        resultado = {}
        for n in range(38):
            if n not in total or total[n] == 0:
                resultado[n] = []
                continue
            t = total[n]
            scores = [(n2, cnt / t * 100, cnt) for n2, cnt in coex[n].items()]
            scores.sort(key=lambda x: -x[1])
            resultado[n] = scores[:top_k]
        return resultado

    def get_matriz_coocurrencia_por_animal(self, datos, top_k=38):
        """Wrapper de get_matriz_coocurrencia con claves de animal."""
        m = self.get_matriz_coocurrencia(datos, top_k=top_k)
        return {self.num_int_a_animal.get(k, str(k)): v for k, v in m.items()}

    def get_prediccion_combinada(self, datos, num_origen, hora_origen, hora_destino, top_k=10, incluir_trasnocho=False):
        """
        Combina 4 fuentes para predecir el siguiente Num_Int:
          1. Markov global
          2. Markov x hora
          3. Frecuencia historica por hora (Num_Int)
          4. Co-ocurrencia en el mismo dia (Num_Int)
        Retorna [(siguiente_num, score_combinado, muestras), ...] top_k.
        Construye las matrices de Markov una sola vez.
        """
        from collections import defaultdict
        d = self.preparar_datos_markov(datos)
        trans_g, total_g, trans_h, total_h = self.construir_matrices_markov(d, incluir_trasnocho=incluir_trasnocho)

        # --- 1. Markov global ---
        g_items = []
        if num_origen in total_g and total_g[num_origen]:
            tot = total_g[num_origen]
            g_items = [(a2, cnt / tot * 100, cnt) for a2, cnt in trans_g[num_origen].items()]
            g_items.sort(key=lambda x: -x[1])
        g_scores = {n: p for n, p, _ in g_items}

        # --- 2. Markov x hora ---
        pareja = (hora_origen, hora_destino)
        h_items = []
        th = total_h.get(pareja, {})
        if num_origen in th and th[num_origen]:
            tot = th[num_origen]
            h_items = [(a2, cnt / tot * 100, cnt) for a2, cnt in trans_h[pareja][num_origen].items()]
            h_items.sort(key=lambda x: -x[1])
        h_scores = {n: p for n, p, _ in h_items}

        # --- 3. Frecuencia historica por hora ---
        hora_12 = pd.to_datetime(hora_origen, format='%H:%M:%S').strftime('%I:%M %p')
        sub = d[d['Solo_hora'] == hora_12] if 'Solo_hora' in d.columns else d
        freq_hora = sub['Num_Int'].value_counts(normalize=True).mul(100).to_dict()

        # --- 4. Co-ocurrencia en el mismo dia ---
        cooc = defaultdict(int)
        cooc_total = 0
        for _, grupo in d.groupby('Fecha'):
            nums = set(grupo['Num_Int'].unique())
            if num_origen in nums:
                cooc_total += 1
                for n2 in nums:
                    if n2 != num_origen:
                        cooc[n2] += 1
        cooc_scores = {n: c / cooc_total * 100 for n, c in cooc.items()} if cooc_total else {}

        # --- Combinacion ponderada ---
        todos = set(g_scores.keys()) | set(h_scores.keys()) | set(freq_hora.keys()) | set(cooc_scores.keys())

        muestras_h = sum(c for _, _, c in h_items)
        w_h = min(0.9, max(0.1, muestras_h / 50))
        w_g = 0.40 - w_h * 0.5
        w_f = 0.20
        w_c = 0.40 - w_h * 0.5

        combinado = {}
        for n2 in todos:
            pg = g_scores.get(n2, 0)
            ph = h_scores.get(n2, 0)
            pf = freq_hora.get(n2, 0)
            pc = cooc_scores.get(n2, 0)
            score = pg * w_g + ph * w_h + pf * w_f + pc * w_c
            muestras = sum([
                1 if n2 in g_scores else 0,
                1 if n2 in h_scores else 0,
                1 if n2 in freq_hora else 0,
                1 if n2 in cooc_scores else 0,
            ])
            combinado[n2] = (score, muestras)

        sorted_items = sorted(combinado.items(), key=lambda x: -x[1][0])
        return [(n, s, m) for n, (s, m) in sorted_items[:top_k]]

    def analizar_secuencias_aciertos_fallos(self, datos, top_k=38, train_pct=0.7):
        from collections import defaultdict
        d = datos.copy()
        print(f"\n{'='*70}")
        print(f"  RACHAS DE ACIERTOS Y FALLOS (Markov x Hora, Top-{top_k})")
        print(f"{'='*70}")
        fechas_completas = d.groupby('Fecha').size()
        fechas_completas = fechas_completas[fechas_completas == 12].index
        d = d[d['Fecha'].isin(fechas_completas)]
        fechas_ordenadas = sorted(fechas_completas)
        split_idx = int(len(fechas_ordenadas) * train_pct)
        fechas_train = set(fechas_ordenadas[:split_idx])
        fechas_test = fechas_ordenadas[split_idx:]
        n_dias = len(fechas_test)
        print(f"  Train: {len(fechas_train)} dias, Test: {n_dias} dias\n")

        d_train = d[d['Fecha'].isin(fechas_train)]
        dp_train = self.preparar_datos_markov(d_train)
        _, _, trans_h, total_h = self.construir_matrices_markov(dp_train, incluir_trasnocho=False)

        racha_acierto = defaultdict(list)
        racha_fallo = defaultdict(list)
        hora_stats = defaultdict(lambda: {'total': 0, 'aciertos': 0, 'tras_acierto': 0, 'tras_fallo': 0, 'prev_acierto': 0, 'prev_fallo': 0})

        for fecha, grupo in d[d['Fecha'].isin(fechas_test)].groupby('Fecha'):
            grupo = grupo.sort_values('Hora')
            aciertos_dia = []
            for i in range(1, len(grupo)):
                ant = int(grupo.iloc[i-1]['Num_Int'])
                sig = int(grupo.iloc[i]['Num_Int'])
                hp = grupo.iloc[i-1]['Hora']
                hn = grupo.iloc[i]['Hora']
                par = (hp, hn)
                hora_dict = trans_h.get(par, {})
                ant_dict = hora_dict.get(ant, {})
                top = sorted(ant_dict, key=lambda n: -ant_dict[n])[:top_k] if ant_dict else []
                acierto = sig in top
                aciertos_dia.append((par, acierto))
                hora_stats[par]['total'] += 1
                if acierto:
                    hora_stats[par]['aciertos'] += 1

            for i in range(1, len(aciertos_dia)):
                par_curr, curr_ac = aciertos_dia[i]
                _, prev_ac = aciertos_dia[i - 1]
                if prev_ac:
                    hora_stats[par_curr]['prev_acierto'] += 1
                    if curr_ac:
                        hora_stats[par_curr]['tras_acierto'] += 1
                else:
                    hora_stats[par_curr]['prev_fallo'] += 1
                    if curr_ac:
                        hora_stats[par_curr]['tras_fallo'] += 1

            i = 0
            while i < len(aciertos_dia):
                _, ac = aciertos_dia[i]
                if ac:
                    start = i
                    while i < len(aciertos_dia) and aciertos_dia[i][1]:
                        i += 1
                    largo = i - start
                    for n in range(1, largo):
                        racha_acierto[n].append(True)
                    if i < len(aciertos_dia):
                        racha_acierto[largo].append(False)
                else:
                    start = i
                    while i < len(aciertos_dia) and not aciertos_dia[i][1]:
                        i += 1
                    largo = i - start
                    for n in range(1, largo + 1):
                        if n < largo:
                            racha_fallo[n].append(True)
                        elif i < len(aciertos_dia):
                            racha_fallo[n].append(False)

        print(f"\n  RACHAS DE ACIERTOS (Top-{top_k})")
        print(f"  {'# Aciertos seg':<16} {'Sigue acertando':>16} {'%':>6}")
        print(f"  {'-'*40}")
        for n in sorted(racha_acierto.keys()):
            vals = racha_acierto[n]
            if vals:
                pct = sum(vals) / len(vals) * 100
                print(f"  {n} acierto(s) seguido(s)   {sum(vals):>4}/{len(vals):<4} {pct:>5.1f}%")

        print(f"\n  RACHAS DE FALLOS (Top-{top_k})")
        print(f"  {'# Fallos seg':<16} {'Sigue fallando':>16} {'%':>6}")
        print(f"  {'-'*40}")
        for n in sorted(racha_fallo.keys()):
            vals = racha_fallo[n]
            pct = sum(vals) / len(vals) * 100
            print(f"  {n} fallo(s) seguido(s)     {sum(vals):>4}/{len(vals):<4} {pct:>5.1f}%")

        print(f"\n\n  POR HORA")
        print(f"  {'Hora':<14} {'%Acierto':>10} {'Tras acierto':>14} {'Tras fallo':>12}")
        print(f"  {'-'*52}")
        for (hp, hn), st in sorted(hora_stats.items()):
            if st['total'] == 0:
                continue
            pct_ac = st['aciertos'] / st['total'] * 100
            tr_ac = st['tras_acierto'] / max(st['prev_acierto'], 1) * 100
            tr_fa = st['tras_fallo'] / max(st['prev_fallo'], 1) * 100
            h12 = pd.to_datetime(hp, format='%H:%M:%S').strftime('%I:%M %p')
            print(f"  {h12:<14} {pct_ac:>8.1f}%  {tr_ac:>10.1f}%  {tr_fa:>10.1f}%")
        print(f"\n{'='*70}\n")

    def comparar_estrategias(self, datos, top_k=38, train_pct=0.7):
        from collections import defaultdict
        d = datos.copy()
        print(f"\n{'='*80}")
        print(f"  COMPARACION DE ESTRATEGIAS (Top-{top_k})")
        print(f"{'='*80}")
        fechas_completas = set(d.groupby('Fecha').size()
                               [d.groupby('Fecha').size() == 12].index)
        d = d[d['Fecha'].isin(fechas_completas)]
        fechas_ordenadas = sorted(fechas_completas)
        split_idx = int(len(fechas_ordenadas) * train_pct)
        fechas_train = set(fechas_ordenadas[:split_idx])
        fechas_test = fechas_ordenadas[split_idx:]
        print(f"  Train: {len(fechas_train)} dias, Test: {len(fechas_test)} dias\n")

        d_train = d[d['Fecha'].isin(fechas_train)]
        dp_train = self.preparar_datos_markov(d_train)
        trans_g, total_g, trans_h, total_h = self.construir_matrices_markov(dp_train, incluir_trasnocho=False)

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
        solo_stats = defaultdict(lambda: {'g': 0, 'h': 0, 'f': 0, 'c': 0})
        por_hora = defaultdict(lambda: {'total': 0, 'g': 0, 'h': 0, 'gxh': 0, 'full': 0})

        for fecha, grupo in d[d['Fecha'].isin(fechas_test)].groupby('Fecha'):
            grupo = grupo.sort_values('Hora')
            for i in range(1, len(grupo)):
                ant = int(grupo.iloc[i-1]['Num_Int'])
                sig = int(grupo.iloc[i]['Num_Int'])
                hp = grupo.iloc[i-1]['Hora']
                hn = grupo.iloc[i]['Hora']
                par = (hp, hn)

                g_dict = {}
                if ant in total_g and total_g[ant] > 0:
                    t = total_g[ant]
                    g_dict = {n: c / t * 100 for n, c in trans_g[ant].items()}
                g_top = set(sorted(g_dict, key=g_dict.get, reverse=True)[:top_k])

                h_dict = {}
                if ant in total_h.get(par, {}) and total_h[par][ant] > 0:
                    t = total_h[par][ant]
                    h_dict = {n: c / t * 100 for n, c in trans_h[par][ant].items()}
                h_top = set(sorted(h_dict, key=h_dict.get, reverse=True)[:top_k])

                h12 = pd.to_datetime(hp, format='%H:%M:%S').strftime('%I:%M %p')
                f_series = freq_por_hora.get(h12, pd.Series(dtype=float))
                f_top = set(f_series.head(top_k).index)

                c_dict = {n: c / cooc_total_global[ant] * 100 for n, c in cooc_global[ant].items()} if cooc_total_global[ant] > 0 else {}
                c_top = set(sorted(c_dict, key=c_dict.get, reverse=True)[:top_k])

                gxh_scores = {}
                for n in range(38):
                    pg = g_dict.get(n, 0)
                    ph = h_dict.get(n, 0)
                    if pg > 0 or ph > 0:
                        gxh_scores[n] = pg + ph
                gxh_top = set(sorted(gxh_scores, key=gxh_scores.get, reverse=True)[:top_k])

                w_h = min(0.9, max(0.1, total_h.get(par, {}).get(ant, 0) / 50))
                w_g = 0.40 - w_h * 0.5
                w_f = 0.20
                w_c = 0.40 - w_h * 0.5
                full_scores = {}
                for n in range(38):
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

                hits = {'g': sig in g_top, 'h': sig in h_top, 'f': sig in f_top, 'c': sig in c_top}
                solo_count = sum(hits.values())
                if solo_count == 1:
                    for k, v in hits.items():
                        if v:
                            solo_stats[k][k] += 1

                por_hora[hp]['total'] += 1
                if sig in g_top: por_hora[hp]['g'] += 1
                if sig in h_top: por_hora[hp]['h'] += 1
                if sig in gxh_top: por_hora[hp]['gxh'] += 1
                if sig in full_top: por_hora[hp]['full'] += 1

        totals = {'total': sum(r['total'] for r in resultados.values())}
        for key in ['g', 'h', 'f', 'c', 'gxh', 'full']:
            totals[key] = sum(r[key] for r in resultados.values())

        print(f"\n  GLOBAL (todas las transiciones)")
        print(f"  {'Estrategia':<20} {'Aciertos':>10} {'Total':>8} {'%':>6}")
        print(f"  {'-'*46}")
        labels = {'g': 'G (Markov global)', 'h': 'H (Markov x hora)', 'f': 'F (Frec. hora)',
                  'c': 'C (Co-ocurrencia)', 'gxh': 'GxH (Global x Hora)', 'full': 'Full (4 fuentes)'}
        for key in ['g', 'h', 'f', 'c', 'gxh', 'full']:
            pct = totals[key] / totals['total'] * 100 if totals['total'] else 0
            print(f"  {labels[key]:<20} {totals[key]:>10} {totals['total']:>8} {pct:>5.1f}%")

        print(f"\n  ACIERTOS SOLITARIOS (solo una estrategia acerto)")
        print(f"  {'Estrategia':<20} {'Casos':>8}")
        print(f"  {'-'*30}")
        for k in ['g', 'h', 'f', 'c']:
            print(f"  {labels[k]:<20} {solo_stats[k][k]:>8}")

        print(f"\n  TOP-3 HORAS MEJORES (Full)")
        horas_sorted = sorted(por_hora.items(), key=lambda x: x[1]['full']/max(x[1]['total'],1), reverse=True)
        for o, st in horas_sorted[:3]:
            h12 = pd.to_datetime(o, format='%H:%M:%S').strftime('%I:%M %p')
            pct_g = st['g']/st['total']*100
            pct_h = st['h']/st['total']*100
            pct_gxh = st['gxh']/st['total']*100
            pct_full = st['full']/st['total']*100
            print(f"  {h12:<10} G:{pct_g:.0f}% H:{pct_h:.0f}% GxH:{pct_gxh:.0f}% Full:{pct_full:.0f}%")

        print(f"\n  TOP-3 HORAS PEORES (Full)")
        for o, st in horas_sorted[-3:]:
            h12 = pd.to_datetime(o, format='%H:%M:%S').strftime('%I:%M %p')
            pct_g = st['g']/st['total']*100
            pct_h = st['h']/st['total']*100
            pct_gxh = st['gxh']/st['total']*100
            pct_full = st['full']/st['total']*100
            print(f"  {h12:<10} G:{pct_g:.0f}% H:{pct_h:.0f}% GxH:{pct_gxh:.0f}% Full:{pct_full:.0f}%")

        full_win = sum(1 for st in resultados.values() if st['full'] > st['gxh'])
        gxh_win = sum(1 for st in resultados.values() if st['gxh'] > st['full'])
        print(f"\n  Full GANA a GxH en {full_win} transiciones")
        print(f"  GxH GANA a Full en {gxh_win} transiciones")
        print(f"\n{'='*80}\n")

    def main_menu(self, datos):
        opciones = [
            "Ingresar Sorteo del Dia (Actualizar Excel)",
            "Validacion Cruzada de Precision del Modelo (Markov)",
            "Mostrar Matriz de Probabilidad de Transicion",
            "Prediccion Siguiente (Basado en Ultimo Numero - Markov)",
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
                        modelo_cargado, le_y_cargado, datos_con_features.copy(), k=25
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
                    current = datetime.strptime(inicio, "%Y-%m-%d")
                    end_dt = datetime.strptime(fin, "%Y-%m-%d")
                    while current <= end_dt:
                        all_dates.add(current.strftime("%Y-%m-%d"))
                        current += timedelta(days=1)
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
