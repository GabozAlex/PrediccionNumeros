"""
Machine Learning utilities for lottery prediction.
This module contains ML pipeline creation, training, and optimization methods.
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd
from datetime import datetime


class MLModelHandler:
    """Handles ML model creation, training, and optimization for lottery prediction."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.numeric_features = [
            'Numero', 'Num_Int', 'Num_Int_Prev', 'Dif_Ciclica_N', 'Media_5_N', 'Std_5_N',
            'Hora_Num', 'Dia_Semana', 'Mes', 'Frecuencia_Animal_10', 'Media_Movil_5', 'Std_Movil_5'
        ]
        self.categorical_features = [
            'Animal', 'Color_Numero', 'Color_Previo', 'Paridad_Numero',
            'Paridad_Previo', 'Grupo_Ruleta', 'Grupo_Ruleta_Previo',
            'Repite_Animal', 'Mismo_Animal_3_Sorteos', 'Repite_Num', 'Mismo_Num_3'
        ]

    def preparar_datos_ml_completo(self, datos):
        """Prepara datos para ML con todas las características necesarias."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numeric_features),
                ('cat', 'onehot', self.categorical_features)
            ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        return pipeline

    def crear_pipeline_ml(self, modelo, numeric_features=None, categorical_features=None):
        """Crea un pipeline ML para entrenamiento."""
        if numeric_features is None:
            numeric_features = self.numeric_features
        if categorical_features is None:
            categorical_features = self.categorical_features

        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', 'onehot', categorical_features)
            ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', modelo)
        ])
        return pipeline

    def calcular_precision_top_k(self, y_real, y_proba, k=3):
        """Calcula precisión para Top-K predicciones."""
        top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
        aciertos = 0
        for i, real in enumerate(y_real):
            if real in top_k_indices[i]:
                aciertos += 1
        return aciertos / len(y_real)

    def entrenar_modelo_ml(self, X, Y, modelo, modelo_nombre, numeric_features=None, categorical_features=None):
        """Entrena un modelo ML y registra métricas."""
        if numeric_features is None:
            numeric_features = self.numeric_features
        if categorical_features is None:
            categorical_features = self.categorical_features

        pipeline = self.crear_pipeline_ml(modelo, numeric_features, categorical_features)

        pipeline.fit(X, Y)

        y_pred_proba = pipeline.predict_proba(X)
        precision = self.calcular_precision_top_k(Y, y_pred_proba, k=25)

        metricas = {
            'precision_top_25': precision,
            'modelos': modelo_nombre,
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info(f"Entrenamiento {modelo_nombre}: precisión Top-25 = {precision:.2%}")

        return pipeline, metricas

    def optimizar_hiperparametros_rf(self, X, Y, numeric_features=None, categorical_features=None, cv=3):
        """Realiza búsqueda de hiperparámetros para Random Forest."""
        if numeric_features is None:
            numeric_features = self.numeric_features
        if categorical_features is None:
            categorical_features = self.categorical_features

        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', 'onehot', categorical_features)
            ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])

        param_grid = {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

        tscv = TimeSeriesSplit(n_splits=cv)
        grid_search = RandomizedSearchCV(
            pipeline, param_grid, cv=tscv, n_jobs=-1, verbose=1, scoring='accuracy', n_iter=20
        )

        grid_search.fit(X, Y)
        self.logger.info(f"Mejor RF: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def optimizar_hiperparametros_xgb(self, X, Y, numeric_features=None, categorical_features=None, cv=3):
        """Realiza búsqueda de hiperparámetros para XGBoost."""
        if numeric_features is None:
            numeric_features = self.numeric_features
        if categorical_features is None:
            categorical_features = self.categorical_features

        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', 'onehot', categorical_features)
            ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ])

        param_grid = {
            'model__n_estimators': [100, 200, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 6, 9],
            'model__min_child_weight': [1, 3, 5],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits=cv)
        grid_search = RandomizedSearchCV(
            pipeline, param_grid, cv=tscv, n_jobs=-1, verbose=1, scoring='accuracy', n_iter=20
        )

        grid_search.fit(X, Y)
        self.logger.info(f"Mejor XGB: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def entrenar_modelo_con_optimizacion(self, X, Y, tipo_modelo, numeric_features=None, categorical_features=None):
        """Entrena modelo con optimización de hiperparámetros."""
        if tipo_modelo == 'rf':
            modelo_opt = self.optimizar_hiperparametros_rf(X, Y, numeric_features, categorical_features)
        elif tipo_modelo == 'xgb':
            modelo_opt = self.optimizar_hiperparametros_xgb(X, Y, numeric_features, categorical_features)
        else:
            raise ValueError(f"Modelo no soportado: {tipo_modelo}")

        y_pred_proba = modelo_opt.predict_proba(X)
        precision = self.calcular_precision_top_k(Y, y_pred_proba, k=25)

        metricas = {
            'precision_top_25': precision,
            'modelo': tipo_modelo,
            'timestamp': datetime.now().isoformat()
        }

        return modelo_opt, metricas

    def guardar_modelo(self, modelo, le_y, metricas, nombre_modelo, modelo_dir=None):
        """Guarda modelo y metadatos."""
        if modelo_dir is None:
            modelo_dir = f"modelos/{self.config['nombre_modelo']}"

        import os
        os.makedirs(modelo_dir, exist_ok=True)

        modelo_path = f"{modelo_dir}/{nombre_modelo}.pkl"
        joblib.dump(modelo, modelo_path)

        metadatos = {
            'metricas': metricas,
            'le_y': le_y,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        metadatos_path = f"{modelo_dir}/{nombre_modelo}_meta.pkl"
        joblib.dump(metadatos, metadatos_path)

        self.logger.info(f"Modelo guardado: {modelo_path}")

    def cargar_modelo(self, modelo_path):
        """Carga modelo desde archivo."""
        return joblib.load(modelo_path)

    def cargar_ultimo_modelo(self, tipo_modelo):
        """Carga el último modelo entrenado del tipo especificado."""
        import glob

        modelo_dir = f"modelos/{self.config['nombre_modelo']}"
        patron = f"{modelo_dir}/{tipo_modelo}_*.pkl"
        archivos = glob.glob(patron)

        if not archivos:
            return None

        modelos = []
        for archivo in archivos:
            try:
                modelos.append((archivo, joblib.load(archivo)))
            except Exception as e:
                self.logger.error(f"Error cargando {archivo}: {e}")

        if not modelos:
            return None

        archivos_ordenados = sorted(modelos, key=lambda x: x[0])
        return archivos_ordenados[-1][1]

    def random_forest_optimizado(self, datos):
        """Entrena y guarda modelo Random Forest optimizado."""
        X, Y = self._preparar_datos_para_ml(datos)
        modelo, metricas = self.entrenar_modelo_con_optimizacion(X, Y, 'rf')
        self.guardar_modelo(modelo, None, metricas, 'random_forest')
        return modelo

    def xgboost_optimizado(self, datos):
        """Entrena y guarda modelo XGBoost optimizado."""
        X, Y = self._preparar_datos_para_ml(datos)
        modelo, metricas = self.entrenar_modelo_con_optimizacion(X, Y, 'xgb')
        self.guardar_modelo(modelo, None, metricas, 'xgboost')
        return modelo

    def _preparar_datos_para_ml(self, datos):
        """Prepara datos para ML (placeholder para implementación real)."""
        X = datos[self.numeric_features + self.categorical_features]
        y = datos['Num_Int']
        return X, y

    def predecir_top_k_por_hora(self, pipeline, le_y, df_ml, k=25):
        """Hace predicción Top-K para una hora específica."""
        probabilidades = pipeline.predict_proba(df_ml)
        predicciones = []

        for i in range(len(probabilidades)):
            indices_top_k = np.argsort(probabilidades[i])[-k:][::-1]
            for idx in indices_top_k:
                pred_num = le_y.classes_[idx]
                prob = probabilidades[i][idx]
                predicciones.append((pred_num, prob))

        return predicciones

    def simular_estrategia(self, datos, top_10_map):
        """Simula una estrategia de apuestas basada en predicciones."""
        aciertos = 0
        total_apuestas = 0

        for i in range(1, len(datos)):
            hora_actual = datos.iloc[i]['Hora']
            if hora_actual in top_10_map:
                numero_predicho = top_10_map[hora_actual][0]
                numero_real = datos.iloc[i]['Num_Int']
                if numero_predicho == numero_real:
                    aciertos += 1
                total_apuestas += 1

        if total_apuestas > 0:
            precision = aciertos / total_apuestas
            return {
                'aciertos': aciertos,
                'total_apuestas': total_apuestas,
                'precision': precision
            }
        else:
            return {'aciertos': 0, 'total_apuestas': 0, 'precision': 0}