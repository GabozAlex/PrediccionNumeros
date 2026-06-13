"""
Core Lottery class with basic functionality and configuration.
This module contains the foundational methods for animal-based lottery analysis.
"""

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


class LoteriaCore:
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
        try:
            n = int(entrada)
            if 0 <= n <= self.config['max_numero']:
                animal = self.num_int_a_animal.get(n)
                if animal:
                    return animal
        except (ValueError, TypeError):
            pass
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

    def preparar_datos_ml_completo(self, datos):
        """Prepara datos para ML con todas las características necesarias."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        numeric_features = [
            'Numero', 'Num_Int', 'Num_Int_Prev', 'Dif_Ciclica_N', 'Media_5_N', 'Std_5_N',
            'Hora_Num', 'Dia_Semana', 'Mes', 'Frecuencia_Animal_10', 'Media_Movil_5', 'Std_Movil_5'
        ]
        categorical_features = ['Animal', 'Color_Numero', 'Color_Previo', 'Paridad_Numero',
                                'Paridad_Previo', 'Grupo_Ruleta', 'Grupo_Ruleta_Previo',
                                'Repite_Animal', 'Mismo_Animal_3_Sorteos', 'Repite_Num', 'Mismo_Num_3']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])

        return pipeline

    def crear_pipeline_ml(self, modelo, numeric_features, categorical_features):
        """Crea un pipeline ML para entrenamiento."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
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

    def agregar_caracteristicas_avanzadas(self, datos):
        """Calcula características avanzadas para cada sorteo."""
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

        def color_numero(num):
            if num == 0 or num == 37:
                return 0
            rojos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
            return 1 if num in rojos else 2

        def paridad_numero(num):
            if num == 0 or num == 37:
                return 0
            return 1 if num % 2 == 0 else 2

        df['Color_Numero'] = df['Numero'].apply(color_numero)
        df['Color_Previo'] = df['Color_Numero'].shift(1)
        df['Paridad_Numero'] = df['Numero'].apply(paridad_numero)
        df['Paridad_Previo'] = df['Paridad_Numero'].shift(1)

        return df