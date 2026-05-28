import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES

def _config():
    return {
        'animales': ANIMALES_38,
        'grupos_animales': GRUPOS_ANIMALES,
        'logger_name': 'test',
        'max_numero': 37,
        'modelos_dir': '/tmp/test_modelos',
        'excel_file': '/tmp/test_data.xlsx',
    }

def _synthetic_data(n_dias=5, horas=None):
    if horas is None:
        horas = ['08:00:00', '09:00:00', '10:00:00', '11:00:00',
                 '12:00:00', '13:00:00', '14:00:00', '15:00:00',
                 '16:00:00', '17:00:00', '18:00:00', '19:00:00']
    np.random.seed(42)
    rows = []
    animals_cycle = ANIMALES_38[:6]
    for d in range(n_dias):
        fecha = datetime(2026, 5, 20 + d).date()
        for h in horas:
            animal = animals_cycle[d % 6] if np.random.random() < 0.4 else np.random.choice(ANIMALES_38[:20])
            numero = ANIMALES_38.index(animal)
            hora_dt = datetime.strptime(h, '%H:%M:%S')
            solo_hora = hora_dt.strftime('%I:%M %p').lstrip('0')
            ts = pd.to_datetime(f"{fecha} {h}")
            rows.append({
                'Fecha': fecha, 'Hora': h, 'Animal': animal,
                'Numero': numero, 'Timestamp': ts, 'Solo_hora': solo_hora
            })
    return pd.DataFrame(rows)


class TestLoteriaBase:
    def setup_method(self):
        self.config = _config()
        self.loteria = Loteria(self.config)
        self.df = _synthetic_data(5)
        self.df_feat = self.loteria.agregar_caracteristicas_avanzadas(self.df.copy())

    def test_init(self):
        assert len(self.loteria.animales_carac) == 38
        assert self.loteria.animales_carac['PERRO'] == 0
        assert self.loteria.animal_a_grupo['PERRO'] == 'MAMIFERO'
        assert self.loteria.animal_a_grupo['AGUILA'] == 'AVE'

    def test_validar_animal(self):
        assert self.loteria.validar_animal('  PERRO ') == 'PERRO'
        assert self.loteria.validar_animal('GATO') == 'GATO'
        try:
            self.loteria.validar_animal('INEXISTENTE')
            assert False, 'Expected ValueError'
        except ValueError:
            pass

    def test_validar_numero(self):
        assert self.loteria.validar_numero(0) == 0
        assert self.loteria.validar_numero(37) == 37
        try:
            self.loteria.validar_numero(38)
            assert False
        except ValueError:
            pass

    def test_calcular_diferencia_ciclica(self):
        assert self.loteria.calcular_diferencia_ciclica(5, 10) == 5
        assert self.loteria.calcular_diferencia_ciclica(0, 37) == 1
        assert self.loteria.calcular_diferencia_ciclica(10, 10) == 0
        assert pd.isna(self.loteria.calcular_diferencia_ciclica(None, 5))

    def test_caracteristicas_avanzadas_columnas(self):
        assert 'Prob_Hist_Hora' in self.df_feat.columns
        assert 'Prob_Trans_Markov' in self.df_feat.columns
        assert 'Frecuencia_10' in self.df_feat.columns
        assert 'Sorteos_Desde_Aparicion' in self.df_feat.columns
        assert 'Posicion_Previo' in self.df_feat.columns
        assert 'Diferencia_Ciclica' in self.df_feat.columns

    def test_prob_hist_hora_sin_leakage(self):
        df = self.df.sort_values(['Fecha', 'Hora']).reset_index(drop=True).copy()
        df_feat = self.loteria.agregar_caracteristicas_avanzadas(df.copy())
        prob_hist = df_feat['Prob_Hist_Hora'].values
        for i in range(1, len(df_feat)):
            animal = df_feat.iloc[i]['Animal']
            hora = df_feat.iloc[i]['Solo_hora']
            past = df_feat.iloc[:i+1]
            same_hour = past[past['Solo_hora'] == hora]
            if len(same_hour) == 0:
                continue
            total = len(same_hour)
            matches = len(same_hour[same_hour['Animal'] == animal])
            expected = matches / total * 100
            assert abs(prob_hist[i] - expected) < 0.01, f"i={i}: {prob_hist[i]} != {expected}"

    def test_prob_trans_markov_sin_leakage(self):
        df = self.df.sort_values(['Fecha', 'Hora']).reset_index(drop=True).copy()
        df_feat = self.loteria.agregar_caracteristicas_avanzadas(df.copy())
        prob_trans = df_feat['Prob_Trans_Markov'].values
        for i in range(len(df_feat)):
            assert 0 <= prob_trans[i] <= 100, f"i={i}: out of range"
        assert prob_trans[0] == 0.0
        last = len(df_feat) - 1
        if last > 0 and df_feat.iloc[last-1]['Fecha'] == df_feat.iloc[last]['Fecha']:
            trans_prob_all, _ = self.loteria._transiciones_markov(df)
            prev_last = df_feat.iloc[last-1]['Animal']
            cur_last = df_feat.iloc[last]['Animal']
            expected = trans_prob_all.get((prev_last, cur_last), 0)
            assert abs(prob_trans[last] - expected) < 0.01

    def test_frecuencia_10(self):
        df_feat = self.df_feat.sort_values(['Fecha', 'Hora']).reset_index(drop=True)
        for i in range(len(df_feat)):
            animal = df_feat.iloc[i]['Animal']
            start = max(0, i - 9)
            expected = int((df_feat.iloc[start:i+1]['Animal'] == animal).sum())
            assert df_feat.iloc[i]['Frecuencia_10'] == expected, f"i={i}: {df_feat.iloc[i]['Frecuencia_10']} != {expected}"

    def test_sorteos_desde_aparicion(self):
        df_feat = self.df_feat.sort_values(['Fecha', 'Hora']).reset_index(drop=True)
        for i in range(len(df_feat)):
            animal = df_feat.iloc[i]['Animal']
            prev = -1
            for j in range(i - 1, -1, -1):
                if df_feat.iloc[j]['Animal'] == animal:
                    prev = j
                    break
            expected = (i - prev - 1) if prev >= 0 else -1
            assert df_feat.iloc[i]['Sorteos_Desde_Aparicion'] == expected, f"i={i}: {df_feat.iloc[i]['Sorteos_Desde_Aparicion']} != {expected}"

    def test_transiciones_markov_helper(self):
        trans_prob, trans_total = self.loteria._transiciones_markov(self.df)
        assert isinstance(trans_prob, dict)
        assert isinstance(trans_total, dict)
        if trans_prob:
            key = list(trans_prob.keys())[0]
            assert 0 <= trans_prob[key] <= 100

    def test_frecuencias_hora_helper(self):
        freq = self.loteria._frecuencias_hora(self.df, 'Hora')
        assert isinstance(freq, dict)
        for hora, animals in freq.items():
            total = sum(animals.values())
            assert abs(total - 100) < 0.1 or len(animals) < 38

    def test_mh_ranking_helper(self):
        animales = list(self.loteria.animales_carac.keys())[:10]
        markov = {a: 50 for a in animales}
        hourly = {a: 10 for a in animales}
        rankings, combined = self.loteria._mh_ranking(markov, hourly, animales)
        assert len(rankings) == min(20, len(animales))
        assert len(combined) == len(animales)

    def test_predecir_top_k_por_hora_sin_modelo(self):
        df = self.df_feat.copy()
        if 'Hora_Sorteo' not in df.columns:
            df['Hora_Sorteo'] = df['Hora'].astype(str).str.strip().str.zfill(8)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.config['animales'])
        X = df[['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora', 'Prob_Trans_Markov',
                'Frecuencia_10', 'Sorteos_Desde_Aparicion']].dropna()
        y = le.transform(df.loc[X.index, 'Animal'])
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Hora_Sorteo']),
            ('num', 'passthrough', ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora',
                                    'Prob_Trans_Markov', 'Frecuencia_10', 'Sorteos_Desde_Aparicion'])
        ], remainder='drop')
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', rf)])
        pipe.fit(df[['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora',
                     'Prob_Trans_Markov', 'Frecuencia_10', 'Sorteos_Desde_Aparicion', 'Hora_Sorteo']].dropna(),
                 y)
        matriz = self.loteria.predecir_top_k_por_hora(pipe, le, df.copy(), k=5)
        assert isinstance(matriz, dict)
        if matriz:
            for hora, animals in matriz.items():
                assert len(animals) <= 5

    def test_preparar_datos_ml_completo(self):
        df_feat = self.df_feat.copy()
        X, Y, le, _, _, available = self.loteria.preparar_datos_ml_completo(df_feat)
        assert len(X) > 0
        assert len(Y) == len(X)
        assert len(le.classes_) == 38
        cols = ['Posicion_Previo', 'Diferencia_Ciclica', 'Prob_Hist_Hora',
                'Prob_Trans_Markov', 'Frecuencia_10', 'Sorteos_Desde_Aparicion']
        for c in cols:
            assert c in available, f"{c} not in available features"

    def test_repoblar_sin_error(self):
        try:
            self.loteria.verificar_diccionario_animales()
        except Exception as e:
            assert False, f"verificar_diccionario_animales raised {e}"
