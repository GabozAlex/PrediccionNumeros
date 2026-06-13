import pandas as pd
import datetime

import lotto_activo


def test_generar_prediccion_markov_basic():
    analizador = lotto_activo.analizador
    # Build a small dataframe with repeated Fecha to create transitions
    fecha = datetime.date(2023, 1, 1)
    datos = pd.DataFrame([
        {'Fecha': fecha, 'Solo_hora': '08:00 AM', 'Num_Int': 1},
        {'Fecha': fecha, 'Solo_hora': '08:00 AM', 'Num_Int': 2},
        {'Fecha': fecha, 'Solo_hora': '09:00 AM', 'Num_Int': 2},
        {'Fecha': fecha, 'Solo_hora': '10:00 AM', 'Num_Int': 3},
    ])

    res = analizador.generar_prediccion_markov(datos, top_k=5)
    assert isinstance(res, dict)
    assert 'top' in res and 'por_hora' in res and 'ultimo' in res
    assert isinstance(res['top'], list)
    # top items contain expected keys
    if res['top']:
        item = res['top'][0]
        assert set(['rank', 'num', 'animal', 'score', 'markov', 'hora_pct']).issubset(item.keys())
    # por_hora is a dict mapping horas to lists
    assert isinstance(res['por_hora'], dict)
    for hora, lista in res['por_hora'].items():
        assert isinstance(hora, str)
        assert isinstance(lista, list)
