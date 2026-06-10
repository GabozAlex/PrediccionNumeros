import sys, time, datetime, logging, requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL_BASE = "https://loteriadehoy.com/animalito/lottoactivo/historico"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SALIDA = "data/LottoActivoHistorico.xlsx"

ANIMALES_38 = [
    "DELFIN", "BALLENA", "CARNERO", "TORO", "CIEMPIES", "ALACRAN",
    "LEON", "RANA", "PERICO", "RATON", "AGUILA", "TIGRE", "GATO",
    "CABALLO", "MONO", "PALOMA", "ZORRO", "OSO", "PAVO", "BURRO",
    "CHIVO", "COCHINO", "GALLO", "CAMELLO", "CEBRA", "IGUANA",
    "GALLINA", "VACA", "PERRO", "ZAMURO", "ELEFANTE", "CAIMAN",
    "LAPA", "ARDILLA", "PESCADO", "VENADO", "JIRAFA", "CULEBRA",
]

ANIMAL_A_NUMERO = {animal: i for i, animal in enumerate(ANIMALES_38)}

def scrape_semana(fecha_inicio):
    records = []
    inicio = fecha_inicio.strftime("%Y-%m-%d")
    fin = (fecha_inicio + datetime.timedelta(days=6)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(f"{URL_BASE}/{inicio}/{fin}/", headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return records
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table')
    if not table:
        return records
    rows = table.find_all('tr')
    if len(rows) < 2:
        return records
    fechas = [c.get_text(strip=True) for c in rows[0].find_all(['th','td'])[1:]]
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        hora = cells[0].get_text(strip=True)
        for ci, cell in enumerate(cells[1:], 1):
            if ci > len(fechas): break
            animal = cell.get_text(strip=True).upper()
            if animal and animal in ANIMAL_A_NUMERO:
                records.append({
                    "Fecha": fechas[ci-1],
                    "Hora": hora,
                    "Animal": animal,
                    "Numero": ANIMAL_A_NUMERO[animal],
                })
    return records

def scrapear_todo():
    hoy = datetime.date.today()
    todos = []
    d = datetime.date(2017, 6, 1)
    while d <= hoy:
        sem = scrape_semana(d)
        todos.extend(sem)
        if (d - datetime.date(2017, 6, 1)).days % 56 == 0:
            print(f"  {d} → {len(todos)} registros")
        d += datetime.timedelta(days=7)
        time.sleep(1.0)
    return todos

if __name__ == "__main__":
    print("Scrapeando Lotto Activo (histórico 2017-2026)...")
    todos = scrapear_todo()
    if todos:
        df = pd.DataFrame(todos).drop_duplicates(subset=["Fecha","Hora"], keep="last")
        df = df.sort_values(["Fecha","Hora"]).reset_index(drop=True)
        df.to_excel(SALIDA, index=False)
        print(f"\nGuardados {len(df)} registros en {SALIDA}")
        print(f"Rango: {df['Fecha'].min()} → {df['Fecha'].max()}")
        print(f"Fechas únicas: {df['Fecha'].nunique()}")
    else:
        print("Sin registros")
