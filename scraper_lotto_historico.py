import sys, time, datetime, logging, requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL_BASE = "https://loteriadehoy.com/animalito/lottoactivo/historico"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SALIDA = "LottoActivoHistorico.xlsx"

# Full animal → number mapping from our existing data
ANIMAL_A_NUMERO = {
    "BALLENA":0,"DELFIN":0,"TORO":2,"CIEMPIES":3,"ALACRAN":4,"LEON":5,"RANA":6,"PERICO":7,
    "RATON":8,"AGUILA":9,"TIGRE":10,"GATO":11,"CABALLO":12,"PALOMA":13,"ZORRO":15,"OSO":16,
    "PAVO":17,"BURRO":18,"CABRA":19,"COCHINO":20,"GALLO":21,"CAMELLO":22,"CEBRA":23,
    "IGUANA":24,"GALLINA":25,"VACA":26,"PERRO":27,"ZAMURO":28,"ELEFANTE":29,"CAIMAN":30,
    "LAPA":31,"ARDILLA":32,"PESCADO":33,"VENADO":34,"JIRAFA":35,"CULEBRA":36,"CARNERO":1
}

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
