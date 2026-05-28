import sys
import time
import datetime
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL_BASE = "https://loteriadehoy.com/terminal/terminaltrio/historico"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SALIDA = "TerminalTrio.xlsx"

def scrape_semana(fecha_inicio):
    records = []
    inicio = fecha_inicio.strftime("%Y-%m-%d")
    fin = (fecha_inicio + datetime.timedelta(days=6)).strftime("%Y-%m-%d")
    url = f"{URL_BASE}/{inicio}/{fin}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logging.warning(f"Error {inicio}: {e}")
        return records
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table')
    if not table:
        return records
    rows = table.find_all('tr')
    if len(rows) < 2:
        return records
    headers = [c.get_text(strip=True) for c in rows[0].find_all(['th','td'])]
    fechas = headers[1:]
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        hora = cells[0].get_text(strip=True)
        for ci, cell in enumerate(cells[1:], 1):
            if ci > len(fechas):
                break
            num = cell.get_text(strip=True)
            if num and num.isdigit() and len(num) == 2:
                records.append({
                    "Fecha": fechas[ci-1],
                    "Hora": hora,
                    "Numero": int(num),
                })
    return records

def scrapear_todo():
    hoy = datetime.date.today()
    inicio = datetime.date(2023, 10, 1)
    todos = []
    d = inicio
    while d <= hoy:
        sem = scrape_semana(d)
        todos.extend(sem)
        if (d - inicio).days % 28 == 0:
            print(f"  {d} → {len(todos)} registros")
        d += datetime.timedelta(days=7)
        time.sleep(1.2)
    return todos

def guardar(records):
    if not records:
        print("Sin registros")
        return
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["Fecha", "Hora"], keep="last")
    df = df.sort_values(["Fecha", "Hora"]).reset_index(drop=True)
    df.to_excel(SALIDA, index=False)
    print(f"Guardados {len(df)} registros en {SALIDA}")

if __name__ == "__main__":
    print("Scrapeando Terminal Trio desde 2023-10-01...")
    todos = scrapear_todo()
    guardar(todos)
