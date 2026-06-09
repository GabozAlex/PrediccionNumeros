import sys
import os
import io
import contextlib
import threading
import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import lotto_activo
import la_granjita
import selva_plus
import lotto_rd_int

LOTTERY_MODULES = {
    "Lotto Activo": lotto_activo,
    "La Granjita": la_granjita,
    "Selva Plus": selva_plus,
    "Lotto Activo Rd Int": lotto_rd_int,
}


class RedirectText:
    """Thread-safe redirect for stdout/stderr into a Tkinter Text widget.

    It schedules UI updates on the Tkinter mainloop via `after(0, ...)` so
    background threads can safely write to the widget.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, s):
        if not s:
            return

        def append():
            try:
                # widget operations must run on the main thread
                self.text_widget.insert(tk.END, s)
                self.text_widget.see(tk.END)
            except Exception:
                # widget may be destroyed while a background thread is still
                # producing output; ignore safely
                pass

        try:
            # schedule append on Tk mainloop
            self.text_widget.after(0, append)
        except Exception:
            # fallback: if scheduling fails, print to stdout
            print(s, end='')

    def flush(self):
        # required for file-like interface; no-op
        pass


class LottoPredictorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediccion Numeros - Multi Loteria")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        style = ttk.Style()
        style.theme_use("clam")

        self.datos = {}
        self.modelos_rf = {}
        self.le_y_rf = {}
        self.modelos_xgb = {}
        self.le_y_xgb = {}
        self.current_lottery = "Lotto Activo"
        self._paneles = {}

        self._build_ui()
        self._cargar_loteria("Lotto Activo")
        self._auto_cargar_modelos()

    def _get_mod(self):
        return LOTTERY_MODULES.get(self.current_lottery)

    def _get_analizador(self):
        mod = self._get_mod()
        return mod.analizador if mod else None

    def _get_config(self):
        mod = self._get_mod()
        return mod.CONFIG if mod else None

    def _build_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Loteria:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=2)
        self._loteria_cb = ttk.Combobox(
            top_frame, values=list(LOTTERY_MODULES.keys()), state="readonly", width=20
        )
        self._loteria_cb.set(self.current_lottery)
        self._loteria_cb.pack(side=tk.LEFT, padx=2)
        self._loteria_cb.bind("<<ComboboxSelected>>", self._on_loteria_change)

        self._status_lbl = ttk.Label(top_frame, text="", font=("", 9))
        self._status_lbl.pack(side=tk.LEFT, padx=10)

        ttk.Button(top_frame, text="Recargar Datos", command=self._recargar_datos).pack(side=tk.RIGHT, padx=2)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._tab_crear_dashboard()
        self._tab_crear_prediccion()
        self._tab_crear_modelos()
        self._tab_crear_scraper()

    def _on_loteria_change(self, event=None):
        self.current_lottery = self._loteria_cb.get()
        if self.current_lottery not in self.datos:
            self._cargar_loteria(self.current_lottery)
        self._actualizar_status()

    def _cargar_loteria(self, nombre):
        mod = LOTTERY_MODULES.get(nombre)
        if not mod:
            return
        config = mod.CONFIG
        excel_file = config['excel_file']
        analizador = mod.analizador

        if not os.path.exists(excel_file):
            print(f"AVISO: Archivo no encontrado: {excel_file}")
            print(f"Usa la pestana 'Web Scraper' -> 'Scrapear Faltantes' para descargar datos de {nombre}")
            self.datos[nombre] = pd.DataFrame()
            return

        try:
            datos = pd.read_excel(excel_file)
            datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
            datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce')
            datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
            datos['Hora'] = datos['Hora'].astype(str).str.strip().str.zfill(8)
            datos['Timestamp'] = pd.to_datetime(
                datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce'
            )
            datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
            datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip()
            datos = datos.sort_values(by='Timestamp').reset_index(drop=True)
            datos = analizador.agregar_caracteristicas_avanzadas(datos)
            self.datos[nombre] = datos
            print(f"Datos cargados: {nombre} - {len(datos)} registros")
        except Exception as e:
            print(f"Error cargando {nombre}: {e}")
            self.datos[nombre] = pd.DataFrame()

    def _auto_cargar_modelos(self):
        def tarea():
            for nombre, mod in LOTTERY_MODULES.items():
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    rf, le_rf, _ = mod.cargar_ultimo_modelo("random_forest")
                    xgb, le_xgb, _ = mod.cargar_ultimo_modelo("xgboost")
                if rf:
                    self.modelos_rf[nombre] = rf
                    self.le_y_rf[nombre] = le_rf
                if xgb:
                    self.modelos_xgb[nombre] = xgb
                    self.le_y_xgb[nombre] = le_xgb
            self.root.after(0, self._actualizar_status)
        hilo = threading.Thread(target=tarea, daemon=True)
        hilo.start()

    def _actualizar_status(self):
        datos = self.datos.get(self.current_lottery)
        if datos is not None and not datos.empty:
            self._status_lbl.config(
                text=f"Registros: {len(datos)} | Fechas: {datos['Fecha'].nunique()} | "
                     f"{datos['Fecha'].min()} -> {datos['Fecha'].max()}"
            )
        else:
            self._status_lbl.config(text="Sin datos")

    def _get_datos(self):
        return self.datos.get(self.current_lottery)

    def _crear_paned_grid(self, tab):
        """Reemplaza grid 2x2 con PanedWindow arrastrable. Retorna [[(0,0),(0,1)],[(1,0),(1,1)]]"""
        outer = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left = ttk.PanedWindow(outer, orient=tk.VERTICAL)
        right = ttk.PanedWindow(outer, orient=tk.VERTICAL)
        outer.add(left, weight=1)
        outer.add(right, weight=1)
        tl = ttk.Frame(left, relief=tk.RIDGE, borderwidth=1)
        bl = ttk.Frame(left, relief=tk.RIDGE, borderwidth=1)
        tr = ttk.Frame(right, relief=tk.RIDGE, borderwidth=1)
        br = ttk.Frame(right, relief=tk.RIDGE, borderwidth=1)
        left.add(tl, weight=1)
        left.add(bl, weight=1)
        right.add(tr, weight=1)
        right.add(br, weight=1)
        return [[tl, tr], [bl, br]]

    def _agregar_panel_salida(self, parent, label, name, row, column, columnspan=1):
        frame = parent[row][column]
        ttk.Label(frame, text=label, font=("", 8, "bold")).pack(anchor=tk.W, padx=2)
        text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Courier", 8))
        text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self._paneles[name] = text
        return text

    def _ejecutar_en_hilo(self, func, text_widget, *args):
        def tarea():
            self.root.after(0, lambda: (text_widget.delete("1.0", tk.END), text_widget.insert(tk.END, f"{'='*70}\n")))
            redir = RedirectText(text_widget)
            with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
                try:
                    func(*args)
                except Exception as e:
                    print(f"Error: {e}")
        hilo = threading.Thread(target=tarea, daemon=True)
        hilo.start()

    # ================ TAB: DASHBOARD ================

    def _tab_crear_dashboard(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dashboard")

        frame_btn = ttk.Frame(tab)
        frame_btn.pack(fill=tk.X, padx=5, pady=5)

        btns = [
            ("Estado Rapido del Dia", "dash_estado"),
            ("Ultimos Registros", "dash_ultimos"),
            ("Aciertos x Dia Semana", "dash_aciertos_dia"),
            ("Aciertos x Hora", "dash_aciertos_hora"),
        ]
        for texto, accion in btns:
            btn = ttk.Button(frame_btn, text=texto, command=lambda a=accion: self._dashboard_accion(a))
            btn.pack(side=tk.LEFT, padx=2)

        grid = self._crear_paned_grid(tab)

        self._agregar_panel_salida(grid, "Estado Rapido del Dia", "dash_estado", 0, 0)
        self._agregar_panel_salida(grid, "Ultimos Registros", "dash_ultimos", 0, 1)
        self._agregar_panel_salida(grid, "Aciertos x Dia Semana", "dash_aciertos_dia", 1, 0)
        self._agregar_panel_salida(grid, "Aciertos x Hora", "dash_aciertos_hora", 1, 1)

        self.root.after(500, lambda: self._dashboard_accion("dash_estado"))

    def _dashboard_accion(self, accion):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        d = datos.copy()
        analizador = self._get_analizador()
        if not analizador:
            return
        panel = self._paneles.get(accion)
        if panel is None:
            return
        mapa = {
            "dash_estado": lambda: analizador.ver_estado_actual_dia(d),
            "dash_ultimos": lambda: analizador.ver_ultimos_registros_y_faltantes(d),
            "dash_aciertos_dia": lambda: analizador.analizar_aciertos_por_dia_semana(d),
            "dash_aciertos_hora": lambda: analizador.analizar_aciertos_por_hora(d),
        }
        func = mapa.get(accion)
        if func:
            self._ejecutar_en_hilo(func, panel)

    # ================ TAB: PREDICCION ================

    def _tab_crear_prediccion(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Prediccion")

        frame_btn = ttk.Frame(tab)
        frame_btn.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(frame_btn, text="Markov", command=self._dialogo_markov).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btn, text="Prob. x Hora", command=self._dialogo_prob_hora).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btn, text="Markov x Hora", command=self._dialogo_markov_hora).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btn, text="Global x Hora", command=self._ventana_matriz_combinada).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btn, text="Predecir Siguiente (M+H)", command=self._predecir_siguiente_mh_dialogo).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btn, text="Top-25 General", command=lambda: self._panel_top25()).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btn, text="Patrones", command=self._dialogo_patrones).pack(side=tk.LEFT, padx=2)

    def _panel_top25(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        panel = self._paneles.get("pred_top28")
        if panel:
            self._ejecutar_en_hilo(lambda: analizador.top_25_general(datos.copy()), panel)

    def _evaluar_precision(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Evaluacion Precision - {self.current_lottery}")
        ventana.geometry("600x500")
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert(tk.END, "Evaluando precision en ultimos 1000 sorteos...\n")
        def tarea():
            import pandas as pd
            from collections import defaultdict
            d = datos.copy().sort_values(['Fecha','Hora']).reset_index(drop=True)
            corte = len(d) - 1000
            train = d.iloc[:corte].copy()
            test = d.iloc[corte:].copy()
            dp = analizador.preparar_datos_markov(train)
            tg, tot, th, toh = analizador.construir_matrices_markov(dp, incluir_trasnocho=False)
            def eval_k(k):
                ag=ah=uni=total=0
                for i in range(len(test)-1):
                    if test.iloc[i]['Fecha'] != test.iloc[i+1]['Fecha']:
                        continue
                    ant, hp = test.iloc[i]['Animal'], test.iloc[i]['Hora']
                    sig, hn = test.iloc[i+1]['Animal'], test.iloc[i+1]['Hora']
                    par = (hp, hn)
                    total += 1
                    g_ok = ant in tot and tot[ant] > 0 and sig in [a for a,c in sorted(tg[ant].items(), key=lambda x:-x[1])[:k]]
                    h_ok = par in th and ant in toh[par] and toh[par][ant] > 0 and sig in [a for a,c in sorted(th[par][ant].items(), key=lambda x:-x[1])[:k]]
                    if g_ok: ag += 1
                    if h_ok: ah += 1
                    if g_ok or h_ok: uni += 1
                return ag/total*100, ah/total*100, uni/total*100
            texto = f"Evaluacion Precision (train: {len(train)}, test: {len(test)})\n"
            texto += "=" * 60 + "\n\n"
            texto += f"{'Top-K':>6} {'Global':>8} {'xHora':>8} {'Union':>8} {'Azar':>8}\n"
            texto += '-' * 40 + '\n'
            for k in [1, 3, 5, 10, 25]:
                g, h, u = eval_k(k)
                texto += f"Top-{k:<2}  {g:>6.1f}% {h:>6.1f}% {u:>6.1f}% {k/38*100:>5.1f}%\n"
            texto += "\nUnion = acierta si aparece en Global O xHora\n"
            texto += "Azar  = probabilidad al azar (k/38)\n"
            ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
        threading.Thread(target=tarea, daemon=True).start()

    def _evaluar_por_animal(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Precision x Animal - {self.current_lottery}")
        ventana.geometry("700x600")
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert(tk.END, "Evaluando precision por animal...\n")
        def tarea():
            import pandas as pd
            d = datos.copy().sort_values(['Fecha','Hora']).reset_index(drop=True)
            corte = len(d) - 1000
            train = d.iloc[:corte].copy()
            test = d.iloc[corte:].copy()
            dp = analizador.preparar_datos_markov(train)
            tg, tot, _, _ = analizador.construir_matrices_markov(dp, incluir_trasnocho=False)
            ac_g = {}
            ac_h = {}
            total_animal = {}
            for i in range(len(test)-1):
                if test.iloc[i]['Fecha'] != test.iloc[i+1]['Fecha']:
                    continue
                ant, hp = test.iloc[i]['Animal'], test.iloc[i]['Hora']
                sig = test.iloc[i+1]['Animal']
                if ant not in total_animal:
                    total_animal[ant] = 0
                    ac_g[ant] = 0
                    ac_h[ant] = 0
                total_animal[ant] += 1
                if ant in tot and tot[ant] > 0:
                    if sig in [a for a,c in sorted(tg[ant].items(), key=lambda x:-x[1])[:25]]:
                        ac_g[ant] += 1
            # Sort by worst accuracy first
            ranking = sorted([(a, ac_g.get(a,0)/total_animal[a]*100 if total_animal[a] else 0, total_animal[a]) for a in total_animal], key=lambda x: x[1])
            texto = f"Precision Global Top-25 por Animal (ultimos 1000 sorteos)\n"
            texto += "=" * 60 + "\n\n"
            texto += f"{'Animal':<14} {'Aciertos':>8} {'Total':>6} {'%':>6}\n"
            texto += '-' * 40 + '\n'
            for a, pct, tot in ranking:
                barra = '#' * max(1, int(pct/4))
                texto += f"  {a:<14} {ac_g[a]:>3}/{total_animal[a]:<2} {pct:>5.1f}% {barra}\n"
            texto += f"\nPeores 5: {', '.join(a for a,_,_ in ranking[:5])}"
            texto += f"\nMejores 5: {', '.join(a for a,_,_ in ranking[-5:])}"
            ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
        threading.Thread(target=tarea, daemon=True).start()

    def _dialogo_backtesting(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Backtesting - {self.current_lottery}")
        ventana.geometry("750x650")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_animal = ttk.Entry(frame_top, width=14, font=("", 10))
        entry_animal.pack(side=tk.LEFT, padx=2)
        entry_animal.focus_set()
        ttk.Label(frame_top, text="Desde:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_desde = ttk.Entry(frame_top, width=12, font=("", 10))
        entry_desde.pack(side=tk.LEFT, padx=2)
        entry_desde.insert(0, "2024-06-03")
        ttk.Label(frame_top, text="Hasta:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_hasta = ttk.Entry(frame_top, width=12, font=("", 10))
        entry_hasta.pack(side=tk.LEFT, padx=2)
        entry_hasta.insert(0, "2024-06-09")
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def analizar():
            animal = entry_animal.get().strip().upper()
            desde = entry_desde.get().strip()
            hasta = entry_hasta.get().strip()
            if not animal or not desde or not hasta:
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Analizando {animal} del {desde} al {hasta}...\n")
            def tarea():
                import pandas as pd
                d = datos.copy()
                d['Fecha'] = pd.to_datetime(d['Fecha']).dt.date
                d = d.sort_values(['Fecha','Hora']).reset_index(drop=True)
                dp = analizador.preparar_datos_markov(d)
                tg, tot, _, _ = analizador.construir_matrices_markov(dp, incluir_trasnocho=False)
                desde_dt = pd.to_datetime(desde).date()
                hasta_dt = pd.to_datetime(hasta).date()
                sub = d[(d['Fecha'] >= desde_dt) & (d['Fecha'] <= hasta_dt)].copy()
                hits = 0
                total = 0
                texto = f"Backtesting: {animal} del {desde} al {hasta}\n"
                texto += "=" * 70 + "\n\n"
                for i in range(len(sub)-1):
                    if sub.iloc[i]['Animal'] != animal:
                        continue
                    if sub.iloc[i]['Fecha'] == sub.iloc[i+1]['Fecha']:
                        total += 1
                        f = sub.iloc[i]['Fecha']
                        h = sub.iloc[i]['Hora']
                        sig = sub.iloc[i+1]['Animal']
                        preds = []
                        if animal in tot and tot[animal] > 0:
                            preds = [a for a,c in sorted(tg[animal].items(), key=lambda x:-x[1])[:25]]
                        rank = preds.index(sig) + 1 if sig in preds else 0
                        if rank:
                            hits += 1
                            marca = "✅"
                        else:
                            marca = "❌"
                        texto += f"  {f} {h}  {animal:<12} -> {sig:<12}  puesto #{rank if rank else '—'}{'  ' + marca if rank else ''}\n"
                if total:
                    texto += f"\nResumen: {hits}/{total} aciertos ({hits/total*100:.1f}%) en Top-25\n"
                else:
                    texto += "\nNo se encontraron ocurrencias de este animal en el rango\n"
                # Show top predictions for reference
                preds_list = []
                if animal in tot and tot[animal] > 0:
                    preds_list = [(a, c/tot[animal]*100) for a,c in sorted(tg[animal].items(), key=lambda x:-x[1])[:25]]
                texto += f"\nTop-25 predictivo para {animal}:\n"
                texto += '-' * 40 + '\n'
                for i, (a, p) in enumerate(preds_list, 1):
                    texto += f"  {i:2d}. {a:<14} ({p:.1f}%)\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        entry_animal.bind("<Return>", lambda e: analizar())
        ttk.Button(frame_top, text="Analizar", command=analizar).pack(side=tk.LEFT, padx=5)

    def _dialogo_cadena(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Cadena del Dia - {self.current_lottery}")
        ventana.geometry("700x600")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal inicial:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_animal = ttk.Entry(frame_top, width=15, font=("", 10))
        entry_animal.pack(side=tk.LEFT, padx=2)
        entry_animal.focus_set()
        ttk.Label(frame_top, text="Hora inicio:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        horas = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM"]
        combo = ttk.Combobox(frame_top, values=horas, state="readonly", width=12, font=("", 10))
        combo.pack(side=tk.LEFT, padx=2)
        combo.set(horas[0])
        trasnocho_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_top, text="Trasnocho", variable=trasnocho_var).pack(side=tk.LEFT, padx=5)
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def simular():
            animal = entry_animal.get().strip().upper()
            hora_str = combo.get()
            if not animal or not hora_str:
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Simulando cadena desde {animal} a las {hora_str}...\n")
            def tarea():
                dt = pd.to_datetime(hora_str, format="%I:%M %p")
                h_actual = dt.strftime("%H:%M:%S")
                parejas = analizador.get_parejas_horarias()
                if trasnocho_var.get():
                    parejas = list(parejas) + [('19:00:00','08:00:00')]
                # Find starting index in parejas
                idx = 0
                for i, (o, d) in enumerate(parejas):
                    if o == h_actual:
                        idx = i
                        break
                d = datos.copy()
                texto = f"Cadena del dia desde {animal} ({hora_str})\n"
                texto += "=" * 60 + "\n\n"
                texto += f"{'Hora':>8}  {'Actual':<14}  {'Top-5 siguientes':<55}\n"
                texto += '-' * 80 + '\n'
                animal_act = animal
                for i in range(idx, len(parejas)):
                    o, dest = parejas[i]
                    h_12 = pd.to_datetime(o, format='%H:%M:%S').strftime('%I:%M %p')
                    preds = analizador.get_prediccion_combinada(d, animal_act, o, dest, top_k=5, incluir_trasnocho=trasnocho_var.get())
                    top5 = [f"{a}({p:.0f}%)" for a, p, _ in preds[:5]] if preds else ["(sin datos)"]
                    texto += f"  {h_12:<8}  {animal_act:<14}  {' / '.join(top5):<55}\n"
                    animal_act = preds[0][0] if preds else animal_act
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        entry_animal.bind("<Return>", lambda e: simular())
        ttk.Button(frame_top, text="Simular", command=simular).pack(side=tk.LEFT, padx=5)

    def _dialogo_segundo_orden(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Segundo Orden - {self.current_lottery}")
        ventana.geometry("550x520")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal previo 1:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry1 = ttk.Entry(frame_top, width=12, font=("", 10))
        entry1.pack(side=tk.LEFT, padx=2)
        ttk.Label(frame_top, text="Animal previo 2:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry2 = ttk.Entry(frame_top, width=12, font=("", 10))
        entry2.pack(side=tk.LEFT, padx=2)
        entry2.focus_set()
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            a1 = entry1.get().strip().upper()
            a2 = entry2.get().strip().upper()
            if not a1 or not a2:
                messagebox.showwarning("Error", "Ingresa ambos animales")
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Buscando siguientes de ({a1}, {a2})...\n")
            def tarea():
                items = analizador.get_matriz_segundo_orden(datos.copy(), a1, a2, top_k=25)
                total_m = sum(c for _, _, c in items)
                texto = f"Animales previos: {a1} -> {a2}  |  total pares: {total_m}\n\n"
                texto += f"{'#':>3} {'Animal':<14} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 35 + "\n"
                if not items:
                    texto += "(sin datos para este par)\n"
                for i, (a3, p, c) in enumerate(items, 1):
                    texto += f"  {i:2d} {a3:<14} {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        entry1.bind("<Return>", lambda e: entry2.focus_set())
        entry2.bind("<Return>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_patrones(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Patrones - {self.current_lottery}")
        ventana.geometry("750x650")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)

        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        def ejecutar(accion):
            d = datos.copy()
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Analizando patrones...\n")
            def tarea():
                redir = RedirectText(txt)
                with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
                    if accion == "coocurrencias":
                        analizador.analizar_coocurrencias(d)
                    elif accion == "rango":
                        analizador.analizar_coocurrencias_por_rango(d)
                    elif accion == "dia_semana":
                        analizador.analizar_frecuencia_por_dia_semana(d)
                    ventana.after(0, lambda: txt.see(tk.END))
            threading.Thread(target=tarea, daemon=True).start()

        ttk.Button(frame_top, text="Co-ocurrencias", command=lambda: ejecutar("coocurrencias")).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Rango Horario", command=lambda: ejecutar("rango")).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Dia Semana", command=lambda: ejecutar("dia_semana")).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Cadena Dia", command=self._dialogo_cadena).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="2do Orden", command=self._dialogo_segundo_orden).pack(side=tk.LEFT, padx=2)

    def _dialogo_markov(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Markov - Buscar Animal")
        ventana.geometry("550x520")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal actual:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry = ttk.Entry(frame_top, width=20, font=("", 10))
        entry.pack(side=tk.LEFT, padx=2)
        entry.focus_set()
        trasnocho_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_top, text="7PM→8AM (trasnocho)", variable=trasnocho_var).pack(side=tk.LEFT, padx=5)
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            animal = entry.get().strip().upper()
            if not animal:
                messagebox.showwarning("Error", "Ingresa un animal")
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Buscando siguientes de {animal}...\n")
            def tarea():
                matriz = analizador.get_matriz_global_por_animal(datos.copy(), top_k=25, incluir_trasnocho=trasnocho_var.get())
                items = matriz.get(animal, [])
                total_m = sum(c for _, _, c in items)
                texto = f"Animal actual: {animal}  |  total muestras: {total_m}"
                if trasnocho_var.get():
                    texto += " (con trasnocho 7PM→8AM)"
                texto += f"\n\nTop siguientes:\n"
                texto += f"{'#':>3} {'Animal':<14} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 35 + "\n"
                if not items:
                    texto += "(sin datos para este animal)\n"
                for i, (a2, p, c) in enumerate(items, 1):
                    texto += f"  {i:2d} {a2:<14} {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        entry.bind("<Return>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_prob_hora(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Probabilidad por Hora")
        ventana.geometry("550x500")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Hora:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        horas = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo = ttk.Combobox(frame_top, values=horas, state="readonly", width=15, font=("", 10))
        combo.pack(side=tk.LEFT, padx=2)
        combo.set(horas[-1])
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            hora_str = combo.get()
            if not hora_str:
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Calculando probabilidades para {hora_str}...\n")
            def tarea():
                d = datos.copy()
                d_filtro = d[d["Solo_hora"] == hora_str]
                if d_filtro.empty:
                    ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, f"Sin datos para {hora_str}\n")))
                    return
                total = len(d_filtro)
                conteo = d_filtro["Animal"].value_counts()
                texto = f"Probabilidades para {hora_str} (total: {total} sorteos)\n"
                texto += "=" * 50 + "\n"
                for animal, cnt in conteo.head(25).items():
                    pct = cnt / total * 100
                    texto += f"  {animal:<14} {cnt:4d} ({pct:.1f}%)\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        combo.bind("<<ComboboxSelected>>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_markov_hora(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Markov x Hora")
        ventana.geometry("550x520")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_animal = ttk.Entry(frame_top, width=15, font=("", 10))
        entry_animal.pack(side=tk.LEFT, padx=2)
        entry_animal.focus_set()
        ttk.Label(frame_top, text="Hora:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        horas = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo = ttk.Combobox(frame_top, values=horas, state="readonly", width=12, font=("", 10))
        combo.pack(side=tk.LEFT, padx=2)
        combo.set(horas[-1])
        trasnocho_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_top, text="7PM→8AM (trasnocho)", variable=trasnocho_var).pack(side=tk.LEFT, padx=5)
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            animal = entry_animal.get().strip().upper()
            hora_str = combo.get()
            if not animal or not hora_str:
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Calculando...\n")
            def tarea():
                dt = pd.to_datetime(hora_str, format="%I:%M %p")
                solo_hora = dt.strftime("%H:%M:%S")
                incluir_trasnocho = trasnocho_var.get()
                h_dest = None
                if incluir_trasnocho and solo_hora == '19:00:00':
                    h_dest = '08:00:00'
                else:
                    for o, d in analizador.get_parejas_horarias():
                        if o == solo_hora:
                            h_dest = d
                            break
                if h_dest is None:
                    ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, f"No hay transicion para {hora_str}\n")))
                    return
                matriz = analizador.get_matriz_hora_por_animal(datos.copy(), solo_hora, h_dest, top_k=25, incluir_trasnocho=incluir_trasnocho)
                items = matriz.get(animal, [])
                texto = f"Animal: {animal}  |  {hora_str} -> {pd.to_datetime(h_dest, format='%H:%M:%S').strftime('%I:%M %p')}"
                if incluir_trasnocho:
                    texto += " (con trasnocho)"
                total_m = sum(c for _, _, c in items)
                texto += f"\n" + "=" * 50 + f"\nTotal muestras: {total_m}\n\nTop siguientes:\n"
                texto += f"{'#':>3} {'Animal':<14} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 35 + "\n"
                if not items:
                    texto += "(sin datos para este animal a esta hora)\n"
                for i, (a2, p, c) in enumerate(items, 1):
                    texto += f"  {i:2d} {a2:<14} {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        entry_animal.bind("<Return>", lambda e: buscar())
        combo.bind("<<ComboboxSelected>>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _ventana_matriz_combinada(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        parejas = analizador.get_parejas_horarias()
        opciones = [f"{o} -> {d}" for o, d in parejas] + ['19:00:00 -> 08:00:00']
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Global x Hora - {self.current_lottery}")
        ventana.geometry("600x520")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_animal = ttk.Entry(frame_top, width=14, font=("", 10))
        entry_animal.pack(side=tk.LEFT, padx=2)
        entry_animal.focus_set()
        trasnocho_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_top, text="Trasnocho", variable=trasnocho_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_top, text="Hora:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        combo = ttk.Combobox(frame_top, values=opciones, state="readonly", width=22)
        combo.pack(side=tk.LEFT, padx=2)
        combo.set(opciones[-1])
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert(tk.END, "Selecciona animal y hora, luego clic en Buscar\n")
        frame_bot = ttk.Frame(ventana)
        frame_bot.pack(fill=tk.X, padx=5, pady=5)
        def buscar():
            animal = entry_animal.get().strip().upper()
            seleccion = combo.get()
            if not animal or not seleccion:
                messagebox.showwarning("Error", "Ingresa animal y selecciona hora")
                return
            partes = seleccion.split(" -> ")
            h_o, h_d = partes[0].strip(), partes[1].strip()
            incluir_trasnocho = trasnocho_var.get()
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Calculando {animal} para {h_o} -> {h_d}...\n")
            def tarea():
                d = datos.copy()
                mat_g = analizador.get_matriz_global_por_animal(d, top_k=25, incluir_trasnocho=incluir_trasnocho)
                mat_h = analizador.get_matriz_hora_por_animal(d, h_o, h_d, top_k=25, incluir_trasnocho=incluir_trasnocho)
                items_g = {a: (p, c) for a, p, c in mat_g.get(animal, [])}
                items_h = {a: (p, c) for a, p, c in mat_h.get(animal, [])}
                todos = set(items_g.keys()) | set(items_h.keys())
                m_h = sum(c for _, c in items_h.values())
                m_g = sum(c for _, c in items_g.values())
                texto = f"Animal: {animal}  |  {h_o} -> {h_d}"
                if incluir_trasnocho:
                    texto += " (trasnocho)"
                w_h = min(0.9, max(0.1, m_h / 50))
                w_g = 1 - w_h
                texto += f"  |  G:{m_g} H:{m_h} muestras  pesos G:{w_g:.0%} H:{w_h:.0%}"
                texto += "\n" + "=" * 55 + "\n\n"
                if not todos:
                    texto += "(sin datos)\n"
                else:
                    combinado = {a: items_g.get(a, (0,0))[0]*w_g + items_h.get(a, (0,0))[0]*w_h for a in todos}
                    top = sorted(combinado.items(), key=lambda x: -x[1])[:25]
                    texto += f"{'#':>3} {'Animal':<14} {'Gral':>5} {'(n)':>4} {'Hora':>5} {'(n)':>4} {'Comb':>5}\n"
                    texto += '-' * 45 + '\n'
                    for i, (a2, p) in enumerate(top, 1):
                        pg, cg = items_g.get(a2, (0, 0))
                        ph, ch = items_h.get(a2, (0, 0))
                        texto += f"  {i:2d} {a2:<14} {pg:>4.1f}% {cg:>3} {ph:>4.1f}% {ch:>3} {p:>4.1f}%\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=tarea, daemon=True).start()
        entry_animal.bind("<Return>", lambda e: buscar())
        btn_buscar = ttk.Button(frame_bot, text="Buscar", command=buscar)
        btn_buscar.pack()

    def _predecir_siguiente_mh_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Predecir Siguiente (M+H)")
        ventana.geometry("380x270")
        ventana.transient(self.root)
        ventana.grab_set()
        ttk.Label(ventana, text="Animal que acaba de salir:", font=("", 10)).pack(pady=(15, 3))
        entry_animal = ttk.Entry(ventana, width=25, font=("", 10))
        entry_animal.pack(pady=3)
        entry_animal.focus_set()
        ttk.Label(ventana, text="Hora del sorteo (HH:MM AM/PM):", font=("", 10)).pack(pady=(12, 3))
        horas_opciones = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                          "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                          "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo_hora = ttk.Combobox(ventana, values=horas_opciones, state="readonly", width=15, font=("", 10))
        combo_hora.pack(pady=3)
        combo_hora.set(horas_opciones[-1])
        def ejecutar():
            animal = entry_animal.get().strip().upper()
            hora = combo_hora.get().strip()
            if not animal:
                messagebox.showwarning("Error", "Debes ingresar un animal")
                return
            if not hora:
                messagebox.showwarning("Error", "Debes seleccionar una hora")
                return
            ventana.destroy()
            hilo = threading.Thread(
                target=self._predecir_siguiente_mh_task, args=(animal, hora), daemon=True
            )
            hilo.start()
        ttk.Button(ventana, text="Predecir", command=ejecutar).pack(pady=15)

    def _predecir_siguiente_mh_task(self, animal, hora_str):
        panel = self._paneles.get("pred_sig_mh")
        if panel is None:
            return
        redir = RedirectText(panel)
        self.root.after(0, lambda: panel.delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            datos = self._get_datos()
            if datos is None or datos.empty:
                print("ERROR: Sin datos")
                return
            analizador = self._get_analizador()
            if not analizador:
                return
            d = datos.copy()
            if animal not in analizador.animales_carac:
                animales_ok = ", ".join(sorted(analizador.animales_carac.keys()))
                print(f"ERROR: Animal '{animal}' no valido. Validos: {animales_ok}")
                return
            # Convert 12h hour to 24h format
            try:
                dt_hora = pd.to_datetime(hora_str, format='%I:%M %p')
                hora_24h = dt_hora.strftime('%H:%M:%S')
                solo_hora = dt_hora.strftime('%I:%M %p')
            except Exception:
                print(f"ERROR: Hora '{hora_str}' no valida. Usa formato HH:MM AM/PM")
                return
            # Override the last row with the user's animal and hour
            d.iloc[-1, d.columns.get_loc('Animal')] = animal
            d.iloc[-1, d.columns.get_loc('Numero')] = 0
            d.iloc[-1, d.columns.get_loc('Hora')] = hora_24h
            d.iloc[-1, d.columns.get_loc('Solo_hora')] = solo_hora
            analizador.prediccion_markov_hora(d)

    def _predecir_siguiente_ml_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        rf, le_rf, xgb, le_xgb = self._get_modelos()
        if not rf and not xgb:
            messagebox.showwarning("Sin modelos", "No hay modelos entrenados. Entrena RF o XGB primero.")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ultimo = datos.iloc[-1]
        animal_real = ultimo['Animal']
        hora_real = ultimo['Hora']
        try:
            hora_12h = pd.to_datetime(hora_real, format='%H:%M:%S').strftime('%I:%M %p').lstrip('0')
        except Exception:
            hora_12h = hora_real

        ventana = tk.Toplevel(self.root)
        ventana.title("Predecir Siguiente - Modelos ML")
        ventana.geometry("420x420")
        ventana.minsize(380, 380)
        ventana.transient(self.root)
        ventana.grab_set()

        modo = tk.StringVar(value="real")

        ttk.Label(ventana, text="Modo de entrada:", font=("", 10, "bold")).pack(anchor=tk.W, padx=15, pady=(12, 5))

        frame_real = ttk.LabelFrame(ventana, text="Modo A — Último dato real", padding=8)
        frame_real.pack(fill=tk.X, padx=15, pady=4)
        rb_real = ttk.Radiobutton(frame_real, text="", variable=modo, value="real")
        rb_real.pack(anchor=tk.W)
        ttk.Label(frame_real, text=f"Último resultado: {animal_real} a las {hora_12h}",
                  font=("", 9)).pack(anchor=tk.W, padx=18)

        frame_manual = ttk.LabelFrame(ventana, text="Modo B — Ingresar manualmente", padding=8)
        frame_manual.pack(fill=tk.X, padx=15, pady=4)
        rb_manual = ttk.Radiobutton(frame_manual, text="", variable=modo, value="manual")
        rb_manual.pack(anchor=tk.W)
        ttk.Label(frame_manual, text="Animal que acaba de salir:", font=("", 9)).pack(anchor=tk.W, padx=18, pady=(5, 2))
        entry_animal = ttk.Entry(frame_manual, width=25, font=("", 10))
        entry_animal.pack(anchor=tk.W, padx=18, pady=2)
        ttk.Label(frame_manual, text="Hora del sorteo (HH:MM AM/PM):", font=("", 9)).pack(anchor=tk.W, padx=18, pady=(5, 2))
        horas_opciones = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                          "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                          "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo_hora = ttk.Combobox(frame_manual, values=horas_opciones, state="readonly", width=15, font=("", 10))
        combo_hora.pack(anchor=tk.W, padx=18, pady=2)
        combo_hora.set(horas_opciones[-1])

        def _toggle_modo(*_):
            state = tk.NORMAL if modo.get() == "manual" else tk.DISABLED
            entry_animal.config(state=state)
            combo_hora.config(state=state)
        modo.trace_add("write", _toggle_modo)
        _toggle_modo()

        def ejecutar():
            animal = None
            hora_24h = None
            solo_hora = None
            if modo.get() == "manual":
                animal = entry_animal.get().strip().upper()
                if not animal or animal not in analizador.animales_carac:
                    animales_ok = ", ".join(sorted(analizador.animales_carac.keys()))
                    messagebox.showwarning("Error", f"Animal invalido. Validos: {animales_ok}")
                    return
                hora_str = combo_hora.get().strip()
                if not hora_str:
                    messagebox.showwarning("Error", "Selecciona una hora")
                    return
                try:
                    dt_hora = pd.to_datetime(hora_str, format='%I:%M %p')
                    hora_24h = dt_hora.strftime('%H:%M:%S')
                    solo_hora = dt_hora.strftime('%I:%M %p')
                except Exception:
                    messagebox.showwarning("Error", "Hora invalida")
                    return
            ventana.destroy()
            hilo = threading.Thread(
                target=self._predecir_siguiente_ml_task,
                args=(animal, hora_24h, solo_hora), daemon=True
            )
            hilo.start()

        frame_btn = ttk.Frame(ventana)
        frame_btn.pack(pady=15)
        ttk.Button(frame_btn, text="Predecir", command=ejecutar).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_btn, text="Cancelar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)

    def _predecir_siguiente_ml_task(self, animal=None, hora_24h=None, solo_hora=None):
        panel = self._paneles.get("ml_predecir")
        if panel is None:
            return
        redir = RedirectText(panel)
        self.root.after(0, lambda: panel.delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            datos = self._get_datos()
            if datos is None or datos.empty:
                print("ERROR: Sin datos")
                return
            analizador = self._get_analizador()
            if not analizador:
                return
            rf, le_rf, xgb, le_xgb = self._get_modelos()
            d = datos.copy()
            if animal is not None:
                d.iloc[-1, d.columns.get_loc('Animal')] = animal
                d.iloc[-1, d.columns.get_loc('Numero')] = 0
                d.iloc[-1, d.columns.get_loc('Hora')] = hora_24h
                d.iloc[-1, d.columns.get_loc('Solo_hora')] = solo_hora
                print(f"[Modo B] Estado simulado: {animal} a las {solo_hora}")
            else:
                try:
                    h12 = pd.to_datetime(d.iloc[-1]['Hora'], format='%H:%M:%S').strftime('%I:%M %p').lstrip('0')
                except Exception:
                    h12 = d.iloc[-1]['Hora']
                print(f"[Modo A] Ultimo resultado real: {d.iloc[-1]['Animal']} a las {h12}")
            d2 = analizador.agregar_caracteristicas_avanzadas(d)
            analizador.prediccion_completa_hoy(d2, rf, le_rf, xgb, le_xgb)

    # ================ TAB: MODELOS ML ================

    def _tab_crear_modelos(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Modelos ML")

        frame_btn = ttk.Frame(tab)
        frame_btn.pack(fill=tk.X, padx=5, pady=5)

        btns = [
            ("Entrenar Random Forest", "ml_rf"),
            ("Entrenar XGBoost", "ml_xgb"),
            ("Evaluar con IA", "ml_eval_ia"),
            ("Auto-Evaluacion", "ml_auto_eval"),
            ("Predecir Siguiente", "ml_predecir"),
            ("Patrones", "ml_patrones"),
        ]
        for texto, accion in btns:
            btn = ttk.Button(frame_btn, text=texto, command=lambda a=accion: self._modelos_accion(a))
            btn.pack(side=tk.LEFT, padx=2)

        grid = self._crear_paned_grid(tab)

        self._agregar_panel_salida(grid, "Entrenar RF / XGB / Patrones", "ml_entrenamiento", 0, 0)
        self._agregar_panel_salida(grid, "Evaluar con IA", "ml_eval_ia", 0, 1)
        self._agregar_panel_salida(grid, "Auto-Evaluacion", "ml_auto_eval", 1, 0)
        self._agregar_panel_salida(grid, "Predecir Siguiente", "ml_predecir", 1, 1)

    def _get_modelos(self):
        rf = self.modelos_rf.get(self.current_lottery)
        le_rf = self.le_y_rf.get(self.current_lottery)
        xgb = self.modelos_xgb.get(self.current_lottery)
        le_xgb = self.le_y_xgb.get(self.current_lottery)
        return rf, le_rf, xgb, le_xgb

    def _modelos_accion(self, accion):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        d = datos.copy()
        analizador = self._get_analizador()
        mod = self._get_mod()
        if not analizador or not mod:
            return

        panel_key = accion if accion != "ml_patrones" else "ml_entrenamiento"
        if accion in ("ml_rf", "ml_xgb"):
            panel_key = "ml_entrenamiento"
        panel = self._paneles.get(panel_key)
        if panel is None:
            return

        def ejecutar_en_panel(func):
            def tarea():
                self.root.after(0, lambda: panel.delete("1.0", tk.END))
                redir = RedirectText(panel)
                with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
                    func()
            hilo = threading.Thread(target=tarea, daemon=True)
            hilo.start()

        if accion in ("ml_rf", "ml_xgb", "ml_patrones"):
            if accion == "ml_rf":
                ejecutar_en_panel(lambda: self._entrenar_rf(analizador, mod, d))
            elif accion == "ml_xgb":
                ejecutar_en_panel(lambda: self._entrenar_xgb(analizador, mod, d))
            elif accion == "ml_patrones":
                ejecutar_en_panel(lambda: analizador.analizar_patrones_sorteo(d))

        elif accion == "ml_eval_ia":
            rf, le_rf, xgb, le_xgb = self._get_modelos()
            modelo = xgb or rf
            le_y = le_xgb or le_rf
            def tarea():
                d2 = analizador.agregar_caracteristicas_avanzadas(d.copy())
                print("=" * 60)
                print("  EVALUACION COMPLETA CON IA")
                print("=" * 60)
                print("\n--- 1. PREDICCION COMBINADA PARA HOY (Ensemble) ---")
                analizador.prediccion_hoy_ensemble(d2, modelo, le_y, k=25)
                if rf and le_rf:
                    print("\n--- 2. MATRIZ RANDOM FOREST (Top-25 por hora) ---")
                    matriz_rf = analizador.predecir_top_k_por_hora(rf, le_rf, d2.copy(), k=25)
                    analizador.mostrar_matriz_prediccion(matriz_rf)
                if xgb and le_xgb:
                    print("\n--- 3. MATRIZ XGBoost (Top-25 por hora) ---")
                    matriz_xgb = analizador.predecir_top_k_por_hora(xgb, le_xgb, d2.copy(), k=25)
                    analizador.mostrar_matriz_prediccion(matriz_xgb)
            ejecutar_en_panel(tarea)

        elif accion == "ml_auto_eval":
            rf, le_rf, xgb, le_xgb = self._get_modelos()
            def tarea():
                d2 = analizador.agregar_caracteristicas_avanzadas(d.copy())
                analizador.evaluar_predicciones_historicas(d2, rf, le_rf, xgb, le_xgb)
            ejecutar_en_panel(tarea)

        elif accion == "ml_predecir":
            self._predecir_siguiente_ml_dialogo()

    def _entrenar_rf(self, analizador, mod, d):
        analizador.random_forest_optimizado(d)
        rf, le_rf, _ = mod.cargar_ultimo_modelo("random_forest")
        if rf:
            self.modelos_rf[self.current_lottery] = rf
            self.le_y_rf[self.current_lottery] = le_rf
            print("\nRandom Forest entrenado y cargado")

    def _entrenar_xgb(self, analizador, mod, d):
        analizador.xgboost_optimizado(d)
        xgb, le_xgb, _ = mod.cargar_ultimo_modelo("xgboost")
        if xgb:
            self.modelos_xgb[self.current_lottery] = xgb
            self.le_y_xgb[self.current_lottery] = le_xgb
            print("\nXGBoost entrenado y cargado")

    # ================ TAB: SCRAPER ================

    def _tab_crear_scraper(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Web Scraper")

        frame_btn = ttk.Frame(tab)
        frame_btn.pack(fill=tk.X, padx=5, pady=5)

        btns = [
            ("Scrapear Faltantes (Actual)", "scraper_faltantes"),
            ("Scrapear Todas", "scraper_todas"),
            ("Scrapear Dia Especifico", "scraper_dia"),
            ("Scrapear Rango", "scraper_rango"),
        ]
        for texto, accion in btns:
            btn = ttk.Button(frame_btn, text=texto, command=lambda a=accion: self._scraper_accion(a))
            btn.pack(side=tk.LEFT, padx=2)

        grid = self._crear_paned_grid(tab)

        self._agregar_panel_salida(grid, "Scrapear Faltantes", "scraper_faltantes", 0, 0)
        self._agregar_panel_salida(grid, "Scrapear Todas", "scraper_todas", 0, 1)
        self._agregar_panel_salida(grid, "Scrapear Dia Especifico", "scraper_dia", 1, 0)
        self._agregar_panel_salida(grid, "Scrapear Rango", "scraper_rango", 1, 1)

    def _scraper_accion(self, accion):
        if accion == "scraper_faltantes":
            self._scraper_faltantes()
        elif accion == "scraper_todas":
            self._scraper_todas()
        elif accion == "scraper_dia":
            self._scraper_dialogo_dia()
        elif accion == "scraper_rango":
            self._scraper_dialogo_rango()

    def _scraper_faltantes(self):
        hilo = threading.Thread(target=self._scraper_faltantes_task, daemon=True)
        hilo.start()

    def _scraper_faltantes_task(self):
        redir = RedirectText(self._paneles["scraper_faltantes"])
        self.root.after(0, lambda: self._paneles["scraper_faltantes"].delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            nombre = self.current_lottery
            mod = LOTTERY_MODULES.get(nombre)
            if not mod:
                return
            config = mod.CONFIG
            excel_file = config['excel_file']

            try:
                try:
                    df = pd.read_excel(excel_file)
                    fechas_existentes = set(pd.to_datetime(df['Fecha']).dt.strftime("%Y-%m-%d").unique())
                    ultima = datetime.datetime.strptime(max(fechas_existentes), "%Y-%m-%d").date()
                except (FileNotFoundError, Exception):
                    fechas_existentes = set()
                    ultima = datetime.date(2024, 1, 1) - datetime.timedelta(days=1)
                hoy = datetime.date.today()
                if ultima >= hoy:
                    print(f"OK - {nombre}: Datos ya estan al dia")
                    return
                desde = ultima + datetime.timedelta(days=1)
                faltantes = []
                d = desde
                while d <= hoy:
                    ds = d.strftime("%Y-%m-%d")
                    if ds not in fechas_existentes:
                        faltantes.append(ds)
                    d += datetime.timedelta(days=1)
                if not faltantes:
                    print(f"OK - {nombre}: No hay fechas faltantes")
                    return

                print(f"{nombre}: Scrapeando {len(faltantes)} fechas desde {desde}...")

                if nombre == "Lotto Activo":
                    from scraper_lotto import scrape_date, save_to_excel
                elif nombre == "La Granjita":
                    from scraper_la_granjita import scrape_date, save_to_excel
                elif nombre == "Selva Plus":
                    from scraper_selva_plus import scrape_date, save_to_excel
                elif nombre == "Lotto Activo Rd Int":
                    from scraper_lotto_rd_int import scrape_date, save_to_excel
                else:
                    return

                todos = []
                for i, fs in enumerate(faltantes):
                    r = scrape_date(fs)
                    todos.extend(r)
                    if (i+1) % 10 == 0:
                        print(f"  Progreso: {i+1}/{len(faltantes)}")
                    time.sleep(1.5)
                if todos:
                    df_nuevo = pd.DataFrame(todos)
                    save_to_excel(df_nuevo, excel_file)
                    print(f"{nombre}: {len(df_nuevo)} registros agregados")
                    self._cargar_loteria(nombre)
                    self.root.after(100, self._actualizar_status)
                else:
                    print(f"{nombre}: Sin nuevos registros")
            except Exception as e:
                print(f"Error: {e}")

    def _scraper_todas(self):
        hilo = threading.Thread(target=self._scraper_todas_task, daemon=True)
        hilo.start()

    def _scraper_todas_task(self):
        text = self._paneles["scraper_todas"]
        redir = RedirectText(text)
        self.root.after(0, lambda: text.delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            scrapers = {
                "Lotto Activo": ("scraper_lotto", "LottoActivoCompleto.xlsx"),
                "La Granjita": ("scraper_la_granjita", "LaGranjita.xlsx"),
                "Selva Plus": ("scraper_selva_plus", "SelvaPlus.xlsx"),
                "Lotto Activo Rd Int": ("scraper_lotto_rd_int", "LottoActivoRDInt.xlsx"),
            }
            for nombre, (mod_name, excel_file) in scrapers.items():
                try:
                    mod = __import__(mod_name)
                    try:
                        df = pd.read_excel(excel_file)
                        fechas_existentes = set(pd.to_datetime(df['Fecha']).dt.strftime("%Y-%m-%d").unique())
                        ultima = datetime.datetime.strptime(max(fechas_existentes), "%Y-%m-%d").date()
                    except (FileNotFoundError, Exception):
                        fechas_existentes = set()
                        ultima = datetime.date(2024, 1, 1) - datetime.timedelta(days=1)
                    hoy = datetime.date.today()
                    if ultima >= hoy:
                        print(f"OK - {nombre}: Datos al dia")
                        continue
                    desde = ultima + datetime.timedelta(days=1)
                    faltantes = []
                    d = desde
                    while d <= hoy:
                        ds = d.strftime("%Y-%m-%d")
                        if ds not in fechas_existentes:
                            faltantes.append(ds)
                        d += datetime.timedelta(days=1)
                    if not faltantes:
                        print(f"OK - {nombre}: Sin faltantes")
                        continue
                    print(f"{nombre}: {len(faltantes)} fechas faltantes")
                    todos = []
                    for i, fs in enumerate(faltantes):
                        r = mod.scrape_date(fs)
                        todos.extend(r)
                        if (i+1) % 10 == 0:
                            print(f"  {i+1}/{len(faltantes)}")
                        time.sleep(1.5)
                    if todos:
                        df_nuevo = pd.DataFrame(todos)
                        mod.save_to_excel(df_nuevo, excel_file)
                        print(f"  +{len(df_nuevo)} registros")
                        self._cargar_loteria(nombre)
                except Exception as e:
                    print(f"Error {nombre}: {e}")
                time.sleep(2)
            print("\nScraping completado para todas las loterias")
            self.root.after(100, self._actualizar_status)

    def _scraper_dialogo_dia(self):
        ventana = tk.Toplevel(self.root)
        ventana.title("Scrapear Dia")
        ventana.geometry("300x150")
        ventana.transient(self.root)
        ventana.grab_set()

        ttk.Label(ventana, text="Fecha (YYYY-MM-DD):").pack(pady=(10, 0))
        entry = ttk.Entry(ventana)
        entry.pack(pady=5)
        entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

        def ejecutar():
            fecha = entry.get().strip()
            ventana.destroy()
            hilo = threading.Thread(target=self._scraper_dia_task, args=(fecha,), daemon=True)
            hilo.start()

        ttk.Button(ventana, text="Scrapear", command=ejecutar).pack(pady=10)

    def _scraper_dia_task(self, fecha):
        text = self._paneles["scraper_dia"]
        redir = RedirectText(text)
        self.root.after(0, lambda: text.delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            nombre = self.current_lottery
            mod = LOTTERY_MODULES.get(nombre)
            if not mod:
                return
            config = mod.CONFIG
            excel_file = config['excel_file']

            if nombre == "Lotto Activo":
                from scraper_lotto import scrape_date, save_to_excel
            elif nombre == "La Granjita":
                from scraper_la_granjita import scrape_date, save_to_excel
            elif nombre == "Selva Plus":
                from scraper_selva_plus import scrape_date, save_to_excel
            elif nombre == "Lotto Activo Rd Int":
                from scraper_lotto_rd_int import scrape_date, save_to_excel
            else:
                return

            try:
                records = scrape_date(fecha)
                if records:
                    df = pd.DataFrame(records)
                    save_to_excel(df, excel_file)
                    print(f"{nombre}: {len(records)} registros de {fecha}")
                    self._cargar_loteria(nombre)
                    self.root.after(100, self._actualizar_status)
                else:
                    print(f"Sin registros para {fecha}")
            except Exception as e:
                print(f"Error: {e}")

    def _scraper_dialogo_rango(self):
        ventana = tk.Toplevel(self.root)
        ventana.title("Scrapear Rango")
        ventana.geometry("300x180")
        ventana.transient(self.root)
        ventana.grab_set()

        ttk.Label(ventana, text="Desde (YYYY-MM-DD):").pack(pady=(10, 0))
        entry_desde = ttk.Entry(ventana)
        entry_desde.pack(pady=2)
        entry_desde.insert(0, "2024-01-01")

        ttk.Label(ventana, text="Hasta (YYYY-MM-DD):").pack(pady=(5, 0))
        entry_hasta = ttk.Entry(ventana)
        entry_hasta.pack(pady=2)
        entry_hasta.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

        def ejecutar():
            desde = entry_desde.get().strip()
            hasta = entry_hasta.get().strip()
            ventana.destroy()
            hilo = threading.Thread(target=self._scraper_rango_task, args=(desde, hasta), daemon=True)
            hilo.start()

        ttk.Button(ventana, text="Scrapear", command=ejecutar).pack(pady=10)

    def _scraper_rango_task(self, desde, hasta):
        text = self._paneles["scraper_rango"]
        redir = RedirectText(text)
        self.root.after(0, lambda: text.delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            nombre = self.current_lottery
            mod = LOTTERY_MODULES.get(nombre)
            if not mod:
                return
            config = mod.CONFIG
            excel_file = config['excel_file']

            if nombre == "Lotto Activo":
                from scraper_lotto import scrape_range, save_to_excel
            elif nombre == "La Granjita":
                from scraper_la_granjita import scrape_range, save_to_excel
            elif nombre == "Selva Plus":
                from scraper_selva_plus import scrape_range, save_to_excel
            elif nombre == "Lotto Activo Rd Int":
                from scraper_lotto_rd_int import scrape_range, save_to_excel
            else:
                return

            try:
                df = scrape_range(desde, hasta)
                if not df.empty:
                    save_to_excel(df, excel_file)
                    print(f"{nombre}: {len(df)} registros agregados")
                    self._cargar_loteria(nombre)
                    self.root.after(100, self._actualizar_status)
                else:
                    print("Sin registros en ese rango")
            except Exception as e:
                print(f"Error: {e}")

    def _recargar_datos(self):
        self._cargar_loteria(self.current_lottery)
        self._actualizar_status()
        messagebox.showinfo("OK", f"Datos recargados: {self.current_lottery}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LottoPredictorUI(root)
    root.mainloop()
