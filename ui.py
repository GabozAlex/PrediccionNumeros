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

    def _agregar_panel_salida(self, parent, label, name, row, column, columnspan=1):
        frame = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        frame.grid(row=row, column=column, columnspan=columnspan, sticky='nsew', padx=2, pady=2)
        ttk.Label(frame, text=label, font=("", 8, "bold")).pack(anchor=tk.W, padx=2)
        text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Courier", 8))
        text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(row, weight=1)
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

        grid = ttk.Frame(tab)
        grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

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

        btns = [
            ("Matriz Transicion Markov", "pred_matriz"),
            ("Probabilidad por Hora", "pred_prob_hora"),
            ("Predecir Siguiente (M+H)", "pred_sig_mh"),
            ("Top-25 General", "pred_top25"),
        ]
        for texto, accion in btns:
            btn = ttk.Button(frame_btn, text=texto, command=lambda a=accion: self._prediccion_accion(a))
            btn.pack(side=tk.LEFT, padx=2)

        grid = ttk.Frame(tab)
        grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        self._agregar_panel_salida(grid, "Matriz Transicion Markov", "pred_matriz", 0, 0)
        self._agregar_panel_salida(grid, "Probabilidad por Hora", "pred_prob_hora", 0, 1)
        self._agregar_panel_salida(grid, "Top-25 General", "pred_top25", 1, 0)
        self._agregar_panel_salida(grid, "Predecir Siguiente (M+H)", "pred_sig_mh", 1, 1)

        self._precargar_prediccion()

    def _precargar_prediccion(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            return
        d = datos.copy()
        analizador = self._get_analizador()
        if not analizador:
            return
        self._ejecutar_en_hilo(lambda: analizador.matriz_probabilidad_transicion(d), self._paneles["pred_matriz"])
        self._ejecutar_en_hilo(lambda: analizador.probabilidad_maxima_por_hora(d), self._paneles["pred_prob_hora"])

    def _prediccion_accion(self, accion):
        if accion == "pred_sig_mh":
            self._predecir_siguiente_mh_dialogo()
            return
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
            "pred_matriz": lambda: analizador.matriz_probabilidad_transicion(d),
            "pred_prob_hora": lambda: analizador.probabilidad_maxima_por_hora(d),
            "pred_markov_hora": lambda: analizador.prediccion_markov_hora(d),
            "pred_dia_completo": lambda: analizador.prediccion_dia_completo(d),
            "pred_top25": lambda: analizador.top_25_general(d),
        }
        func = mapa.get(accion)
        if func:
            self._ejecutar_en_hilo(func, panel)

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
        ventana.geometry("420x340")
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

        grid = ttk.Frame(tab)
        grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for c in range(2):
            grid.columnconfigure(c, weight=1)
        for r in range(2):
            grid.rowconfigure(r, weight=1)

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

        grid = ttk.Frame(tab)
        grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

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
