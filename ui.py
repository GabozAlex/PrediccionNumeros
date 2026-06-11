import sys
import os
import io
import contextlib
import threading
import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import traceback

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def _safe_thread(tarea_func, ventana, txt):
    """Wraps a thread tarea_func to catch and display exceptions."""
    def wrapper():
        try:
            tarea_func()
        except Exception:
            err = traceback.format_exc()
            ventana.after(0, lambda: (
                txt.delete("1.0", tk.END),
                txt.insert(tk.END, f"ERROR:\n{err}")
            ))
    return wrapper

from utils import ANIMAL_A_NUM_INT

import lotto_activo
import la_granjita
import selva_plus
import lotto_rd_int
import lotto_activo_rd
import lotto_activo_unificado

LOTTERY_MODULES = {
    "Lotto Activo": lotto_activo,
    "La Granjita": la_granjita,
    "Selva Plus": selva_plus,
    "Lotto Activo Rd Int": lotto_rd_int,
    "Lotto Activo RD": lotto_activo_rd,
    "Lotto Activo Unificado": lotto_activo_unificado,
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


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tw, text=self.text, justify=tk.LEFT,
                          background="#ffffcc", relief=tk.SOLID, borderwidth=1,
                          wraplength=350, padding=5)
        label.pack()

    def leave(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


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

        self._configurar_estilos()
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

    def _configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure('Analisis.TButton', font=('', 9), padding=6)
        style.configure('Markov.TButton', font=('', 9), padding=6)
        style.configure('ML.TButton', font=('', 9), padding=6)
        style.configure('Scraper.TButton', font=('', 9), padding=6)
        style.configure('Experimentacion.TButton', font=('', 9), padding=6, foreground='#2E7D32')
        style.configure('Evaluacion.TButton', font=('', 9, 'bold'), padding=6, foreground='#1565C0')
        style.configure('TLabelframe.Label', font=('', 10, 'bold'))

    def _crear_ventana_salida(self, titulo, ancho=750, alto=620):
        ventana = tk.Toplevel(self.root)
        ventana.title(f"{titulo} - {self.current_lottery}")
        ventana.geometry(f"{ancho}x{alto}")
        ventana.minsize(500, 300)
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert(tk.END, f"Cargando {titulo.lower()}...\n")
        return ventana, txt

    def _ejecutar_en_ventana(self, txt, func):
        def tarea():
            redir = RedirectText(txt)
            with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
                try:
                    func()
                except Exception as e:
                    print(f"Error: {e}")
            txt.after(0, lambda: txt.see(tk.END))
        hilo = threading.Thread(target=tarea, daemon=True)
        hilo.start()

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
            datos['Num_Int'] = datos['Animal'].map(ANIMAL_A_NUM_INT)
            datos = datos.dropna(subset=['Num_Int']).reset_index(drop=True)
            datos['Num_Int'] = datos['Num_Int'].astype(int)
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

    # ================ TAB: DASHBOARD ================

    def _tab_crear_dashboard(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dashboard")

        group = ttk.LabelFrame(tab, text="📊 Analisis Rapido", padding=10)
        group.pack(fill=tk.X, padx=10, pady=10)

        btns = [
            ("Estado del Dia", "Muestra sorteos de hoy, horas pendientes y resumen rapido"),
            ("Ultimos Registros", "Ultimos 10 sorteos, faltantes de hoy, numeros calientes/frios"),
        ]
        btn_frame = ttk.Frame(group)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        for texto, tip in btns:
            btn = ttk.Button(btn_frame, text=texto, style='Analisis.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        btns2 = [
            ("Aciertos x Dia", "% de aciertos GxHora por dia de la semana"),
            ("Fallos x Dia", "% de fallos GxHora por dia de la semana"),
        ]
        btn_frame2 = ttk.Frame(group)
        btn_frame2.pack(fill=tk.X, pady=(0, 5))
        for texto, tip in btns2:
            btn = ttk.Button(btn_frame2, text=texto, style='Analisis.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        btns3 = [
            ("Aciertos x Hora", "% de aciertos GxHora por hora del dia"),
            ("Fallos x Hora", "% de fallos GxHora por hora del dia"),
        ]
        btn_frame3 = ttk.Frame(group)
        btn_frame3.pack(fill=tk.X)
        for texto, tip in btns3:
            btn = ttk.Button(btn_frame3, text=texto, style='Analisis.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        group_exp = ttk.LabelFrame(tab, text="🔬 Experimentacion", padding=10)
        group_exp.pack(fill=tk.X, padx=10, pady=(0, 10))

        exp_btns1 = [
            ("Evaluar Markov x Dia", "Evalua Markov segmentado por dia de semana"),
            ("GxHora + Filtro Dia", "Predice con GxHora filtrando solo numeros que han salido en este dia de semana"),
        ]
        exp_frame1 = ttk.Frame(group_exp)
        exp_frame1.pack(fill=tk.X, pady=(0, 5))
        for texto, tip in exp_btns1:
            btn = ttk.Button(exp_frame1, text=texto, style='Experimentacion.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        exp_btns2 = [
            ("Evaluar GxHora+Filtro", "Compara GxHora normal vs GxHora con filtro por dia de semana"),
            ("MkHora + Dia", "Predice combinando Markov x Hora + dia de semana (triplas)"),
        ]
        exp_frame2 = ttk.Frame(group_exp)
        exp_frame2.pack(fill=tk.X, pady=(0, 5))
        for texto, tip in exp_btns2:
            btn = ttk.Button(exp_frame2, text=texto, style='Experimentacion.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        exp_btns3 = [
            ("Evaluar MkHora+Dia", "Compara MkHora normal vs MkHora+Dia (fragmentado por dia de semana)"),
        ]
        exp_frame3 = ttk.Frame(group_exp)
        exp_frame3.pack(fill=tk.X)
        for texto, tip in exp_btns3:
            btn = ttk.Button(exp_frame3, text=texto, style='Experimentacion.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        group_eval = ttk.LabelFrame(tab, text="📊 Evaluaciones", padding=10)
        group_eval.pack(fill=tk.X, padx=10, pady=(0, 10))

        eval_btns1 = [
            ("Markov Global", "% acierto global, por hora, por dia, mejor/peor"),
            ("Frec. x Hora", "% acierto de frecuencia historica por hora"),
        ]
        eval_frame1 = ttk.Frame(group_eval)
        eval_frame1.pack(fill=tk.X, pady=(0, 5))
        for texto, tip in eval_btns1:
            btn = ttk.Button(eval_frame1, text=texto, style='Evaluacion.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        eval_btns2 = [
            ("Markov x Hora", "% acierto de Markov x Hora (MkHora)"),
            ("GxHora", "% acierto de GxHora (Markov + MkHora combinado)"),
        ]
        eval_frame2 = ttk.Frame(group_eval)
        eval_frame2.pack(fill=tk.X, pady=(0, 5))
        for texto, tip in eval_btns2:
            btn = ttk.Button(eval_frame2, text=texto, style='Evaluacion.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        eval_btns3 = [
            ("Markov x Dia", "% acierto de Markov segmentado por dia de semana"),
            ("Evaluar Top-5 Completo", "Tabla detallada con los 5 modelos: fecha, predicho, real, hits, ranks"),
        ]
        eval_frame3 = ttk.Frame(group_eval)
        eval_frame3.pack(fill=tk.X)
        for texto, tip in eval_btns3:
            btn = ttk.Button(eval_frame3, text=texto, style='Evaluacion.TButton',
                             command=lambda t=texto: self._dashboard_dialogo(t))
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

    def _dashboard_dialogo(self, texto):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        d = datos.copy()
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida(texto)
        mapa = {
            "Estado del Dia": lambda: analizador.ver_estado_actual_dia(d),
            "Ultimos Registros": lambda: analizador.ver_ultimos_registros_y_faltantes(d),
            "Aciertos x Dia": lambda: analizador.analizar_aciertos_por_dia_semana(d),
            "Fallos x Dia": lambda: analizador.analizar_fallos_por_dia_semana(d),
            "Aciertos x Hora": lambda: analizador.analizar_aciertos_por_hora(d),
            "Fallos x Hora": lambda: analizador.analizar_fallos_por_hora(d),
            "Evaluar Markov x Dia": lambda: analizador.evaluar_markov_dia_semana(d),
            "GxHora + Filtro Dia": lambda: analizador.prediccion_gxhora_filtro_dia(d),
            "Evaluar GxHora+Filtro": lambda: analizador.evaluar_gxhora_filtro_dia(d),
            "MkHora + Dia": lambda: analizador.prediccion_markov_hora_dia(d),
            "Evaluar MkHora+Dia": lambda: analizador.evaluar_markov_hora_dia(d),
            "Markov Global": lambda: analizador.evaluar_markov_global(d),
            "Frec. x Hora": lambda: analizador.evaluar_frecuencia_hora(d),
            "Markov x Hora": lambda: analizador.evaluar_markov_hora(d),
            "GxHora": lambda: analizador.evaluar_gxhora(d),
            "Markov x Dia": lambda: analizador.evaluar_markov_dia_semana(d),
            "Evaluar Top-5 Completo": lambda: analizador.evaluar_top5_completo(d),
        }
        func = mapa.get(texto)
        if func:
            self._ejecutar_en_ventana(txt, func)

    # ================ TAB: PREDICCION ================

    def _tab_crear_prediccion(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Prediccion")

        grupo_markov = ttk.LabelFrame(tab, text="🔮 Markov", padding=8)
        grupo_markov.pack(fill=tk.X, padx=10, pady=(10, 5))

        row1 = ttk.Frame(grupo_markov)
        row1.pack(fill=tk.X, pady=2)
        row2 = ttk.Frame(grupo_markov)
        row2.pack(fill=tk.X, pady=2)

        markov_btns = [
            ("Matriz Global", self._dialogo_markov, "Dado un numero, muestra los que mas le siguen (Markov global)"),
            ("Prob. x Hora", self._dialogo_prob_hora, "Frecuencia de cada numero para una hora especifica"),
            ("Markov x Hora", self._dialogo_markov_hora, "Dado animal+hora, muestra los siguientes mas probables"),
        ]
        for texto, cmd, tip in markov_btns:
            btn = ttk.Button(row1, text=texto, style='Markov.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        markov_btns2 = [
            ("Global + Hora", self._ventana_matriz_combinada, "Combinacion ponderada de Markov global y por hora"),
            ("Siguiente M+H", self._predecir_siguiente_mh_dialogo, "Predice el proximo numero usando Markov + frecuencia por hora"),
        ]
        for texto, cmd, tip in markov_btns2:
            btn = ttk.Button(row2, text=texto, style='Markov.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        row3 = ttk.Frame(grupo_markov)
        row3.pack(fill=tk.X, pady=2)

        markov_btns3 = [
            ("Co-ocurrencia", self._dialogo_coocurrencia, "Predice mostrando numeros que suelen salir el mismo dia que el numero actual"),
            ("Full (4 fuentes)", self._dialogo_full, "Predice combinando Markov global, Markov x hora, frecuencia x hora y co-ocurrencia"),
            ("Markov x Dia", self._dialogo_markov_dia_semana, "Predice usando matriz de Markov del dia de la semana actual"),
        ]
        for texto, cmd, tip in markov_btns3:
            btn = ttk.Button(row3, text=texto, style='Markov.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        grupo_rank = ttk.LabelFrame(tab, text="📋 Rankings y Patrones", padding=8)
        grupo_rank.pack(fill=tk.X, padx=10, pady=5)

        row_r1 = ttk.Frame(grupo_rank)
        row_r1.pack(fill=tk.X, pady=2)
        row_r2 = ttk.Frame(grupo_rank)
        row_r2.pack(fill=tk.X, pady=2)

        rank_btns = [
            ("Top-25 General", self._dialogo_top25, "Ranking global de los 25 numeros mas frecuentes"),
            ("Co-ocurrencias", self._patron_coocurrencias, "Analiza que numeros suelen salir juntos en un mismo dia"),
            ("Rango Horario", self._patron_rango, "Analiza co-ocurrencias segmentadas por rango de hora"),
            ("Dia Semana", self._patron_dia_semana, "Frecuencia de cada numero segun el dia de la semana"),
        ]
        for texto, cmd, tip in rank_btns:
            btn = ttk.Button(row_r1, text=texto, style='Markov.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        rank_btns2 = [
            ("Rachas", self._patron_rachas, "Analiza secuencias de aciertos y fallos por cada numero"),
            ("Comparar", self._patron_comparar, "Compara rendimiento de estrategias: global, hora, frecuencia"),
            ("Cadena Dia", self._dialogo_cadena, "Muestra la secuencia completa de numeros de un dia especifico"),
            ("2do Orden", self._dialogo_segundo_orden, "Matriz Markov de segundo orden (ultimos 2 numeros)"),
        ]
        for texto, cmd, tip in rank_btns2:
            btn = ttk.Button(row_r2, text=texto, style='Markov.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

    def _dialogo_top25(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida("Top-25 General", ancho=550, alto=500)
        self._ejecutar_en_ventana(txt, lambda: analizador.top_25_general(datos.copy()))

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
                    ant, hp = test.iloc[i]['Num_Int'], test.iloc[i]['Hora']
                    sig, hn = test.iloc[i+1]['Num_Int'], test.iloc[i+1]['Hora']
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
        threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()

    def _evaluar_por_animal(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Precision x Numero - {self.current_lottery}")
        ventana.geometry("700x600")
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert(tk.END, "Evaluando precision por numero...\n")
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
                ant, hp = test.iloc[i]['Num_Int'], test.iloc[i]['Hora']
                sig = test.iloc[i+1]['Num_Int']
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
            a2an = analizador.num_int_a_animal
            texto = f"Precision Global Top-25 por Numero (ultimos 1000 sorteos)\n"
            texto += "=" * 60 + "\n\n"
            texto += f"{'Num(Animal)':<15} {'Aciertos':>8} {'Total':>6} {'%':>6}\n"
            texto += '-' * 40 + '\n'
            for a, pct, tot in ranking:
                barra = '#' * max(1, int(pct/4))
                animal = a2an.get(a, '?')
                texto += f"  {a:>2}({animal:<10}) {ac_g[a]:>3}/{total_animal[a]:<2} {pct:>5.1f}% {barra}\n"
            texto += f"\nPeores 5: {', '.join(str(a)+'('+a2an.get(a, '?')+')' for a,_,_ in ranking[:5])}"
            texto += f"\nMejores 5: {', '.join(str(a)+'('+a2an.get(a, '?')+')' for a,_,_ in ranking[-5:])}"
            ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
        threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()

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
        ttk.Label(frame_top, text="Numero (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_numero = ttk.Entry(frame_top, width=14, font=("", 10))
        entry_numero.pack(side=tk.LEFT, padx=2)
        entry_numero.focus_set()
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
            desde = entry_desde.get().strip()
            hasta = entry_hasta.get().strip()
            if not desde or not hasta:
                return
            try:
                n = int(entry_numero.get().strip())
                if n < 0 or n > 37:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Ingrese un numero entre 0 y 37")
                return
            animal = analizador.num_int_a_animal.get(n, '')
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Analizando {n}({animal}) del {desde} al {hasta}...\n")
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
                texto = f"Backtesting: {n}({animal}) del {desde} al {hasta}\n"
                texto += "=" * 70 + "\n\n"
                for i in range(len(sub)-1):
                    if sub.iloc[i]['Num_Int'] != n:
                        continue
                    if sub.iloc[i]['Fecha'] == sub.iloc[i+1]['Fecha']:
                        total += 1
                        f = sub.iloc[i]['Fecha']
                        h = sub.iloc[i]['Hora']
                        sig = sub.iloc[i+1]['Num_Int']
                        preds = []
                        if n in tot and tot[n] > 0:
                            preds = [a for a,c in sorted(tg[n].items(), key=lambda x:-x[1])[:25]]
                        rank = preds.index(sig) + 1 if sig in preds else 0
                        if rank:
                            hits += 1
                            marca = "✅"
                        else:
                            marca = "❌"
                        sig_animal = analizador.num_int_a_animal.get(sig, '?')
                        texto += f"  {f} {h}  {n:>2}({animal:<10}) -> {sig:>2}({sig_animal:<10})  puesto #{rank if rank else '—'}{'  ' + marca if rank else ''}\n"
                if total:
                    texto += f"\nResumen: {hits}/{total} aciertos ({hits/total*100:.1f}%) en Top-25\n"
                else:
                    texto += "\nNo se encontraron ocurrencias de este numero en el rango\n"
                # Show top predictions for reference
                preds_list = []
                if n in tot and tot[n] > 0:
                    preds_list = [(a, c/tot[n]*100) for a,c in sorted(tg[n].items(), key=lambda x:-x[1])[:25]]
                texto += f"\nTop-25 predictivo para {n}({animal}):\n"
                texto += '-' * 40 + '\n'
                for i, (a, p) in enumerate(preds_list, 1):
                    a_animal = analizador.num_int_a_animal.get(a, '?')
                    texto += f"  {i:2d}. {a:>2}({a_animal:<10}) ({p:.1f}%)\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry_numero.bind("<Return>", lambda e: analizar())
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
                idx = 0
                for i, (o, d) in enumerate(parejas):
                    if o == h_actual:
                        idx = i
                        break
                d = analizador.preparar_datos_markov(datos.copy())
                # Precompute freq per hora and cooc matrix once
                freq_cache = {}
                for o, _ in parejas:
                    h12 = pd.to_datetime(o, format='%H:%M:%S').strftime('%I:%M %p')
                    sub = d[d['Solo_hora'] == h12] if 'Solo_hora' in d.columns else d
                    freq_cache[o] = sub['Num_Int'].value_counts(normalize=True).mul(100).to_dict()
                from collections import defaultdict
                cooc_mat = defaultdict(lambda: defaultdict(int))
                cooc_tot = defaultdict(int)
                for _, grupo in d.groupby('Fecha'):
                    nums = set(grupo['Num_Int'].unique())
                    for n in nums:
                        cooc_tot[n] += 1
                        for n2 in nums:
                            if n2 != n:
                                cooc_mat[n][n2] += 1
                # Markov matrices (will cache)
                trans_g, total_g, trans_h, total_h = analizador.construir_matrices_markov(d, incluir_trasnocho=trasnocho_var.get())
                texto = f"Cadena del dia desde {animal} ({hora_str})\n"
                texto += "=" * 60 + "\n\n"
                texto += f"{'Hora':>8}  {'Actual':<14}  {'Top-5 siguientes':<55}\n"
                texto += '-' * 80 + '\n'
                num_act = analizador.animal_a_num_int.get(animal, -1)
                if num_act < 0:
                    texto = f"Animal '{animal}' no encontrado en el diccionario.\n"
                    ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
                    return
                for i in range(idx, len(parejas)):
                    o, dest = parejas[i]
                    h_12 = pd.to_datetime(o, format='%H:%M:%S').strftime('%I:%M %p')
                    # Global Markov
                    g_scores = {}
                    if num_act in total_g and total_g[num_act]:
                        g_scores = {a2: cnt / total_g[num_act] * 100 for a2, cnt in trans_g[num_act].items()}
                    # Markov x hora
                    h_scores = {}
                    th = total_h.get((o, dest), {})
                    if num_act in th and th[num_act]:
                        h_scores = {a2: cnt / th[num_act] * 100 for a2, cnt in trans_h[(o, dest)][num_act].items()}
                    # Frecuencia
                    f_scores = freq_cache.get(o, {})
                    # Co-ocurrencia
                    c_scores = {n2: c / cooc_tot[num_act] * 100 for n2, c in cooc_mat[num_act].items()} if cooc_tot.get(num_act, 0) else {}
                    # Weighted combination
                    todos = set(g_scores) | set(h_scores) | set(f_scores) | set(c_scores)
                    mh_muestras = th.get(num_act, 0)
                    w_h = min(0.35, max(0.05, mh_muestras / 100 * 0.35))
                    w_g = 0.40 - w_h * 0.5; w_f = 0.20; w_c = 0.40 - w_h * 0.5
                    comb = {n2: g_scores.get(n2, 0) * w_g + h_scores.get(n2, 0) * w_h +
                            f_scores.get(n2, 0) * w_f + c_scores.get(n2, 0) * w_c
                            for n2 in todos if n2 != num_act}
                    preds = [(n, s, 0) for n, s in sorted(comb.items(), key=lambda x: -x[1])[:5]]
                    animal_label = analizador.num_int_a_animal.get(num_act, f"?({num_act})")
                    top5 = [f"{a:>2}({analizador.num_int_a_animal.get(a, '?'):<10}){p:.0f}%" for a, p, _ in preds] if preds else ["(sin datos)"]
                    texto += f"  {h_12:<8}  {num_act:2d}({animal_label:<10})  {' / '.join(top5):<55}\n"
                    num_act = preds[0][0] if preds else num_act
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
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
        ttk.Label(frame_top, text="Numero previo 1 (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry1 = ttk.Entry(frame_top, width=5, font=("", 10))
        entry1.pack(side=tk.LEFT, padx=2)
        ttk.Label(frame_top, text="Numero previo 2 (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry2 = ttk.Entry(frame_top, width=5, font=("", 10))
        entry2.pack(side=tk.LEFT, padx=2)
        entry2.focus_set()
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            n1_str = entry1.get().strip()
            n2_str = entry2.get().strip()
            if not n1_str.isdigit() or not n2_str.isdigit() or not (0 <= int(n1_str) <= 37) or not (0 <= int(n2_str) <= 37):
                messagebox.showwarning("Error", "Ingresa dos numeros validos (0-37)")
                return
            n1, n2 = int(n1_str), int(n2_str)
            a1 = analizador.num_int_a_animal.get(n1, None)
            a2 = analizador.num_int_a_animal.get(n2, None)
            if a1 is None or a2 is None:
                messagebox.showwarning("Error", f"Numero(s) no validos")
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Buscando siguientes de ({a1}, {a2})...\n")
            def tarea():
                items = analizador.get_matriz_segundo_orden(datos.copy(), n1, n2, top_k=25)
                total_m = sum(c for _, _, c in items)
                texto = f"Animales previos: {a1} -> {a2}  |  total pares: {total_m}\n\n"
                texto += f"{'#':>3} {'Num(Animal)':<18} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 40 + "\n"
                if not items:
                    texto += "(sin datos para este par)\n"
                for i, (a3, p, c) in enumerate(items, 1):
                    texto += f"  {i:2d} {a3:>2} ({analizador.num_int_a_animal.get(a3, '?'):<14}) {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry1.bind("<Return>", lambda e: entry2.focus_set())
        entry2.bind("<Return>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _patron_coocurrencias(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida("Co-ocurrencias", ancho=650, alto=500)
        self._ejecutar_en_ventana(txt, lambda: analizador.analizar_coocurrencias(datos.copy()))

    def _patron_rango(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida("Rango Horario", ancho=650, alto=500)
        self._ejecutar_en_ventana(txt, lambda: analizador.analizar_coocurrencias_por_rango(datos.copy()))

    def _patron_dia_semana(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida("Dia Semana", ancho=650, alto=500)
        self._ejecutar_en_ventana(txt, lambda: analizador.analizar_frecuencia_por_dia_semana(datos.copy()))

    def _patron_rachas(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida("Rachas", ancho=650, alto=500)
        self._ejecutar_en_ventana(txt, lambda: analizador.analizar_secuencias_aciertos_fallos(datos.copy()))

    def _patron_comparar(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana, txt = self._crear_ventana_salida("Comparar Estrategias", ancho=700, alto=600)
        self._ejecutar_en_ventana(txt, lambda: analizador.comparar_estrategias(datos.copy()))

    def _dialogo_markov(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Markov - Buscar Numero")
        ventana.geometry("600x520")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Numero (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry = ttk.Entry(frame_top, width=8, font=("", 10))
        entry.pack(side=tk.LEFT, padx=2)
        entry.focus_set()
        trasnocho_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_top, text="7PM→8AM (trasnocho)", variable=trasnocho_var).pack(side=tk.LEFT, padx=5)
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            n_str = entry.get().strip()
            if not n_str.isdigit() or not (0 <= int(n_str) <= 37):
                messagebox.showwarning("Error", "Ingresa un numero valido (0-37)")
                return
            n = int(n_str)
            animal = analizador.num_int_a_animal.get(n, None)
            if animal is None:
                messagebox.showwarning("Error", f"Numero {n} no valido")
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
                texto += f"{'#':>3} {'Num(Animal)':<18} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 42 + "\n"
                if not items:
                    texto += "(sin datos para este animal)\n"
                for i, (a2, p, c) in enumerate(items, 1):
                    texto += f"  {i:2d} {a2:>2} ({analizador.num_int_a_animal.get(a2, '?'):<14}) {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry.bind("<Return>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_prob_hora(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
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
                texto += "=" * 60 + "\n"
                for animal, cnt in conteo.head(25).items():
                    num = analizador.animal_a_num_int.get(animal, '?')
                    pct = cnt / total * 100
                    texto += f"  {num:>2} ({animal:<14}) {cnt:4d} ({pct:.1f}%)\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
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
                texto += f"{'#':>3} {'Num(Animal)':<18} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 40 + "\n"
                if not items:
                    texto += "(sin datos para este animal a esta hora)\n"
                for i, (a2, p, c) in enumerate(items, 1):
                    texto += f"  {i:2d} {a2:>2} ({analizador.num_int_a_animal.get(a2, '?'):<14}) {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry_animal.bind("<Return>", lambda e: buscar())
        combo.bind("<<ComboboxSelected>>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_markov_dia_semana(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title("Markov x Dia de Semana")
        ventana.geometry("650x540")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Animal:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_animal = ttk.Entry(frame_top, width=15, font=("", 10))
        entry_animal.pack(side=tk.LEFT, padx=2)
        entry_animal.focus_set()
        ttk.Label(frame_top, text="Hora:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        horas = ["", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo = ttk.Combobox(frame_top, values=horas, state="normal", width=12, font=("", 10))
        combo.pack(side=tk.LEFT, padx=2)
        combo.set("")
        trasnocho_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_top, text="7PM→8AM", variable=trasnocho_var).pack(side=tk.LEFT, padx=5)
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(frame_top, text="(dejar vacio = ultimo sorteo)", font=("", 8)).pack(side=tk.LEFT, padx=5)
        def buscar():
            animal = entry_animal.get().strip().upper()
            hora_str = combo.get().strip()
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, "Calculando...\n")
            def tarea():
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    analizador.prediccion_markov_dia_semana(
                        datos.copy(), top_k=25,
                        animal=animal if animal else None,
                        hora=hora_str if hora_str else None,
                        incluir_trasnocho=trasnocho_var.get()
                    )
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, buf.getvalue())))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry_animal.bind("<Return>", lambda e: buscar())
        combo.bind("<<ComboboxSelected>>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_coocurrencia(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Co-ocurrencia - {self.current_lottery}")
        ventana.geometry("550x520")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Numero (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry = ttk.Entry(frame_top, width=8, font=("", 10))
        entry.pack(side=tk.LEFT, padx=2)
        entry.focus_set()
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def buscar():
            n_str = entry.get().strip()
            if not n_str.isdigit() or not (0 <= int(n_str) <= 37):
                messagebox.showwarning("Error", "Ingresa un numero valido (0-37)")
                return
            n = int(n_str)
            animal = analizador.num_int_a_animal.get(n, None)
            if animal is None:
                messagebox.showwarning("Error", f"Numero {n} no valido")
                return
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Buscando co-ocurrencias de {animal}...\n")
            def tarea():
                matriz = analizador.get_matriz_coocurrencia_por_animal(datos.copy(), top_k=25)
                items = matriz.get(animal, [])
                total_m = sum(c for _, _, c in items)
                texto = f"Animal actual: {animal}  |  total muestras (dias que aparece): {total_m}"
                texto += f"\n\nNumeros que mas suelen salir el MISMO DIA que {animal}:\n"
                texto += f"{'#':>3} {'Num(Animal)':<18} {'%':>5} {'Muestras':>8}\n"
                texto += "-" * 42 + "\n"
                if not items:
                    texto += "(sin datos para este numero)\n"
                for i, (a2, p, c) in enumerate(items, 1):
                    ni = analizador.animal_a_num_int.get(a2, '?')
                    texto += f"  {i:2d} {str(ni):>2} ({a2:<14}) {p:>4.1f}% {c:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry.bind("<Return>", lambda e: buscar())
        ttk.Button(frame_top, text="Buscar", command=buscar).pack(side=tk.LEFT, padx=5)

    def _dialogo_full(self):
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
        ventana.title(f"Full (4 fuentes) - {self.current_lottery}")
        ventana.geometry("600x550")
        frame_top = ttk.Frame(ventana)
        frame_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame_top, text="Numero (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry = ttk.Entry(frame_top, width=8, font=("", 10))
        entry.pack(side=tk.LEFT, padx=2)
        entry.focus_set()
        ttk.Label(frame_top, text="Hora:", font=("", 10)).pack(side=tk.LEFT, padx=2)
        combo = ttk.Combobox(frame_top, values=opciones, state="readonly", width=22)
        combo.pack(side=tk.LEFT, padx=2)
        combo.set(opciones[-1])
        txt = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        txt.insert(tk.END, "Selecciona numero y hora, luego clic en Buscar\n")
        def buscar():
            n_str = entry.get().strip()
            seleccion = combo.get()
            if not n_str.isdigit() or not (0 <= int(n_str) <= 37) or not seleccion:
                messagebox.showwarning("Error", "Ingresa un numero valido (0-37) y selecciona hora")
                return
            n = int(n_str)
            animal = analizador.num_int_a_animal.get(n, None)
            if animal is None:
                messagebox.showwarning("Error", f"Numero {n} no valido")
                return
            partes = seleccion.split(" -> ")
            h_o, h_d = partes[0].strip(), partes[1].strip()
            txt.delete("1.0", tk.END)
            txt.insert(tk.END, f"Calculando Full (4 fuentes) para {animal} en {h_o} -> {h_d}...\n")
            def tarea():
                items = analizador.get_prediccion_combinada(datos.copy(), n, h_o, h_d, top_k=25, incluir_trasnocho=False)
                texto = f"PREDICCION FULL (4 fuentes): {animal} | {h_o} -> {h_d}\n"
                texto += f"Combina: Markov Global + Markov x Hora + Frec. x Hora + Co-ocurrencia\n"
                texto += f"\n{'#':>3} {'Num(Animal)':<18} {'Score':>7} {'Fuentes':>8}\n"
                texto += "-" * 42 + "\n"
                if not items:
                    texto += "(sin datos)\n"
                for i, (n2, score, fuentes) in enumerate(items, 1):
                    a2 = analizador.num_int_a_animal.get(n2, "?")
                    texto += f"  {i:2d} {n2:>2} ({a2:<14}) {score:>6.2f} {fuentes:>8}\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
        entry.bind("<Return>", lambda e: buscar())
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
        ttk.Label(frame_top, text="Numero (0-37):", font=("", 10)).pack(side=tk.LEFT, padx=2)
        entry_animal = ttk.Entry(frame_top, width=8, font=("", 10))
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
        txt.insert(tk.END, "Selecciona numero y hora, luego clic en Buscar\n")
        frame_bot = ttk.Frame(ventana)
        frame_bot.pack(fill=tk.X, padx=5, pady=5)
        def buscar():
            n_str = entry_animal.get().strip()
            seleccion = combo.get()
            if not n_str.isdigit() or not (0 <= int(n_str) <= 37) or not seleccion:
                messagebox.showwarning("Error", "Ingresa un numero valido (0-37) y selecciona hora")
                return
            n = int(n_str)
            animal = analizador.num_int_a_animal.get(n, None)
            if animal is None:
                messagebox.showwarning("Error", f"Numero {n} no valido")
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
                    texto += f"{'#':>3} {'Num(Animal)':<18} {'Gral':>5} {'(n)':>4} {'Hora':>5} {'(n)':>4} {'Comb':>5}\n"
                    texto += '-' * 52 + '\n'
                    for i, (a2, p) in enumerate(top, 1):
                        pg, cg = items_g.get(a2, (0, 0))
                        ph, ch = items_h.get(a2, (0, 0))
                        texto += f"  {i:2d} {a2:>2} ({analizador.num_int_a_animal.get(a2, '?'):<14}) {pg:>4.1f}% {cg:>3} {ph:>4.1f}% {ch:>3} {p:>4.1f}%\n"
                ventana.after(0, lambda: (txt.delete("1.0", tk.END), txt.insert(tk.END, texto)))
            threading.Thread(target=_safe_thread(tarea, ventana, txt), daemon=True).start()
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
        ttk.Label(ventana, text="Numero que acaba de salir (0-37):", font=("", 10)).pack(pady=(15, 3))
        entry_animal = ttk.Entry(ventana, width=8, font=("", 10))
        entry_animal.pack(pady=3)
        entry_animal.focus_set()
        ttk.Label(ventana, text="Hora del sorteo (HH:MM AM/PM):", font=("", 10)).pack(pady=(12, 5))
        horas_opciones = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                          "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                          "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo_hora = ttk.Combobox(ventana, values=horas_opciones, state="readonly", width=15, font=("", 10))
        combo_hora.pack(pady=3)
        combo_hora.set(horas_opciones[-1])
        def ejecutar():
            n_str = entry_animal.get().strip()
            hora = combo_hora.get().strip()
            if not n_str.isdigit() or not (0 <= int(n_str) <= 37):
                messagebox.showwarning("Error", "Debes ingresar un numero valido (0-37)")
                return
            n = int(n_str)
            if not hora:
                messagebox.showwarning("Error", "Debes seleccionar una hora")
                return
            ventana.destroy()
            salida, txt = self._crear_ventana_salida("Predicción Markov+Hora", ancho=550, alto=400)
            hilo = threading.Thread(
                target=self._predecir_siguiente_mh_task, args=(n, hora, txt), daemon=True
            )
            hilo.start()
        ttk.Button(ventana, text="Predecir", command=ejecutar).pack(pady=15)

    def _predecir_siguiente_mh_task(self, n, hora_str, txt):
        redir = RedirectText(txt)
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            datos = self._get_datos()
            if datos is None or datos.empty:
                print("ERROR: Sin datos")
                return
            analizador = self._get_analizador()
            if not analizador:
                return
            d = datos.copy()
            if n < 0 or n > 37:
                print(f"ERROR: Numero '{n}' fuera de rango (0-37)")
                return
            animal = analizador.num_int_a_animal.get(n, None)
            if animal is None:
                print(f"ERROR: Numero '{n}' no tiene animal asociado")
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
            d.iloc[-1, d.columns.get_loc('Numero')] = n
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
            salida, txt = self._crear_ventana_salida("Predicción ML", ancho=650, alto=500)
            hilo = threading.Thread(
                target=self._predecir_siguiente_ml_task,
                args=(txt, animal, hora_24h, solo_hora), daemon=True
            )
            hilo.start()

        frame_btn = ttk.Frame(ventana)
        frame_btn.pack(pady=15)
        ttk.Button(frame_btn, text="Predecir", command=ejecutar).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_btn, text="Cancelar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)

    def _predecir_siguiente_ml_task(self, txt, animal=None, hora_24h=None, solo_hora=None):
        redir = RedirectText(txt)
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

    def _predecir_rf_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        rf, le_rf = self.modelos_rf.get(self.current_lottery), self.le_y_rf.get(self.current_lottery)
        if rf is None:
            messagebox.showwarning("Sin modelo", "No hay modelo Random Forest. Entrena RF primero.")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Pred. RF - {self.current_lottery}")
        ventana.geometry("380x200")
        ventana.transient(self.root)
        ventana.grab_set()
        ttk.Label(ventana, text="Animal que acaba de salir:", font=("", 10)).pack(anchor=tk.W, padx=15, pady=(12, 2))
        entry_animal = ttk.Entry(ventana, width=25, font=("", 10))
        entry_animal.pack(anchor=tk.W, padx=15, pady=2)
        entry_animal.focus_set()
        ttk.Label(ventana, text="Hora del sorteo:", font=("", 10)).pack(anchor=tk.W, padx=15, pady=(8, 2))
        horas = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo_hora = ttk.Combobox(ventana, values=horas, state="readonly", width=15, font=("", 10))
        combo_hora.pack(anchor=tk.W, padx=15, pady=2)
        combo_hora.set(horas[-1])
        def ejecutar():
            animal = entry_animal.get().strip().upper()
            if not animal or animal not in analizador.animales_carac:
                messagebox.showwarning("Error", f"Animal invalido")
                return
            hora_str = combo_hora.get()
            dt_h = pd.to_datetime(hora_str, format='%I:%M %p')
            h24 = dt_h.strftime('%H:%M:%S')
            sh = dt_h.strftime('%I:%M %p')
            ventana.destroy()
            salida, txt = self._crear_ventana_salida("Prediccion RF", ancho=650, alto=500)
            threading.Thread(target=self._predecir_rf_task, args=(txt, animal, h24, sh), daemon=True).start()
        frame_b = ttk.Frame(ventana)
        frame_b.pack(pady=15)
        ttk.Button(frame_b, text="Predecir", command=ejecutar).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_b, text="Cancelar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)

    def _predecir_rf_task(self, txt, animal, hora_24h, solo_hora):
        redir = RedirectText(txt)
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            datos = self._get_datos()
            if datos is None or datos.empty:
                print("ERROR: Sin datos"); return
            analizador = self._get_analizador()
            if not analizador: return
            rf, le_rf = self.modelos_rf.get(self.current_lottery), self.le_y_rf.get(self.current_lottery)
            if not rf: print("ERROR: Modelo RF no disponible"); return
            num = analizador.animal_a_num_int.get(animal)
            if num is None:
                print(f"ERROR: Animal '{animal}' no valido"); return
            d = datos.copy()
            d.iloc[-1, d.columns.get_loc('Animal')] = animal
            d.iloc[-1, d.columns.get_loc('Num_Int')] = num
            d.iloc[-1, d.columns.get_loc('Numero')] = num
            d.iloc[-1, d.columns.get_loc('Hora')] = hora_24h
            d.iloc[-1, d.columns.get_loc('Solo_hora')] = solo_hora
            print(f"Prediccion RF desde {animal} a las {solo_hora}")
            analizador.imprimir_prediccion_modelo(d, animal, hora_24h, rf, le_rf, "RF")

    def _predecir_xgb_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        xgb, le_xgb = self.modelos_xgb.get(self.current_lottery), self.le_y_xgb.get(self.current_lottery)
        if xgb is None:
            messagebox.showwarning("Sin modelo", "No hay modelo XGBoost. Entrena XGB primero.")
            return
        analizador = self._get_analizador()
        if not analizador:
            return
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Pred. XGB - {self.current_lottery}")
        ventana.geometry("380x200")
        ventana.transient(self.root)
        ventana.grab_set()
        ttk.Label(ventana, text="Animal que acaba de salir:", font=("", 10)).pack(anchor=tk.W, padx=15, pady=(12, 2))
        entry_animal = ttk.Entry(ventana, width=25, font=("", 10))
        entry_animal.pack(anchor=tk.W, padx=15, pady=2)
        entry_animal.focus_set()
        ttk.Label(ventana, text="Hora del sorteo:", font=("", 10)).pack(anchor=tk.W, padx=15, pady=(8, 2))
        horas = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"]
        combo_hora = ttk.Combobox(ventana, values=horas, state="readonly", width=15, font=("", 10))
        combo_hora.pack(anchor=tk.W, padx=15, pady=2)
        combo_hora.set(horas[-1])
        def ejecutar():
            animal = entry_animal.get().strip().upper()
            if not animal or animal not in analizador.animales_carac:
                messagebox.showwarning("Error", f"Animal invalido")
                return
            hora_str = combo_hora.get()
            dt_h = pd.to_datetime(hora_str, format='%I:%M %p')
            h24 = dt_h.strftime('%H:%M:%S')
            sh = dt_h.strftime('%I:%M %p')
            ventana.destroy()
            salida, txt = self._crear_ventana_salida("Prediccion XGB", ancho=650, alto=500)
            threading.Thread(target=self._predecir_xgb_task, args=(txt, animal, h24, sh), daemon=True).start()
        frame_b = ttk.Frame(ventana)
        frame_b.pack(pady=15)
        ttk.Button(frame_b, text="Predecir", command=ejecutar).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_b, text="Cancelar", command=ventana.destroy).pack(side=tk.LEFT, padx=5)

    def _predecir_xgb_task(self, txt, animal, hora_24h, solo_hora):
        redir = RedirectText(txt)
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            datos = self._get_datos()
            if datos is None or datos.empty:
                print("ERROR: Sin datos"); return
            analizador = self._get_analizador()
            if not analizador: return
            xgb, le_xgb = self.modelos_xgb.get(self.current_lottery), self.le_y_xgb.get(self.current_lottery)
            if not xgb: print("ERROR: Modelo XGB no disponible"); return
            num = analizador.animal_a_num_int.get(animal)
            if num is None:
                print(f"ERROR: Animal '{animal}' no valido"); return
            d = datos.copy()
            d.iloc[-1, d.columns.get_loc('Animal')] = animal
            d.iloc[-1, d.columns.get_loc('Num_Int')] = num
            d.iloc[-1, d.columns.get_loc('Numero')] = num
            d.iloc[-1, d.columns.get_loc('Hora')] = hora_24h
            d.iloc[-1, d.columns.get_loc('Solo_hora')] = solo_hora
            print(f"Prediccion XGB desde {animal} a las {solo_hora}")
            analizador.imprimir_prediccion_modelo(d, animal, hora_24h, xgb, le_xgb, "XGB")

    # ================ TAB: MODELOS ML ================

    def _tab_crear_modelos(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Modelos ML")

        grupo_ml = ttk.LabelFrame(tab, text="🤖 Machine Learning", padding=10)
        grupo_ml.pack(fill=tk.X, padx=10, pady=10)

        row1 = ttk.Frame(grupo_ml)
        row1.pack(fill=tk.X, pady=2)
        row2 = ttk.Frame(grupo_ml)
        row2.pack(fill=tk.X, pady=2)

        ml_btns1 = [
            ("Entrenar RF", self._ml_entrenar_rf_dialogo, "Entrena Random Forest con optimizacion de hiperparametros"),
            ("Entrenar XGB", self._ml_entrenar_xgb_dialogo, "Entrena XGBoost con optimizacion de hiperparametros"),
            ("Auto-Evaluacion", self._ml_auto_eval_dialogo, "Backtesting historico de todos los modelos (ultimos 500)"),
        ]
        for texto, cmd, tip in ml_btns1:
            btn = ttk.Button(row1, text=texto, style='ML.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        ml_btns2 = [
            ("Pred. RF", self._predecir_rf_dialogo, "Predice el siguiente numero usando Random Forest"),
            ("Pred. XGB", self._predecir_xgb_dialogo, "Predice el siguiente numero usando XGBoost"),
        ]
        for texto, cmd, tip in ml_btns2:
            btn = ttk.Button(row2, text=texto, style='ML.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

    def _get_modelos(self):
        rf = self.modelos_rf.get(self.current_lottery)
        le_rf = self.le_y_rf.get(self.current_lottery)
        xgb = self.modelos_xgb.get(self.current_lottery)
        le_xgb = self.le_y_xgb.get(self.current_lottery)
        return rf, le_rf, xgb, le_xgb

    def _ml_entrenar_rf_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        mod = self._get_mod()
        if not analizador or not mod:
            return
        d = datos.copy()
        ventana, txt = self._crear_ventana_salida("Entrenar Random Forest", ancho=700, alto=550)
        self._ejecutar_en_ventana(txt, lambda: self._entrenar_rf(analizador, mod, d))

    def _ml_entrenar_xgb_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        mod = self._get_mod()
        if not analizador or not mod:
            return
        d = datos.copy()
        ventana, txt = self._crear_ventana_salida("Entrenar XGBoost", ancho=700, alto=550)
        self._ejecutar_en_ventana(txt, lambda: self._entrenar_xgb(analizador, mod, d))

    def _ml_auto_eval_dialogo(self):
        datos = self._get_datos()
        if datos is None or datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos cargados")
            return
        analizador = self._get_analizador()
        mod = self._get_mod()
        if not analizador or not mod:
            return
        d = datos.copy()
        rf, le_rf, xgb, le_xgb = self._get_modelos()
        ventana, txt = self._crear_ventana_salida("Auto-Evaluacion", ancho=750, alto=650)
        self._ejecutar_en_ventana(txt, lambda: self._ml_ejecutar_auto_eval(d, analizador, rf, le_rf, xgb, le_xgb))

    def _ml_ejecutar_auto_eval(self, d, analizador, rf, le_rf, xgb, le_xgb):
        d2 = analizador.agregar_caracteristicas_avanzadas(d.copy())
        analizador.evaluar_predicciones_historicas(d2, rf, le_rf, xgb, le_xgb)

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

        grupo_scraper = ttk.LabelFrame(tab, text="🌐 Web Scraper", padding=10)
        grupo_scraper.pack(fill=tk.X, padx=10, pady=10)

        row1 = ttk.Frame(grupo_scraper)
        row1.pack(fill=tk.X, pady=2)
        row2 = ttk.Frame(grupo_scraper)
        row2.pack(fill=tk.X, pady=2)

        scraper_btns = [
            ("Faltantes", self._scraper_faltantes_dialogo, "Scrapea fechas faltantes desde la ultima disponible hasta hoy"),
            ("Todas las Loterias", self._scraper_todas_dialogo, "Scrapea fechas faltantes para TODAS las loterias"),
        ]
        for texto, cmd, tip in scraper_btns:
            btn = ttk.Button(row1, text=texto, style='Scraper.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

        scraper_btns2 = [
            ("Dia Especifico", self._scraper_dialogo_dia, "Scrapea una fecha en especifico (YYYY-MM-DD)"),
            ("Rango de Fechas", self._scraper_dialogo_rango, "Scrapea todas las fechas en un rango"),
        ]
        for texto, cmd, tip in scraper_btns2:
            btn = ttk.Button(row2, text=texto, style='Scraper.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=3)
            ToolTip(btn, tip)

    def _scraper_faltantes_dialogo(self):
        ventana, txt = self._crear_ventana_salida("Scrapear Faltantes", ancho=650, alto=500)
        hilo = threading.Thread(target=self._scraper_faltantes_task, args=(txt,), daemon=True)
        hilo.start()

    def _scraper_faltantes_task(self, txt):
        redir = RedirectText(txt)
        txt.after(0, lambda: txt.delete("1.0", tk.END))
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
                elif nombre == "Lotto Activo RD":
                    from scraper_lotto_activo_rd import scrape_date, save_to_excel
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

    def _scraper_todas_dialogo(self):
        ventana, txt = self._crear_ventana_salida("Scrapear Todas las Loterias", ancho=700, alto=600)
        hilo = threading.Thread(target=self._scraper_todas_task, args=(txt,), daemon=True)
        hilo.start()

    def _scraper_todas_task(self, txt):
        redir = RedirectText(txt)
        txt.after(0, lambda: txt.delete("1.0", tk.END))
        with contextlib.redirect_stdout(redir), contextlib.redirect_stderr(redir):
            scrapers = {
                "Lotto Activo": ("scraper_lotto", "data/LottoActivoINT.xlsx"),
                "La Granjita": ("scraper_la_granjita", "data/LaGranjita.xlsx"),
                "Selva Plus": ("scraper_selva_plus", "data/SelvaPlus.xlsx"),
                "Lotto Activo Rd Int": ("scraper_lotto_rd_int", "data/LottoActivoRDInt.xlsx"),
                "Lotto Activo RD": ("scraper_lotto_activo_rd", "data/LottoActivoRD.xlsx"),
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
        ventana.geometry("350x150")
        ventana.transient(self.root)
        ventana.grab_set()

        ttk.Label(ventana, text="Fecha (YYYY-MM-DD):", font=("", 10)).pack(pady=(10, 0))
        entry = ttk.Entry(ventana, font=("", 10))
        entry.pack(pady=5)
        entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        entry.focus_set()

        def ejecutar():
            fecha = entry.get().strip()
            ventana.destroy()
            salida, txt = self._crear_ventana_salida(f"Scrapear Dia: {fecha}", ancho=600, alto=400)
            hilo = threading.Thread(target=self._scraper_dia_task, args=(fecha, txt), daemon=True)
            hilo.start()

        entry.bind("<Return>", lambda e: ejecutar())
        ttk.Button(ventana, text="Scrapear", command=ejecutar).pack(pady=10)

    def _scraper_dia_task(self, fecha, txt):
        redir = RedirectText(txt)
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
            elif nombre == "Lotto Activo RD":
                from scraper_lotto_activo_rd import scrape_date, save_to_excel
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
        ventana.geometry("350x200")
        ventana.transient(self.root)
        ventana.grab_set()

        ttk.Label(ventana, text="Desde (YYYY-MM-DD):", font=("", 10)).pack(pady=(10, 0))
        entry_desde = ttk.Entry(ventana, font=("", 10))
        entry_desde.pack(pady=2)
        entry_desde.insert(0, "2024-01-01")

        ttk.Label(ventana, text="Hasta (YYYY-MM-DD):", font=("", 10)).pack(pady=(5, 0))
        entry_hasta = ttk.Entry(ventana, font=("", 10))
        entry_hasta.pack(pady=2)
        entry_hasta.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

        def ejecutar():
            desde = entry_desde.get().strip()
            hasta = entry_hasta.get().strip()
            ventana.destroy()
            salida, txt = self._crear_ventana_salida(f"Scrapear Rango: {desde} a {hasta}", ancho=600, alto=400)
            hilo = threading.Thread(target=self._scraper_rango_task, args=(desde, hasta, txt), daemon=True)
            hilo.start()

        entry_hasta.bind("<Return>", lambda e: ejecutar())
        ttk.Button(ventana, text="Scrapear", command=ejecutar).pack(pady=10)

    def _scraper_rango_task(self, desde, hasta, txt):
        redir = RedirectText(txt)
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
            elif nombre == "Lotto Activo RD":
                from scraper_lotto_activo_rd import scrape_range, save_to_excel
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
