"""
Modernized UI interface for lottery prediction system.
This module provides the graphical user interface with improved UX.
"""

import sys
import os
import time
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog

import traceback
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


def _safe_thread(tarea_func, ventana, txt):
    """Wraps a thread tarea_func to catch and display exceptions."""
    def wrapper():
        try:
            tarea_func()
        except Exception:
            err = traceback.format_exc()
            ventana.after(0, lambda: txt.insert(tk.END, f"ERROR:\n{err}\n"))
    return wrapper


class RedirectText:
    """Redirects print statements to a tkinter widget."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, s):
        self.widget.insert(tk.END, s)
        self.widget.see(tk.END)

    def flush(self):
        pass


from utils import ANIMAL_A_NUM_INT, NUM_INT_A_ANIMAL

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


class ModernLottoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Loterías")
        self.root.minsize(800, 600)

        self.current_lottery = "Lotto Activo"
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.modelos_rf = {}
        self.le_y_rf = {}
        self.modelos_xgb = {}
        self.le_y_xgb = {}
        self.modelos_lgb = {}
        self.le_y_lgb = {}

        self._crear_interfaz()
        self._configurar_estilo()
        self._actualizar_info_modelos()

    def _configurar_estilo(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('ML.TButton', font=('', 10, 'bold'), padding=8)
        style.map('ML.TButton',
                  background=[('active', '#45a049'), ('!active', '#4CAF50')],
                  foreground=[('active', 'white'), ('!active', 'white')])

    def _crear_interfaz(self):
        self._crear_pestana_prediccion()
        self._crear_pestana_dashboard()
        self._crear_pestana_scraper()

    def _crear_progress_dialog(self, title):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("500x300")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        txt = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Consolas', 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        return dialog, txt

    def _log_progress(self, txt_widget, msg):
        txt_widget.insert(tk.END, msg + "\n")
        txt_widget.see(tk.END)

    # ── Pestaña Predicción ──────────────────────────────────────────────

    def _crear_pestana_prediccion(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Predicción")
        main_frame = ttk.Frame(tab, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        titulo = ttk.Label(main_frame, text="Predicción de Próximo Sorteo", font=('', 14, 'bold'))
        titulo.pack(pady=(0, 15))
        pred_frame = ttk.LabelFrame(main_frame, text="Predicción Principal", padding=15)
        pred_frame.pack(fill=tk.X, pady=(0, 15))

        # Lottery selector
        row0 = ttk.Frame(pred_frame)
        row0.pack(fill=tk.X, pady=5)
        ttk.Label(row0, text="Lotería:", width=12).pack(side=tk.LEFT)
        loterias = list(LOTTERY_MODULES.keys())
        self.combo_pred_loteria = ttk.Combobox(row0, values=loterias, state="readonly", width=25)
        self.combo_pred_loteria.pack(side=tk.LEFT, padx=5)
        try:
            self.combo_pred_loteria.current(loterias.index(self.current_lottery))
        except Exception:
            self.combo_pred_loteria.current(0)
        def _on_pred_loteria_change(evt=None):
            sel = self.combo_pred_loteria.get()
            if sel:
                self.current_lottery = sel
                self._actualizar_info_modelos()
        self.combo_pred_loteria.bind('<<ComboboxSelected>>', _on_pred_loteria_change)

        # Method
        row1 = ttk.Frame(pred_frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Método:", width=12).pack(side=tk.LEFT)
        self.combo_metodo = ttk.Combobox(row1, values=[
            "Markov", "Markov x Hora", "Markov Orden 2",
            "Frecuencia x Hora",
            "Markov x Dia", "Markov x Hora + Frecuencia x Hora",
            "Random Forest", "XGBoost", "LightGBM"
        ], state="readonly", width=37)
        self.combo_metodo.pack(side=tk.LEFT, padx=5)
        self.combo_metodo.current(1)

        ttk.Label(row1, text="Top:").pack(side=tk.LEFT, padx=(5, 0))
        self.combo_top_k = ttk.Combobox(row1, values=[
            "Top 5", "Top 10", "Top 15", "Top 20", "Top 25"
        ], state="readonly", width=8)
        self.combo_top_k.pack(side=tk.LEFT, padx=5)
        self.combo_top_k.current(4)

        # Train buttons next to method
        self.btn_entrenar_rf = ttk.Button(row1, text="Entrenar RF",
                                          command=lambda: self._entrenar_modelo_con_progreso('rf'))
        self.btn_entrenar_rf.pack(side=tk.LEFT, padx=2)
        self.btn_entrenar_xgb = ttk.Button(row1, text="Entrenar XGB",
                                           command=lambda: self._entrenar_modelo_con_progreso('xgb'))
        self.btn_entrenar_xgb.pack(side=tk.LEFT, padx=2)
        self.btn_entrenar_lgb = ttk.Button(row1, text="Entrenar LGB",
                                           command=lambda: self._entrenar_modelo_con_progreso('lgb'))
        self.btn_entrenar_lgb.pack(side=tk.LEFT, padx=2)

        # Info labels for last training date
        info_row = ttk.Frame(pred_frame)
        info_row.pack(fill=tk.X, pady=(0, 2))
        self.info_rf_label = ttk.Label(info_row, text="RF: --", font=('', 8), foreground="gray")
        self.info_rf_label.pack(side=tk.LEFT, padx=(12, 30))
        self.info_xgb_label = ttk.Label(info_row, text="XGB: --", font=('', 8), foreground="gray")
        self.info_xgb_label.pack(side=tk.LEFT, padx=0)
        self.info_lgb_label = ttk.Label(info_row, text="LGB: --", font=('', 8), foreground="gray")
        self.info_lgb_label.pack(side=tk.LEFT, padx=0)

        # Animal/Num input
        row2 = ttk.Frame(pred_frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="Animal/Num:", width=12).pack(side=tk.LEFT)
        self.entry_animal = ttk.Entry(row2)
        self.entry_animal.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(row2, text="Hora:", width=8).pack(side=tk.LEFT)
        self.combo_hora = ttk.Combobox(row2, values=["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                                                    "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                                                    "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"],
                                        state="readonly")
        self.combo_hora.pack(side=tk.LEFT, padx=5)
        self.combo_hora.current(0)

        btn_frame = ttk.Frame(pred_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        self.btn_predecir = ttk.Button(btn_frame, text="Predecir", command=self._predecir_principal, style='ML.TButton')
        self.btn_predecir.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Ver Top por Hora", command=self._ver_top_por_hora_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Limpiar", command=self._limpiar_campos).pack(side=tk.LEFT, padx=5)

        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        tree_frame = ttk.Frame(results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("#", "Num", "Animal", "Score", "Markov%", "+Hora%")
        self.tree_resultados = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        for col in columns:
            self.tree_resultados.heading(col, text=col)
            self.tree_resultados.column(col, width=60, anchor="center")
        self.tree_resultados.column("Animal", width=120, anchor="w")
        self.tree_resultados.column("#", width=40)
        self.tree_resultados.column("Score", width=70)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree_resultados.yview)
        self.tree_resultados.configure(yscrollcommand=vsb.set)
        self.tree_resultados.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        for col in columns:
            self.tree_resultados.heading(col, text=col, command=lambda c=col: self._tree_sort_by(self.tree_resultados, c, False))

    def _actualizar_info_modelos(self):
        """Lee la fecha del último modelo entrenado y actualiza las etiquetas."""
        try:
            analizador = self._get_analizador()
            for tipo, label in [('random_forest', self.info_rf_label), ('xgboost', self.info_xgb_label), ('lightgbm', self.info_lgb_label)]:
                try:
                    modelo_dir = None
                    search_dirs = [analizador.config.get('modelos_dir', 'modelos'), 'modelos']
                    for sd in search_dirs:
                        if not os.path.exists(sd):
                            continue
                        dirs = [d for d in os.listdir(sd) if d.startswith(tipo)]
                        if dirs:
                            dirs.sort(reverse=True)
                            modelo_dir = os.path.join(sd, dirs[0])
                            break
                    if modelo_dir:
                        info_path = os.path.join(modelo_dir, 'info.json')
                        if os.path.exists(info_path):
                            with open(info_path) as f:
                                info = json.load(f)
                            fec = info.get('fecha_entrenamiento', 'desconocida')
                            # Convert YYYYMMDD_HHMMSS to readable
                            try:
                                dt = datetime.datetime.strptime(fec, "%Y%m%d_%H%M%S")
                                fec = dt.strftime("%d/%m/%Y %H:%M")
                            except Exception:
                                pass
                            label.config(text=f"{tipo.replace('_',' ').title()}: {fec}", foreground="gray")
                        else:
                            label.config(text=f"{tipo.replace('_',' ').title()}: fecha no disponible", foreground="gray")
                    else:
                        label.config(text=f"{tipo.replace('_',' ').title()}: nunca entrenado", foreground="gray")
                except Exception:
                    label.config(text=f"{tipo.replace('_',' ').title()}: error", foreground="red")
        except Exception:
            pass

    def _entrenar_modelo_con_progreso(self, tipo):
        dialog, txt = self._crear_progress_dialog(f"Entrenando {tipo.upper()}...")
        analizador = self._get_analizador()
        if analizador is None:
            dialog.destroy()
            return
        loteria = self.combo_pred_loteria.get()
        excel = self._get_excel_path(loteria)

        def entrenar():
            old_stdout = sys.stdout
            sys.stdout = RedirectText(txt)
            try:
                self._log_progress(txt, f"Iniciando entrenamiento {tipo.upper()} para {loteria}...")
                self._log_progress(txt, "Cargando datos...")
                datos = pd.read_excel(excel)
                self._log_progress(txt, f"Datos cargados: {len(datos)} registros")
                from utils import load_and_prepare_data
                datos = load_and_prepare_data(excel, analizador)
                self._log_progress(txt, "Datos preparados con features avanzadas.")
                self._log_progress(txt, "Iniciando optimización de hiperparámetros (esto puede tomar varios minutos)...")
                if tipo == 'rf':
                    resultado = analizador.random_forest_optimizado(datos)
                elif tipo == 'lgb':
                    resultado = analizador.lightgbm_optimizado(datos)
                else:
                    resultado = analizador.xgboost_optimizado(datos)
                if resultado is not None:
                    self._log_progress(txt, "Modelo entrenado y guardado exitosamente.")
                else:
                    self._log_progress(txt, "ERROR: No se pudo entrenar el modelo (datos insuficientes o error).")
            except Exception as e:
                self._log_progress(txt, f"ERROR: {e}")
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                self._log_progress(txt, "\nProceso completado. Puede cerrar esta ventana.")
                self.root.after(0, self._actualizar_info_modelos)

        t = threading.Thread(target=_safe_thread(entrenar, dialog, txt), daemon=True)
        t.start()

    def _tree_sort_by(self, tree, col, descending):
        data = [(tree.set(child, col), child) for child in tree.get_children('')]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=descending)
        except Exception:
            data.sort(key=lambda t: t[0], reverse=descending)
        for i, item in enumerate(data):
            tree.move(item[1], '', i)
        tree.heading(col, command=lambda: self._tree_sort_by(tree, col, not descending))

    def _limpiar_campos(self):
        self.entry_animal.delete(0, tk.END)
        self.tree_resultados.delete(*self.tree_resultados.get_children())

    def _mostrar_prediccion_ml_en_tree(self, prediccion, analizador, nombre_modelo):
        """Agrega resultados de un modelo ML (RF/XGB/LGB) al tree_resultados.
        prediccion: dict hora -> [lista de ints 0-37 en orden de probabilidad]
        """
        self.tree_resultados.delete(*self.tree_resultados.get_children())
        if not prediccion:
            return
        from collections import defaultdict
        scores = defaultdict(float)
        freq = defaultdict(int)
        total_horas = len(prediccion)
        for hora, nums in prediccion.items():
            for pos, n in enumerate(nums[:25]):
                peso = 25 - pos
                scores[int(n)] += peso
                freq[int(n)] += 1
        top = sorted(scores.items(), key=lambda x: (-x[1], -freq[x[0]]))[:25]
        for rank, (num, score) in enumerate(top, 1):
            animal = analizador.num_int_a_animal.get(num, "?")
            pct_hora = freq[num] / total_horas * 100
            self.tree_resultados.insert('', 'end', values=(
                rank, num, animal, round(score, 1), '--', f'{pct_hora:.0f}%'
            ))
        print(f"Top 25 predicciones {nombre_modelo} (agregadas de {total_horas} horas):")
        for rank, (num, score) in enumerate(top, 1):
            animal = analizador.num_int_a_animal.get(num, "?")
            print(f"  {rank:2d}. {num:2d} ({animal}) - Score: {score:.1f}")

    def _predecir_principal(self):
        self.tree_resultados.delete(*self.tree_resultados.get_children())
        metodo = self.combo_metodo.get()
        entrada = self.entry_animal.get().strip()
        hora_sel = self.combo_hora.get()
        top_k_str = self.combo_top_k.get()
        top_k_map = {"Top 5": 5, "Top 10": 10, "Top 15": 15, "Top 20": 20, "Top 25": 25}
        top_k = top_k_map.get(top_k_str, 25)
        try:
            sel_loteria = self.combo_pred_loteria.get()
            if sel_loteria:
                self.current_lottery = sel_loteria
        except Exception:
            pass
        analizador = self._get_analizador()
        if analizador is None:
            return

        try:
            print(f"=== Predicción por {metodo} ===")
            print(f"Loteria: {self.current_lottery}")
            if entrada:
                print(f"Consulta: {entrada}")
            print("-" * 50)

            excel = self._get_excel_path(self.current_lottery)
            if not os.path.exists(excel):
                print(f"Archivo no encontrado: {excel}")
                return
            from utils import load_and_prepare_data
            datos = load_and_prepare_data(excel, analizador)
            if datos is None or len(datos) < 2:
                print("Datos insuficientes para predicción.")
                return

            if metodo == "Markov":
                analizador.generar_matriz_probabilidad(datos)
                ultimo = datos.iloc[-1]
                ultimo_num = int(ultimo['Num_Int'])
                print(f"\nMatriz de transición generada. Último número: {ultimo_num}")
                print("(Usar consola para ver detalles completos)")

            elif metodo == "Markov Orden 2":
                analizador.generar_prediccion_markov_segundo_orden(datos, top_k=top_k)

            elif metodo == "Markov x Hora":
                data = analizador.generar_prediccion_markov(datos, top_k=top_k)
                self.last_prediccion_data = data
                self.tree_resultados.delete(*self.tree_resultados.get_children())
                for item in data.get('top', []):
                    self.tree_resultados.insert('', 'end', values=(
                        item['rank'], item['num'], item['animal'],
                        item['score'], item['markov'], item['hora_pct']
                    ))
                por_hora = data.get('por_hora', {})
                lines = []
                for h, lista in por_hora.items():
                    lines.append(f"{h}:")
                    for it in lista:
                        lines.append(f"  #{it['num']:2d} {it['animal']:<14} Score: {it['score']}")
                    lines.append("")
                print('\n'.join(lines))

            elif metodo == "Frecuencia x Hora":
                analizador.probabilidad_maxima_por_hora(datos)

            elif metodo == "Markov x Dia":
                analizador.prediccion_markov_dia_semana(datos, top_k=top_k)

            elif metodo == "Markov x Hora + Frecuencia x Hora":
                data = analizador.generar_prediccion_markov(datos, top_k=top_k)
                self.last_prediccion_data = data
                self.tree_resultados.delete(*self.tree_resultados.get_children())
                for item in data.get('top', []):
                    self.tree_resultados.insert('', 'end', values=(
                        item['rank'], item['num'], item['animal'],
                        item['score'], item['markov'], item['hora_pct']
                    ))
                por_hora = data.get('por_hora', {})
                lines = []
                for h, lista in por_hora.items():
                    lines.append(f"{h}:")
                    for it in lista:
                        lines.append(f"  #{it['num']:2d} {it['animal']:<14} Score: {it['score']}")
                    lines.append("")
                print('\n'.join(lines))

            elif metodo == "Random Forest":
                modelo, le_y, metricas = analizador.cargar_ultimo_modelo('random_forest')
                if modelo is None:
                    print("No hay modelo Random Forest guardado. Use 'Entrenar RF' primero.")
                else:
                    fec = metricas.get('fecha_entrenamiento', 'desconocida')
                    muestras = metricas.get('num_muestras', 'N/A')
                    print(f"Modelo RF entrenado: {fec} ({muestras} muestras)")
                    print("-" * 50)
                    prediccion = analizador.predecir_top_k_por_hora(modelo, le_y, datos, k=25)
                    self._mostrar_prediccion_ml_en_tree(prediccion, analizador, "Random Forest")

            elif metodo == "XGBoost":
                modelo, le_y, metricas = analizador.cargar_ultimo_modelo('xgboost')
                if modelo is None:
                    print("No hay modelo XGBoost guardado. Use 'Entrenar XGB' primero.")
                else:
                    fec = metricas.get('fecha_entrenamiento', 'desconocida')
                    muestras = metricas.get('num_muestras', 'N/A')
                    print(f"Modelo XGB entrenado: {fec} ({muestras} muestras)")
                    print("-" * 50)
                    prediccion = analizador.predecir_top_k_por_hora(modelo, le_y, datos, k=25)
                    self._mostrar_prediccion_ml_en_tree(prediccion, analizador, "XGBoost")

            elif metodo == "LightGBM":
                modelo, le_y, metricas = analizador.cargar_ultimo_modelo('lightgbm')
                if modelo is None:
                    print("No hay modelo LightGBM guardado. Use 'Entrenar LGB' primero.")
                else:
                    fec = metricas.get('fecha_entrenamiento', 'desconocida')
                    muestras = metricas.get('num_muestras', 'N/A')
                    print(f"Modelo LGB entrenado: {fec} ({muestras} muestras)")
                    print("-" * 50)
                    prediccion = analizador.predecir_top_k_por_hora(modelo, le_y, datos, k=25)
                    self._mostrar_prediccion_ml_en_tree(prediccion, analizador, "LightGBM")

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    def _ver_top_por_hora_dialog(self):
        data = getattr(self, 'last_prediccion_data', None)
        if not data:
            messagebox.showinfo('Info', 'Primero genere una predicción (Presione "Predecir" con Markov x Hora).')
            return
        dialog = tk.Toplevel(self.root)
        dialog.title('Top por Hora')
        dialog.geometry('500x400')
        row = ttk.Frame(dialog, padding=8)
        row.pack(fill=tk.X)
        ttk.Label(row, text='Hora:').pack(side=tk.LEFT)
        horas = list(data.get('por_hora', {}).keys())
        combo = ttk.Combobox(row, values=horas, state='readonly')
        combo.pack(side=tk.LEFT, padx=8)
        tree = ttk.Treeview(dialog, columns=('Num', 'Animal', 'Score'), show='headings')
        tree.heading('Num', text='Num')
        tree.heading('Animal', text='Animal')
        tree.heading('Score', text='Score')
        tree.pack(fill=tk.BOTH, expand=True, pady=8)
        def on_select(event=None):
            h = combo.get()
            tree.delete(*tree.get_children())
            lista = data.get('por_hora', {}).get(h, [])
            for it in lista:
                tree.insert('', 'end', values=(it['num'], it['animal'], it['score']))
        combo.bind('<<ComboboxSelected>>', on_select)
        if horas:
            combo.current(0)
            on_select()

    # ── Pestaña Dashboard ───────────────────────────────────────────────

    def _crear_pestana_dashboard(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dashboard")
        main_frame = ttk.Frame(tab, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        titulo = ttk.Label(main_frame, text="Dashboard de Análisis", font=('', 14, 'bold'))
        titulo.pack(pady=(0, 15))

        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding=15)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        row = ttk.Frame(control_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Loteria:", width=10).pack(side=tk.LEFT)
        loterias = list(LOTTERY_MODULES.keys())
        self.combo_dash_loteria = ttk.Combobox(row, values=loterias, state="readonly", width=25)
        self.combo_dash_loteria.pack(side=tk.LEFT, padx=5)
        self.combo_dash_loteria.current(loterias.index(self.current_lottery))

        # Buttons row 1 - basic analysis
        btn_row = ttk.Frame(control_frame)
        btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row, text="Aciertos por Día", command=lambda: self._analisis_dash('aciertos_dia')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Aciertos por Hora", command=lambda: self._analisis_dash('aciertos_hora')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Aciertos por Mes", command=lambda: self._analisis_dash('aciertos_mes')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Fallos por Día", command=lambda: self._analisis_dash('fallos_dia')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Fallos por Hora", command=lambda: self._analisis_dash('fallos_hora')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Fallos por Mes", command=lambda: self._analisis_dash('fallos_mes')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Resultados del Día", command=self._ver_resultados_dia).pack(side=tk.LEFT, padx=2)

        # Area de texto para resultados de análisis
        analisis_frame = ttk.LabelFrame(main_frame, text="Resultados del Análisis", padding=5)
        analisis_frame.pack(fill=tk.BOTH, expand=True)
        self.text_dashboard = scrolledtext.ScrolledText(analisis_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.text_dashboard.pack(fill=tk.BOTH, expand=True)

    def _analisis_dash(self, tipo):
        self.text_dashboard.delete(1.0, tk.END)
        loteria_name = self.combo_dash_loteria.get()
        analizador = self._get_analizador()
        if analizador is None:
            return
        excel = self._get_excel_path(loteria_name)
        if not excel or not os.path.exists(excel):
            self.text_dashboard.insert(tk.END, f"Archivo no encontrado: {excel}\n")
            return
        try:
            df = pd.read_excel(excel)
            if df.empty:
                self.text_dashboard.insert(tk.END, "No hay datos.\n")
                return
            from utils import load_and_prepare_data
            datos = load_and_prepare_data(excel, analizador)
            if datos is None or len(datos) < 2:
                self.text_dashboard.insert(tk.END, "Datos insuficientes.\n")
                return
            old_stdout = sys.stdout
            sys.stdout = RedirectText(self.text_dashboard)
            try:
                print(f"=== {loteria_name} ===\n")
                if tipo == 'aciertos_dia':
                    analizador.analizar_aciertos_por_dia_semana(datos, top_k=25)
                elif tipo == 'aciertos_hora':
                    analizador.analizar_aciertos_por_hora(datos, top_k=25)
                elif tipo == 'aciertos_mes':
                    self._analisis_por_mes(datos, 'aciertos')
                elif tipo == 'fallos_dia':
                    analizador.analizar_fallos_por_dia_semana(datos, top_k=25)
                elif tipo == 'fallos_hora':
                    analizador.analizar_fallos_por_hora(datos, top_k=25)
                elif tipo == 'fallos_mes':
                    self._analisis_por_mes(datos, 'fallos')
            finally:
                sys.stdout = old_stdout
        except Exception as e:
            self.text_dashboard.insert(tk.END, f"ERROR: {e}\n")
            traceback.print_exc()

    def _analisis_por_mes(self, datos, modo):
        """Analiza aciertos/fallos agrupados por mes."""
        df = datos.copy()
        df['Mes'] = pd.to_datetime(df['Fecha'].astype(str)).dt.to_period('M').astype(str)
        from collections import defaultdict
        meses = defaultdict(lambda: {'aciertos': 0, 'fallos': 0, 'total': 0})
        # Reuse the Markov validation logic at per-month groupings with a simple top-25 heuristic
        trans_prob, trans_total = self._get_analizador()._transiciones_markov(df) if hasattr(self._get_analizador(), '_transiciones_markov') else ({}, {})
        # simpler approach: just show frequency per month
        print(f"\n{'='*60}")
        print(f"  ANALISIS POR MES - {'ACIERTOS' if modo == 'aciertos' else 'FALLOS'} (Top-25)")
        print(f"{'='*60}")
        for mes, grupo in df.groupby('Mes'):
            total = len(grupo)
            freq = grupo['Num_Int'].value_counts()
            top25 = set(freq.head(25).index)
            aciertos = sum(1 for _, r in grupo.iterrows() if int(r['Num_Int']) in top25)
            fallos = total - aciertos
            tasa = (aciertos / total) * 100 if total > 0 else 0
            if modo == 'aciertos':
                print(f"MES: {mes}")
                print(f"  Total sorteos: {total}")
                print(f"  Aciertos (en top-25): {aciertos}")
                print(f"  Tasa de acierto: {tasa:.1f}%")
            else:
                print(f"MES: {mes}")
                print(f"  Total sorteos: {total}")
                print(f"  Fallos (fuera de top-25): {fallos}")
                print(f"  Tasa de fallo: {100-tasa:.1f}%")
            print(f"  {'─'*40}")

    def _ver_resultados_dia(self):
        """Muestra los sorteos registrados hoy (reemplaza el Ver Estado del scraper)."""
        self.text_dashboard.delete(1.0, tk.END)
        loteria_name = self.combo_dash_loteria.get()
        excel = self._get_excel_path(loteria_name)
        if not excel or not os.path.exists(excel):
            self.text_dashboard.insert(tk.END, f"Archivo no encontrado: {excel}\n")
            return
        try:
            df = pd.read_excel(excel)
            if df.empty:
                self.text_dashboard.insert(tk.END, "No hay datos en el archivo.\n")
                return
            last = df.tail(10)
            self.text_dashboard.insert(tk.END, f"Archivo: {excel}\n")
            self.text_dashboard.insert(tk.END, f"Últimos 10 registros:\n")
            self.text_dashboard.insert(tk.END, last.to_string(index=False))
            self.text_dashboard.insert(tk.END, "\n\n")
            try:
                df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
                today = datetime.date.today()
                df_today = df[df['Fecha'] == today]
                if df_today.empty:
                    self.text_dashboard.insert(tk.END, f"No hay sorteos registrados para hoy ({today}).\n")
                else:
                    registros_hoy = df_today[['Hora', 'Animal', 'Numero']].sort_values('Hora')
                    self.text_dashboard.insert(tk.END, f"SORTEOS REGISTRADOS HOY ({today}) ({len(registros_hoy)}):\n")
                    self.text_dashboard.insert(tk.END, registros_hoy.to_string(index=False))
                    self.text_dashboard.insert(tk.END, "\n")
                    animales = registros_hoy['Animal'].unique().tolist()
                    self.text_dashboard.insert(tk.END, f"Animales registrados hoy: {animales}\n")
                    horas_posibles = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00',
                                     '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00',
                                     '18:00:00', '19:00:00']
                    horas_registradas = registros_hoy['Hora'].astype(str).tolist()
                    faltantes = [h for h in horas_posibles if h not in horas_registradas]
                    if faltantes:
                        self.text_dashboard.insert(tk.END, f"Horas faltantes hoy: {faltantes}\n")
                    else:
                        self.text_dashboard.insert(tk.END, "TODAS LAS HORAS DE HOY ESTAN COMPLETAS!\n")
            except Exception:
                pass
        except Exception as e:
            self.text_dashboard.insert(tk.END, f"ERROR: {e}\n")

    # ── Pestaña Scraper ─────────────────────────────────────────────────

    def _crear_pestana_scraper(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Scraper")
        main_frame = ttk.Frame(tab, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        titulo = ttk.Label(main_frame, text="Extracción de Datos (Scraper)", font=('', 14, 'bold'))
        titulo.pack(pady=(0, 15))
        controls_frame = ttk.LabelFrame(main_frame, text="Controles", padding=15)
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        row = ttk.Frame(controls_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Loteria:", width=10).pack(side=tk.LEFT)
        loterias = list(LOTTERY_MODULES.keys())
        self.combo_scraper_loteria = ttk.Combobox(row, values=loterias, state="readonly", width=25)
        self.combo_scraper_loteria.pack(side=tk.LEFT, padx=5)
        self.combo_scraper_loteria.current(loterias.index(self.current_lottery))

        btn_row2 = ttk.Frame(controls_frame)
        btn_row2.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row2, text="Scrapear por Día", command=self._scrapear_por_dia).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row2, text="Scrapear por Rango", command=self._scrapear_rango).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row2, text="Agregar Sorteo Manual", command=self._agregar_sorteo_manual).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=15)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.text_scraper = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.text_scraper.pack(fill=tk.BOTH, expand=True)

    def _scrapear_con_progreso(self, titulo, start, end):
        dialog, txt = self._crear_progress_dialog(titulo)
        import scrape_all
        scrape_all.import_scrapers()
        loteria_name = self.combo_scraper_loteria.get()
        lot = None
        for l in scrape_all.LOTTERIES:
            if l.get('name') == loteria_name:
                lot = l
                break
        if lot is None:
            txt.insert(tk.END, f"No se encontró la lotería: {loteria_name}\n")
            return

        def tarea():
            old_stdout = sys.stdout
            sys.stdout = RedirectText(txt)
            try:
                if loteria_name == "Lotto Activo Unificado":
                    print(f"Scrapeando fuentes individuales para Lotto Activo Unificado...")
                    sources = ["Lotto Activo", "Lotto Activo Rd Int", "Lotto Activo RD"]
                    for l in scrape_all.LOTTERIES:
                        if l.get('name') in sources:
                            scrape_all.scrape_lottery(l, start, end)
                            time.sleep(1)
                    import lotto_activo_unificado
                    lotto_activo_unificado.regenerar_cache()
                    print("\nScraping y regeneración de cache completados.")
                else:
                    print(f"Iniciando scraper para: {loteria_name} desde {start} hasta {end}")
                    scrape_all.scrape_lottery(lot, start, end)
                    print("\nScraping completado.")
                # Show summary of what was found for the scraped dates
                try:
                    excel = self._get_excel_path(loteria_name)
                    if excel and os.path.exists(excel):
                        df = pd.read_excel(excel)
                        if not df.empty:
                            df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
                            if start == end:
                                # Single day summary
                                target = datetime.datetime.strptime(start, "%Y-%m-%d").date()
                                registros = df[df['Fecha'] == target]
                                if not registros.empty:
                                    print(f"\n{'='*50}")
                                    print(f"  DATOS ENCONTRADOS PARA {start}:")
                                    print(f"{'='*50}")
                                    for _, r in registros.iterrows():
                                        num = r.get('Numero', '?')
                                        animal = r.get('Animal', '?')
                                        print(f"  {r['Hora']} → {num} ({animal})")
                                    # Check missing hours
                                    horas_esperadas = [f"{h:02d}:00:00" for h in range(8, 20)]
                                    horas_encontradas = set(registros['Hora'].values)
                                    faltantes = [h for h in horas_esperadas if h not in horas_encontradas]
                                    if faltantes:
                                        print(f"\n  ⚠ Horas faltantes: {', '.join(faltantes)}")
                                    else:
                                        print(f"\n  ✓ TODAS LAS HORAS COMPLETAS!")
                                else:
                                    print(f"\n  No se encontraron datos para {start}.")
                            else:
                                # Range summary
                                target_start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
                                target_end = datetime.datetime.strptime(end, "%Y-%m-%d").date()
                                registros = df[(df['Fecha'] >= target_start) & (df['Fecha'] <= target_end)]
                                if not registros.empty:
                                    total_dias = (target_end - target_start).days + 1
                                    print(f"\n{'-'*50}")
                                    print(f"  Rango: {start} a {end} ({total_dias} días)")
                                    print(f"  Registros scrapeados: {len(registros)}")
                                    print(f"  Fechas con datos: {len(registros['Fecha'].unique())}/{total_dias}")
                except Exception as e_sum:
                    print(f"\n(No se pudo mostrar resumen: {e_sum})")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                print("\nPuede cerrar esta ventana.")

        t = threading.Thread(target=_safe_thread(tarea, dialog, txt), daemon=True)
        t.start()

    def _scrapear_por_dia(self):
        hoy = datetime.date.today().strftime("%Y-%m-%d")
        dia = simpledialog.askstring('Scrapear por Día', 'Fecha (YYYY-MM-DD):', initialvalue=hoy, parent=self.root)
        if not dia:
            return
        self._scrapear_con_progreso(f"Scrapeando {dia}...", dia, dia)

    def _scrapear_rango(self):
        hoy = datetime.date.today()
        default_start = (hoy - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        start = simpledialog.askstring('Fecha inicio', 'Fecha inicio (YYYY-MM-DD):', initialvalue=default_start, parent=self.root)
        if not start:
            return
        end = simpledialog.askstring('Fecha fin', 'Fecha fin (YYYY-MM-DD):', initialvalue=hoy.strftime('%Y-%m-%d'), parent=self.root)
        if not end:
            return
        self._scrapear_con_progreso("Scrapeando por Rango...", start, end)

    def _agregar_sorteo_manual(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Agregar Sorteo Manual")
        dialog.geometry("350x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        main = ttk.Frame(dialog, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Agregar Sorteo Manual", font=('', 12, 'bold')).pack(pady=(0, 15))

        ttk.Label(main, text="Fecha (YYYY-MM-DD):").pack(anchor=tk.W)
        entry_fecha = ttk.Entry(main)
        entry_fecha.pack(fill=tk.X, pady=(0, 8))
        entry_fecha.insert(0, datetime.date.today().strftime("%Y-%m-%d"))

        ttk.Label(main, text="Hora (HH:MM, ej: 14:00):").pack(anchor=tk.W)
        entry_hora = ttk.Entry(main)
        entry_hora.pack(fill=tk.X, pady=(0, 8))
        entry_hora.insert(0, "")

        ttk.Label(main, text="Número o Animal:").pack(anchor=tk.W)
        entry_valor = ttk.Entry(main)
        entry_valor.pack(fill=tk.X, pady=(0, 15))
        entry_valor.insert(0, "")

        msg_label = ttk.Label(main, text="", foreground="green")
        msg_label.pack()

        def guardar():
            fecha_str = entry_fecha.get().strip()
            hora_str = entry_hora.get().strip()
            valor = entry_valor.get().strip()
            if not fecha_str or not hora_str or not valor:
                msg_label.config(text="Todos los campos son obligatorios.", foreground="red")
                return
            # Validate date
            try:
                fecha_dt = datetime.datetime.strptime(fecha_str, "%Y-%m-%d").date()
            except Exception:
                msg_label.config(text="Fecha inválida. Use YYYY-MM-DD.", foreground="red")
                return
            # Resolve number/animal
            num_int = None
            try:
                num_int = int(valor)
                if num_int < 0 or num_int > 37:
                    msg_label.config(text="Número debe ser 0-37.", foreground="red")
                    return
            except ValueError:
                animal_key = valor.strip().upper()
                if animal_key in ANIMAL_A_NUM_INT:
                    num_int = ANIMAL_A_NUM_INT[animal_key]
                else:
                    msg_label.config(text=f"Animal '{valor}' no reconocido.", foreground="red")
                    return

            # Build the row
            animal_name = NUM_INT_A_ANIMAL.get(num_int, "?")
            timestamp_dt = datetime.datetime.strptime(f"{fecha_str} {hora_str}", "%Y-%m-%d %H:%M")
            new_row = pd.DataFrame([{
                'Fecha': fecha_dt,
                'Hora': hora_str + ":00",
                'Animal': animal_name,
                'Numero': num_int,
                'Num_Int': num_int,
                'Timestamp': timestamp_dt,
                'Solo_hora': timestamp_dt.strftime('%I:%M %p'),
            }])

            loteria_name = self.combo_scraper_loteria.get()
            excel = self._get_excel_path(loteria_name)
            if not excel:
                msg_label.config(text="No se pudo determinar el archivo Excel.", foreground="red")
                return
            try:
                if os.path.exists(excel):
                    df_existente = pd.read_excel(excel)
                    df_final = pd.concat([df_existente, new_row], ignore_index=True)
                else:
                    df_final = new_row
                df_final = df_final.sort_values(['Fecha', 'Hora']).reset_index(drop=True)
                df_final.to_excel(excel, index=False)
                msg_label.config(text=f"Sorteo guardado: {animal_name} ({num_int}) a las {hora_str} el {fecha_str}",
                                 foreground="green")
            except Exception as e:
                msg_label.config(text=f"Error al guardar: {e}", foreground="red")

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Guardar", command=guardar, style='ML.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cerrar", command=dialog.destroy).pack(side=tk.LEFT, padx=5)


    # ── Helper methods ──────────────────────────────────────────────────

    def _get_analizador(self):
        try:
            loteria_map = {
                "Lotto Activo": lotto_activo,
                "La Granjita": la_granjita,
                "Selva Plus": selva_plus,
                "Lotto Activo Rd Int": lotto_rd_int,
                "Lotto Activo RD": lotto_activo_rd,
                "Lotto Activo Unificado": lotto_activo_unificado,
            }
            mod = loteria_map.get(self.current_lottery)
            if mod:
                return mod.analizador
        except Exception as e:
            messagebox.showerror("Error", f"Error obteniendo analizador: {e}")
            return None
        return None

    def _get_excel_path(self, loteria_name):
        import la_granjita as _lg
        import lotto_activo as _la
        import selva_plus as _sp
        import lotto_rd_int as _ri
        import lotto_activo_rd as _ar
        import lotto_activo_unificado as _au
        map_config = {
            "Lotto Activo": _la.CONFIG,
            "La Granjita": _lg.CONFIG,
            "Selva Plus": _sp.CONFIG,
            "Lotto Activo Rd Int": _ri.CONFIG,
            "Lotto Activo RD": _ar.CONFIG,
            "Lotto Activo Unificado": _au.CONFIG,
        }
        cfg = map_config.get(loteria_name)
        if cfg:
            return cfg['excel_file']
        return ""


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernLottoUI(root)
    root.mainloop()
