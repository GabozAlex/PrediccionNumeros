"""
Modernized UI interface for lottery prediction system.
This module provides the graphical user interface with improved UX.
"""

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
            ventana.after(0, lambda: txt.insert(tk.END, f"ERROR:\n{err}\n"))
    return wrapper


class RedirectText:
    """Redirects print statements to a tkinter widget."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, s):
        if s.strip():
            self.widget.insert(tk.END, s)
            self.widget.see(tk.END)

    def flush(self):
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
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffcc", relief=tk.SOLID, borderwidth=1,
                         padx=5, pady=2, font=("", 8))
        label.pack()

    def leave(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


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


class ModernLottoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Loterías")
        # Do not force initial geometry; allow the window manager to choose.
        self.root.minsize(800, 600)

        self.current_lottery = "Lotto Activo"
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.modelos_rf = {}
        self.le_y_rf = {}
        self.modelos_xgb = {}
        self.le_y_xgb = {}

        self._crear_interfaz()
        self._configurar_estilo()

    def _configurar_estilo(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('ML.TButton', font=('', 10, 'bold'), padding=8)
        style.map('ML.TButton',
                  background=[('active', '#45a049'), ('!active', '#4CAF50')],
                  foreground=[('active', 'white'), ('!active', 'white')])
        style.configure('Dashboard.TFrame', background='#f5f5f5')
        style.configure('Dashboard.TLabel', background='#f5f5f5', font=('', 10))

    def _crear_interfaz(self):
        self._crear_pestana_prediccion()
        self._crear_pestana_dashboard()
        self._crear_pestana_modelos()
        self._crear_pestana_scraper()

    def _crear_pestana_prediccion(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Predicción")
        main_frame = ttk.Frame(tab, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        titulo = ttk.Label(main_frame, text="Predicción de Próximo Sorteo", font=('', 14, 'bold'))
        titulo.pack(pady=(0, 15))
        pred_frame = ttk.LabelFrame(main_frame, text="Predicción Principal", padding=15)
        pred_frame.pack(fill=tk.X, pady=(0, 15))
        # Lottery selector so user knows/chooses which lottery to predict
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
        # update current_lottery when changed
        def _on_pred_loteria_change(evt=None):
            sel = self.combo_pred_loteria.get()
            if sel:
                self.current_lottery = sel
        self.combo_pred_loteria.bind('<<ComboboxSelected>>', _on_pred_loteria_change)

        row1 = ttk.Frame(pred_frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Método:", width=12).pack(side=tk.LEFT)
        self.combo_metodo = ttk.Combobox(row1, values=["Markov+Hora", "ML Ensemble", "Histórico"], state="readonly")
        self.combo_metodo.pack(side=tk.LEFT, padx=5)
        self.combo_metodo.current(0)
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
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        top_frame = ttk.Frame(results_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        tree_frame = ttk.Frame(top_frame)
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
        # enable sorting by clicking headers
        for col in columns:
            self.tree_resultados.heading(col, text=col, command=lambda c=col: self._tree_sort_by(self.tree_resultados, c, False))

        detail_frame = ttk.LabelFrame(results_frame, text="Detalle por hora", padding=5)
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        # allow the text area to resize freely
        self.text_resultados = scrolledtext.ScrolledText(detail_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.text_resultados.pack(fill=tk.BOTH, expand=True)

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
        ttk.Button(row, text="Actualizar Dashboard", command=self._actualizar_dashboard).pack(side=tk.LEFT, padx=10)
        graficos_frame = ttk.LabelFrame(main_frame, text="Gráficos", padding=15)
        graficos_frame.pack(fill=tk.BOTH, expand=True)
        self._crear_graficos(graficos_frame)

    def _crear_graficos(self, parent):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle("Análisis de Frecuencia", fontsize=12)

        for ax in axes.flat:
            ax.text(0.5, 0.5, "Cargue datos para ver gráficos", ha='center', va='center', fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # keep references for updates
        self.dash_canvas = canvas
        self.dash_fig = fig
        self.dash_axes = axes

    def _actualizar_dashboard(self):
        try:
            loteria_name = self.combo_dash_loteria.get()
            excel = self._get_excel_path(loteria_name)
            if not excel or not os.path.exists(excel):
                messagebox.showwarning("Dashboard", f"Archivo de datos no encontrado: {excel}")
                return
            df = pd.read_excel(excel)
            if df.empty:
                messagebox.showinfo("Dashboard", "No hay datos en el archivo seleccionado.")
                return

            # prepare date/time fields
            try:
                df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
            except Exception:
                pass

            # Axis references
            axes = getattr(self, 'dash_axes', None)
            if axes is None:
                messagebox.showerror("Dashboard", "Canvas del dashboard no inicializado.")
                return

            ax0 = axes[0][0]
            ax1 = axes[0][1]
            ax2 = axes[1][0]
            ax3 = axes[1][1]

            # Clear axes
            for ax in [ax0, ax1, ax2, ax3]:
                ax.clear()

            # Top numbers frequency
            if 'Num_Int' in df.columns:
                top_nums = df['Num_Int'].value_counts().head(10)
                ax0.bar(top_nums.index.astype(str), top_nums.values, color='#4CAF50')
                ax0.set_title('Top 10 Números')
                ax0.set_ylabel('Conteo')
            else:
                ax0.text(0.5, 0.5, 'No hay columna Num_Int', ha='center')

            # Counts per Hora (order by known lottery hours if possible)
            if 'Hora' in df.columns:
                horas_order = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00',
                               '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00',
                               '18:00:00', '19:00:00']
                cnt = df['Hora'].astype(str).value_counts()
                # try to reindex to known order when keys match
                present = [h for h in horas_order if h in cnt.index]
                if present:
                    cnt = cnt.reindex(present)
                else:
                    cnt = cnt.sort_index()
                ax1.plot(cnt.index.astype(str), cnt.values, marker='o')
                ax1.set_title('Conteo por Hora')
                ax1.set_xlabel('Hora')
                ax1.set_ylabel('Conteo')
                for label in ax1.get_xticklabels():
                    label.set_rotation(45)
            else:
                ax1.text(0.5, 0.5, 'No hay columna Hora', ha='center')

            # Top animals
            if 'Animal' in df.columns:
                top_anim = df['Animal'].value_counts().head(10)
                ax2.barh(top_anim.index.astype(str)[::-1], top_anim.values[::-1], color='#2196F3')
                ax2.set_title('Top Animales')
            else:
                ax2.text(0.5, 0.5, 'No hay columna Animal', ha='center')

            # Registraciones por fecha
            if 'Fecha' in df.columns:
                daily = df.groupby('Fecha').size().sort_index()
                ax3.plot(daily.index.astype(str), daily.values, marker='o', color='#f39c12')
                ax3.set_title('Registros por Día')
                ax3.set_xlabel('Fecha')
                for label in ax3.get_xticklabels():
                    label.set_rotation(45)
            else:
                ax3.text(0.5, 0.5, 'No hay columna Fecha', ha='center')

            self.dash_fig.tight_layout()
            self.dash_canvas.draw()
            messagebox.showinfo('Dashboard', 'Dashboard actualizado.')
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror('Dashboard', f'Error actualizando dashboard: {e}')

    def _crear_pestana_modelos(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Modelos ML")
        main_frame = ttk.Frame(tab, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        titulo = ttk.Label(main_frame, text="Gestión de Modelos de Machine Learning", font=('', 14, 'bold'))
        titulo.pack(pady=(0, 15))
        controls_frame = ttk.LabelFrame(main_frame, text="Entrenamiento y Predicción", padding=15)
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        row = ttk.Frame(controls_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Loteria:", width=10).pack(side=tk.LEFT)
        loterias = list(LOTTERY_MODULES.keys())
        self.combo_ml_loteria = ttk.Combobox(row, values=loterias, state="readonly", width=25)
        self.combo_ml_loteria.pack(side=tk.LEFT, padx=5)
        self.combo_ml_loteria.current(loterias.index(self.current_lottery))
        ttk.Button(row, text="Entrenar Modelo", command=self._entrenar_modelo_ml).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Pred. RF+XGB", command=self._dialogo_prediccion_ml).pack(side=tk.LEFT, padx=5)
        results_frame = ttk.LabelFrame(main_frame, text="Resultados del Modelo", padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.text_ml = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.text_ml.pack(fill=tk.BOTH, expand=True)

    def _entrenar_modelo_ml(self):
        self.text_ml.delete(1.0, tk.END)
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.text_ml)
        try:
            analizador = self._get_analizador()
            if analizador is None:
                return
            loteria = self.combo_ml_loteria.get()
            print(f"Entrenando modelos para: {loteria}")
            print("Cargando datos...")
            excel_file = self._get_excel_path(loteria)
            datos = pd.read_excel(excel_file)
            print(f"Datos cargados: {len(datos)} registros")
            print("Iniciando entrenamiento (esto puede tomar varios minutos)...")
            analizador.entrenar_modelo_con_optimizacion(datos)
            print("Modelo entrenado y guardado exitosamente.")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    def _tree_sort_by(self, tree, col, descending):
        # grab values to sort
        data = [(tree.set(child, col), child) for child in tree.get_children('')]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=descending)
        except Exception:
            data.sort(key=lambda t: t[0], reverse=descending)
        for i, item in enumerate(data):
            tree.move(item[1], '', i)
        # reverse sort next time
        tree.heading(col, command=lambda: self._tree_sort_by(tree, col, not descending))

    def _dialogo_prediccion_ml(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Predicción RF + XGBoost")
        dialog.geometry("700x600")
        dialog.resizable(True, True)

        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Predicción Combinada Random Forest + XGBoost",
                  font=('', 12, 'bold')).pack(pady=(0, 15))

        input_frame = ttk.LabelFrame(main_frame, text="Parámetros de Búsqueda", padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 15))

        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Animal/Número:", width=15).pack(side=tk.LEFT)
        entry_animal = ttk.Entry(row, width=25)
        entry_animal.pack(side=tk.LEFT, padx=5)
        entry_animal.focus()

        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="Hora (opcional):", width=15).pack(side=tk.LEFT)
        combo_hora = ttk.Combobox(row2, values=["", "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                                                 "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                                                 "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM"],
                                   state="readonly", width=22)
        combo_hora.pack(side=tk.LEFT, padx=5)
        combo_hora.current(0)

        row3 = ttk.Frame(input_frame)
        row3.pack(fill=tk.X, pady=10)
        resultados_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=('Consolas', 10))
        resultados_text.pack(fill=tk.BOTH, expand=True)

        def buscar():
            resultados_text.delete(1.0, tk.END)
            old_stdout = sys.stdout
            sys.stdout = RedirectText(resultados_text)
            try:
                analizador = self._get_analizador()
                if analizador is None:
                    return
                entrada = entry_animal.get().strip()
                hora = combo_hora.get().strip()
                if not entrada:
                    resultados_text.insert(tk.END, "Ingrese un animal o número.\n")
                    return
                print(f"Buscando predicción para: {entrada}")
                if hora:
                    print(f"Hora: {hora}")
                print("-" * 50)
                try:
                    from loteria_base import ejecutar_prediccion_rf_xgb
                    ejecutar_prediccion_rf_xgb(analizador, entrada, hora)
                except ImportError:
                    print("Función ejecutar_prediccion_rf_xgb no encontrada.")
                    resultados_text.insert(tk.END, "\nBúsqueda completada.\n")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout

        btn_buscar = ttk.Button(row3, text="Buscar", command=buscar, style='ML.TButton')
        btn_buscar.pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="Cerrar", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _limpiar_campos(self):
        self.entry_animal.delete(0, tk.END)
        self.tree_resultados.delete(*self.tree_resultados.get_children())
        self.text_resultados.delete(1.0, tk.END)

    def _predecir_principal(self):
        self.tree_resultados.delete(*self.tree_resultados.get_children())
        self.text_resultados.delete(1.0, tk.END)
        metodo = self.combo_metodo.get()
        entrada = self.entry_animal.get().strip()
        hora = self.combo_hora.get()
        # Determine selected lottery (give priority to prediction combobox)
        try:
            sel_loteria = self.combo_pred_loteria.get()
            if sel_loteria:
                self.current_lottery = sel_loteria
        except Exception:
            pass
        # show which lottery we are using at the top of the output
        analizador = self._get_analizador()
        if analizador is None:
            return
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print(f"=== Predicción por {metodo} ===")
            print(f"Loteria: {self.current_lottery}")
            if entrada:
                print(f"Consulta: {entrada}")
            if hora:
                print(f"Hora: {hora}")
            print("-" * 50)
            if metodo == "Markov+Hora":
                excel = self._get_excel_path(self.current_lottery)
                datos = pd.read_excel(excel) if os.path.exists(excel) else None
                from utils import load_and_prepare_data
                if datos is not None:
                    datos = load_and_prepare_data(excel, analizador)
                if datos is not None:
                    # Use the new structured API to get prediction data
                    data = analizador.generar_prediccion_markov(datos)
                    # store last prediction for dialog usage
                    self.last_prediccion_data = data
                    # populate top table
                    self.tree_resultados.delete(*self.tree_resultados.get_children())
                    for item in data.get('top', []):
                        self.tree_resultados.insert('', 'end', values=(item['rank'], item['num'], item['animal'], item['score'], item['markov'], item['hora_pct']))
                    # populate detalle por hora in text area
                    por_hora = data.get('por_hora', {})
                    detalle_lines = []
                    for hora, lista in por_hora.items():
                        detalle_lines.append(f"{hora}:")
                        for it in lista:
                            detalle_lines.append(f"  #{it['num']:2d} {it['animal']:<14} Score: {it['score']}")
                        detalle_lines.append("")
                    print('\n'.join(detalle_lines))
                else:
                    print("No hay datos para generar predicción.")
            elif metodo == "ML Ensemble":
                print("Usando modelo ML...")
                print("Funcionalidad avanzada próximamente.")
            else:
                print("Usando análisis histórico...")
                analizador.generar_matriz_probabilidad()
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            sys.stdout = old_stdout

        texto = buf.getvalue()
        lineas = texto.split("\n")
        en_tabla = False
        detalle = []
        for linea in lineas:
            if linea.startswith("  #") and "Animal" in linea:
                en_tabla = True
                continue
            if en_tabla:
                if linea.strip().startswith("---"):
                    continue
                if linea.strip().startswith("(Mostrando"):
                    en_tabla = False
                    continue
                if linea.strip() == "":
                    en_tabla = False
                    continue
                partes = linea.strip().split()
                if len(partes) >= 6:
                    try:
                        rank = partes[0]
                        num = partes[1]
                        animal = partes[2]
                        score = partes[3]
                        markov = partes[4].rstrip("%")
                        hora_pct = partes[6].rstrip("%") if len(partes) > 6 else ""
                        self.tree_resultados.insert("", "end", values=(rank, num, animal, score, markov, hora_pct))
                    except (IndexError, ValueError):
                        detalle.append(linea)
                else:
                    detalle.append(linea)
            else:
                detalle.append(linea)
        self.text_resultados.insert(1.0, "\n".join(detalle))

    def _ver_top_por_hora_dialog(self):
        data = getattr(self, 'last_prediccion_data', None)
        if not data:
            messagebox.showinfo('Info', 'Primero genere una predicción (Presione "Predecir").')
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
            hora = combo.get()
            tree.delete(*tree.get_children())
            lista = data.get('por_hora', {}).get(hora, [])
            for it in lista:
                tree.insert('', 'end', values=(it['num'], it['animal'], it['score']))

        combo.bind('<<ComboboxSelected>>', on_select)
        if horas:
            combo.current(0)
            on_select()

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
        ttk.Button(row, text="Scrapear Todo", command=self._scrapear).pack(side=tk.LEFT, padx=10)
        ttk.Button(row, text="Ver Estado", command=self._ver_estado_actual).pack(side=tk.LEFT, padx=5)
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=15)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.text_scraper = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.text_scraper.pack(fill=tk.BOTH, expand=True)

    def _scrapear(self):
        self.text_scraper.delete(1.0, tk.END)
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.text_scraper)
        try:
            import scrape_all
            scrape_all.import_scrapers()
            loteria_name = self.combo_scraper_loteria.get()
            lot = None
            for l in scrape_all.LOTTERIES:
                if l.get('name') == loteria_name:
                    lot = l
                    break
            if lot is None:
                print(f"No se encontro la loteria: {loteria_name}")
                return
            # pedir rango de fechas
            hoy = datetime.date.today()
            default_start = (hoy - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            start = tk.simpledialog.askstring('Fecha inicio', 'Fecha inicio (YYYY-MM-DD):', initialvalue=default_start, parent=self.root)
            if not start:
                print('Cancelado por el usuario.')
                return
            end = tk.simpledialog.askstring('Fecha fin', 'Fecha fin (YYYY-MM-DD):', initialvalue=hoy.strftime('%Y-%m-%d'), parent=self.root)
            if not end:
                print('Cancelado por el usuario.')
                return
            print(f"Iniciando scraper para: {loteria_name} desde {start} hasta {end}")
            scrape_all.scrape_lottery(lot, start, end)
            print("Scraping completado.")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    def _ver_estado_actual(self):
        self.text_scraper.delete(1.0, tk.END)
        loteria_name = self.combo_scraper_loteria.get()
        try:
            import scrape_all
            lot = None
            for l in scrape_all.LOTTERIES:
                if l.get('name') == loteria_name:
                    lot = l
                    break
            if lot is None:
                self.text_scraper.insert(tk.END, f"No se encontro la loteria: {loteria_name}\n")
                return
            excel_file = lot.get('file')
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                self.text_scraper.insert(tk.END, f"No se pudo leer {excel_file}: {e}\n")
                return
            if df.empty:
                self.text_scraper.insert(tk.END, f"Archivo {excel_file} esta vacio.\n")
                return
            last = df.tail(10)
            self.text_scraper.insert(tk.END, f"Ultimos registros de {excel_file}:\n")
            self.text_scraper.insert(tk.END, last.to_string(index=False))
            self.text_scraper.insert(tk.END, "\n\n")
            try:
                df['Fecha'] = pd.to_datetime(df['Fecha']).dt.date
                today = datetime.date.today()
                df_today = df[df['Fecha'] == today]
                if df_today.empty:
                    self.text_scraper.insert(tk.END, f"No hay sorteos registrados para hoy ({today}).\n")
                else:
                    # Show full list of today's registered draws (should be up to 12: 08:00-19:00)
                    try:
                        registros_hoy = df_today[['Hora', 'Num_Int', 'Animal', 'Numero']].sort_values('Hora')
                        self.text_scraper.insert(tk.END, f"SORTEOS REGISTRADOS HOY ({today}) ({len(registros_hoy)}):\n")
                        self.text_scraper.insert(tk.END, registros_hoy.to_string(index=False))
                        self.text_scraper.insert(tk.END, "\n")
                        # Also list unique animals and missing hours (if any)
                        animales = registros_hoy['Animal'].unique().tolist()
                        self.text_scraper.insert(tk.END, f"Animales registrados hoy ({today}): {animales}\n")
                        horas_posibles = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00',
                                         '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00',
                                         '18:00:00', '19:00:00']
                        # Normalize horas present to strings to compare reliably
                        horas_registradas = registros_hoy['Hora'].astype(str).tolist()
                        horas_faltantes = [h for h in horas_posibles if h not in horas_registradas]
                        if horas_faltantes:
                            self.text_scraper.insert(tk.END, f"Horas faltantes hoy: {horas_faltantes}\n")
                        else:
                            self.text_scraper.insert(tk.END, "TODAS LAS HORAS DE HOY ESTAN COMPLETAS!\n")
                    except Exception:
                        # Fallback: show a simple summary if expected columns are missing
                        animales = df_today['Animal'].unique().tolist() if 'Animal' in df_today.columns else []
                        self.text_scraper.insert(tk.END, f"Animales registrados hoy ({today}): {animales}\n")
            except Exception:
                pass
        except Exception as e:
            self.text_scraper.insert(tk.END, f"ERROR: {e}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernLottoUI(root)
    root.mainloop()
