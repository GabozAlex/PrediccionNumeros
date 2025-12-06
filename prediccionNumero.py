import sys
import pandas as pd
import datetime
import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import pickle
import json
import datetime
from datetime import datetime, date

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import scipy.stats as stats

# -----------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# -----------------------------------------------------------

def setup_logging():
    """Configura el sistema de logging profesional"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logger = logging.getLogger('lotto_predictor')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = RotatingFileHandler(
        'logs/lotto_predictor.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# -----------------------------------------------------------
# DICCIONARIO DE ANIMALES COMPLETO (38 ANIMALES)
# -----------------------------------------------------------

caracteristicas_animales = {
    "DELFIN": 0,
    "BALLENA": 37,
    "CARNERO": 1,
    "TORO": 2,
    "CIEMPIES": 3,
    "ALACRAN": 4,
    "LEON": 5,
    "RANA": 6,
    "PERICO": 7,
    "RATON": 8,
    "AGUILA": 9,
    "TIGRE": 10,
    "GATO": 11,
    "CABALLO": 12,
    "MONO": 13,
    "PALOMA": 14,
    "ZORRO": 15,
    "OSO": 16,
    "PAVO": 17,
    "BURRO": 18,
    "CHIVO": 19,
    "COCHINO": 20,
    "GALLO": 21,
    "CAMELLO": 22,
    "CEBRA": 23,
    "IGUANA": 24,
    "GALLINA": 25,
    "VACA": 26,
    "PERRO": 27,
    "ZAMURO": 28,
    "ELEFANTE": 29,
    "CAIMAN": 30,
    "LAPA": 31,
    "ARDILLA": 32,
    "PESCADO": 33,
    "VENADO": 34,
    "JIRAFA": 35,
    "CULEBRA": 36
}

def verificar_diccionario_animales():
    """Verifica que el diccionario tenga todos los 38 animales y números únicos"""
    print("\n🔍 VERIFICANDO DICCIONARIO DE ANIMALES...")
    
    total_animales = len(caracteristicas_animales)
    print(f"• Total de animales en diccionario: {total_animales}")
    
    if total_animales != 38:
        print(f"❌ ERROR: Se esperaban 38 animales, pero hay {total_animales}")
        return False
    
    numeros = list(caracteristicas_animales.values())
    numeros_unicos = set(numeros)
    
    if len(numeros) != len(numeros_unicos):
        print("❌ ERROR: Hay números duplicados en el diccionario")
        duplicados = [num for num in numeros if numeros.count(num) > 1]
        print(f"   Números duplicados: {set(duplicados)}")
        return False
    
    min_num = min(numeros)
    max_num = max(numeros)
    
    if min_num != 0 or max_num != 37:
        print(f"❌ ERROR: Rango numérico incorrecto. Debe ser 0-37, pero es {min_num}-{max_num}")
        return False
    
    print("✅ Diccionario correcto - 38 animales con números 0-37")
    print("\n📋 LISTA COMPLETA DE ANIMALES:")
    for i, (animal, numero) in enumerate(sorted(caracteristicas_animales.items(), key=lambda x: x[1]), 1):
        print(f"   {i:2d}. {animal:<12} -> {numero:2d}")
    
    return True

# -----------------------------------------------------------
# FUNCIONES DE VALIDACIÓN
# -----------------------------------------------------------

def validar_animal(animal):
    """Valida que un animal exista en el diccionario"""
    animal = animal.strip().upper()
    if animal not in caracteristicas_animales:
        animales_validos = ", ".join(sorted(caracteristicas_animales.keys()))
        raise ValueError(f"Animal '{animal}' no válido. Animales válidos: {animales_validos}")
    return animal

def validar_numero(numero):
    """Valida que un número esté en el rango correcto (0-37)"""
    if not (0 <= numero <= 37):
        raise ValueError(f"Número {numero} fuera de rango. Debe ser entre 0-37")
    return numero

# -----------------------------------------------------------
# FUNCIONALIDAD DEL MENÚ
# -----------------------------------------------------------

def mostrar_menu(titulo, opciones):
    """Muestra un menú numerado y solicita la selección del usuario."""
    print(f"\n--- **{titulo}** ---")
    for i, opcion in enumerate(opciones, 1):
        print(f"{i}. {opcion}")
    print("-------------------")
    
    while True:
        try:
            seleccion = input("Selecciona una opción (número): ")
            numero_seleccionado = int(seleccion)
            if 1 <= numero_seleccionado <= len(opciones):
                return numero_seleccionado
            else:
                print(f"**Error:** El número debe estar entre 1 y {len(opciones)}. Inténtalo de nuevo.")
        except ValueError:
            print("**Error:** Por favor, ingresa un número válido. Inténtalo de nuevo.")

# -----------------------------------------------------------
# FUNCIONES DE INGENIERÍA DE CARACTERÍSTICAS
# -----------------------------------------------------------

def calcular_diferencia_ciclica(actual, previo, max_val=38):
    """Calcula la distancia más corta en un círculo de 38 posiciones (0 a 37)."""
    if pd.isna(actual) or pd.isna(previo):
        return np.nan
    
    actual = int(actual)
    previo = int(previo)
    
    diferencia_base = abs(actual - previo)
    diferencia_opuesta = max_val - diferencia_base
    
    return min(diferencia_base, diferencia_opuesta)

def agregar_caracteristicas_atributos(datos):
    """
    Agrega características básicas de la ruleta
    """
    df = datos.copy()

    df['Posicion_Previo'] = df['Numero'].shift(1) 
    
    df['Diferencia_Ciclica'] = df.apply(
        lambda row: calcular_diferencia_ciclica(row['Numero'], row['Posicion_Previo']),
        axis=1
    )

    df['Animal_Previo'] = df['Animal'].shift(1)
    df['Hora_Sorteo'] = df['Hora']

    df.dropna(subset=['Animal_Previo', 'Diferencia_Ciclica'], inplace=True)
    
    return df

def agregar_caracteristicas_avanzadas(datos):
    """
    Añade características temporales y de secuencia avanzadas
    """
    df = datos.copy()
    
    if 'Timestamp' in df.columns:
        df['Dia_Semana'] = df['Timestamp'].dt.dayofweek
        df['Mes'] = df['Timestamp'].dt.month
        df['Hora_Num'] = df['Timestamp'].dt.hour
        df['Es_Fin_Semana'] = df['Dia_Semana'].isin([5, 6]).astype(int)
    
    df['Posicion_Previo'] = df['Numero'].shift(1)
    df['Animal_Previo'] = df['Animal'].shift(1)
    
    df['Diferencia_Ciclica'] = df.apply(
        lambda row: calcular_diferencia_ciclica(row['Numero'], row['Posicion_Previo']),
        axis=1
    )
    
    df['Frecuencia_Animal_10'] = df.groupby('Animal')['Animal'].transform(
        lambda x: x.rolling(10, min_periods=1).count()
    )
    
    df['Repite_Animal'] = (df['Animal'] == df['Animal_Previo']).astype(int)
    df['Mismo_Animal_3_Sorteos'] = (df['Animal'] == df['Animal'].shift(1)) & (df['Animal'] == df['Animal'].shift(2))
    df['Mismo_Animal_3_Sorteos'] = df['Mismo_Animal_3_Sorteos'].astype(int)
    
    df['Media_Movil_5'] = df['Numero'].rolling(5, min_periods=1).mean()
    df['Std_Movil_5'] = df['Numero'].rolling(5, min_periods=1).std()
    
    def grupo_ruleta(numero):
        if numero <= 9: return 0
        elif numero <= 19: return 1
        elif numero <= 29: return 2
        else: return 3
    
    df['Grupo_Ruleta'] = df['Numero'].apply(grupo_ruleta)
    df['Grupo_Ruleta_Previo'] = df['Grupo_Ruleta'].shift(1)
    
    print(f"✅ Características avanzadas añadidas: {len(df.columns)} features totales")
    return df

# -----------------------------------------------------------
# FUNCIONES DE MACHINE LEARNING
# -----------------------------------------------------------

def preparar_datos_ml(datos):
    """Prepara datos usando SOLO características básicas"""
    df_ml = datos.copy()
    
    animales_validos = list(caracteristicas_animales.keys())
    df_ml = df_ml[df_ml['Animal'].isin(animales_validos)].copy()

    # SOLO 3 CARACTERÍSTICAS BÁSICAS
    df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip()
    numeric_features = ['Posicion_Previo', 'Diferencia_Ciclica']
    categorical_features = ['Hora_Sorteo'] 
    
    le_y = LabelEncoder()
    le_y.fit(animales_validos) 
    df_ml['Animal_Encoded'] = le_y.transform(df_ml['Animal'])
    Y = df_ml['Animal_Encoded']
    
    all_categories = [sorted(df_ml[col].astype(str).unique().tolist()) for col in categorical_features]
    
    X = df_ml[numeric_features + categorical_features].copy()
    X.dropna(inplace=True)
    Y = Y.loc[X.index] 

    print(f"✅ Datos ML básicos: {len(X)} muestras, 3 características")
    
    return X, Y, le_y, numeric_features, categorical_features, all_categories


def preparar_datos_ml_completo(datos):
    """
    Preparación de datos SOLO con características básicas siempre disponibles
    """
    df_ml = datos.copy()
    
    # 1. Validar animales
    animales_validos = list(caracteristicas_animales.keys())
    df_ml = df_ml[df_ml['Animal'].isin(animales_validos)].copy()
    
    if len(df_ml) < 50:
        print(f"⚠️  Advertencia: Solo {len(df_ml)} registros válidos. Se recomiendan al menos 50.")
    
    # 2. DEFINIR SOLO 3 CARACTERÍSTICAS BÁSICAS (siempre disponibles)
    numeric_features = ['Posicion_Previo', 'Diferencia_Ciclica']
    categorical_features = ['Hora_Sorteo']
    
    # 3. Asegurar que Hora_Sorteo existe
    if 'Hora_Sorteo' not in df_ml.columns:
        df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip()
    
    # 4. Codificar variable objetivo - CORREGIDO PARA XGBOOST
    le_y = LabelEncoder()
    # FORZAR a usar TODOS los animales del diccionario, no solo los que aparecen en datos
    le_y.fit(list(caracteristicas_animales.keys()))  # ← ESTA ES LA CLAVE

    # Transformar los animales que sí están en los datos
    animales_en_datos = df_ml['Animal'].unique()
    mascara_animales_validos = df_ml['Animal'].isin(le_y.classes_)
    df_ml = df_ml[mascara_animales_validos].copy()

    df_ml['Animal_Encoded'] = le_y.transform(df_ml['Animal'])
    Y = df_ml['Animal_Encoded']

    # VERIFICAR que tenemos todas las clases
    print(f"✅ Clases configuradas: {len(le_y.classes_)} animales")
    print(f"✅ Animales en datos: {len(animales_en_datos)} animales únicos")
    
    # 5. Preparar features (SOLO las 3 básicas)
    available_features = []
    for feature in numeric_features + categorical_features:
        if feature in df_ml.columns:
            available_features.append(feature)
    
    X = df_ml[available_features].copy()
    
    # 6. Limpieza final
    filas_antes = len(X)
    X = X.dropna()
    filas_despues = len(X)
    
    if filas_antes != filas_despues:
        print(f"⚠️  Eliminadas {filas_antes - filas_despues} filas con valores NaN")
    
    Y = Y.loc[X.index]
    
    print(f"✅ Datos ML preparados: {len(X)} muestras, {len(available_features)} características BÁSICAS")
    print(f"   • Características: {available_features}")
    
    return X, Y, le_y, numeric_features, categorical_features, available_features


def crear_pipeline_ml(modelo, numeric_features, categorical_features):
    """Crea el pipeline que maneja la codificación y el modelo en un solo objeto."""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop'
    )
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', modelo)])
    
    return clf

def calcular_precision_top_k(y_real, y_proba, k=3):
    """
    Calcula la precisión Top-K: ¿está la clase real en las K predicciones más probables?
    """
    top_k_predicciones = np.argsort(y_proba, axis=1)[:, -k:]
    correctos = 0
    for i, real in enumerate(y_real):
        if real in top_k_predicciones[i]:
            correctos += 1
    return correctos / len(y_real)

def entrenar_modelo_ml(X, Y, modelo, modelo_nombre, numeric_features, categorical_features):
    """Entrena y evalúa un modelo ML usando validación temporal con Pipeline."""
    print(f"\n--- 🌳 Entrenando {modelo_nombre} con {len(X)} sorteos ---")
    
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = crear_pipeline_ml(modelo, numeric_features, categorical_features)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        pipeline.fit(X_train, Y_train)
        
        accuracy = pipeline.score(X_test, Y_test)
        print(f"Precisión de la validación temporal: {accuracy:.2%}")
        
    return pipeline

def entrenar_modelo_ml_mejorado(X, Y, modelo, modelo_nombre, numeric_features, categorical_features):
    """Entrena y evalúa un modelo ML con métricas mejoradas"""
    print(f"\n--- 🚀 Entrenando {modelo_nombre} con {len(X)} sorteos ---")
    
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = crear_pipeline_ml(modelo, numeric_features, categorical_features)
    
    accuracies = []
    top3_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        pipeline.fit(X_train, Y_train)
        
        accuracy = pipeline.score(X_test, Y_test)
        accuracies.append(accuracy)
        
        y_proba = pipeline.predict_proba(X_test)
        top3_acc = calcular_precision_top_k(Y_test.values, y_proba, k=3)
        top3_accuracies.append(top3_acc)
        
        print(f"   Fold {fold+1}: Accuracy = {accuracy:.2%}, Top-3 = {top3_acc:.2%}")
    
    avg_accuracy = np.mean(accuracies)
    avg_top3 = np.mean(top3_accuracies)
    
    print(f"\n📊 RESULTADOS FINALES {modelo_nombre}:")
    print(f"   • Accuracy Promedio: {avg_accuracy:.2%}")
    print(f"   • Top-3 Accuracy: {avg_top3:.2%}")
    print(f"   • Mejor Fold: {max(accuracies):.2%}")
    print(f"   • Peor Fold: {min(accuracies):.2%}")
    
    return pipeline

def predecir_top_k_por_hora(pipeline, le_y, df_ml, k=10):
    """
    Genera la matriz de predicción Top-K usando SOLO características básicas
    """
    matriz_prediccion_ia = {}
    
    # Asegurar que Hora_Sorteo existe
    if 'Hora_Sorteo' not in df_ml.columns:
        df_ml['Hora_Sorteo'] = df_ml['Hora'].astype(str).str.strip()
    
    horas_sorteo = sorted(df_ml['Hora_Sorteo'].unique())
    
    print("\nGenerando Matriz de Predicción TOP-10 de la IA...")
    
    for hora in horas_sorteo:
        df_hora = df_ml[df_ml['Hora_Sorteo'] == hora].iloc[[-1]].copy()
        
        if df_hora.isnull().any().any() or df_hora.empty:
            continue
        
        # USAR SOLO LAS 3 CARACTERÍSTICAS BÁSICAS
        X_query = df_hora[['Posicion_Previo', 'Diferencia_Ciclica', 'Hora_Sorteo']]

        try:
            y_proba = pipeline.predict_proba(X_query)[0]
            
            indices_top_k = np.argsort(y_proba)[::-1][:k] 
            animales_predichos = le_y.inverse_transform(indices_top_k)
            
            matriz_prediccion_ia[hora] = animales_predichos.tolist()
        except Exception as e:
            print(f"⚠️  Error en hora {hora}: {e}")
            continue
        
    print(f"✅ Matriz generada con {len(matriz_prediccion_ia)} horas")
    return matriz_prediccion_ia

# -----------------------------------------------------------
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# -----------------------------------------------------------

def optimizar_hiperparametros_rf(X, Y, numeric_features, categorical_features):
    """Optimiza hiperparámetros para Random Forest"""
    logger.info("Iniciando optimización de Random Forest...")
    
    param_dist = {
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__max_depth': [5, 10, 15, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    
    modelo_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipeline_base = crear_pipeline_ml(modelo_base, numeric_features, categorical_features)
    
    tscv = TimeSeriesSplit(n_splits=3)
    random_search = RandomizedSearchCV(
        pipeline_base,
        param_distributions=param_dist,
        n_iter=20,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, Y)
    
    logger.info(f"Mejores parámetros RF: {random_search.best_params_}")
    logger.info(f"Mejor score RF: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def optimizar_hiperparametros_xgb(X, Y, numeric_features, categorical_features):
    """Optimiza hiperparámetros para XGBoost"""
    logger.info("Iniciando optimización de XGBoost...")
    
    param_dist = {
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2, 0.3]
    }
    
    modelo_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    
    pipeline_base = crear_pipeline_ml(modelo_base, numeric_features, categorical_features)
    
    tscv = TimeSeriesSplit(n_splits=3)
    random_search = RandomizedSearchCV(
        pipeline_base,
        param_distributions=param_dist,
        n_iter=15,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, Y)
    
    logger.info(f"Mejores parámetros XGB: {random_search.best_params_}")
    logger.info(f"Mejor score XGB: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def entrenar_modelo_con_optimizacion(X, Y, tipo_modelo, numeric_features, categorical_features):
    """
    Entrena un modelo con optimización automática de hiperparámetros
    """
    logger.info(f"Iniciando entrenamiento con optimización para {tipo_modelo}")
    
    start_time = datetime.now()
    
    if tipo_modelo == 'rf':
        modelo_optimizado = optimizar_hiperparametros_rf(X, Y, numeric_features, categorical_features)
        modelo_nombre = "Random Forest Optimizado"
    elif tipo_modelo == 'xgb':
        modelo_optimizado = optimizar_hiperparametros_xgb(X, Y, numeric_features, categorical_features)
        modelo_nombre = "XGBoost Optimizado"
    else:
        raise ValueError("Tipo de modelo no soportado")
    
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    top3_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        if fold == 0:
            modelo_optimizado.fit(X_train, Y_train)
        
        accuracy = modelo_optimizado.score(X_test, Y_test)
        accuracies.append(accuracy)
        
        y_proba = modelo_optimizado.predict_proba(X_test)
        top3_acc = calcular_precision_top_k(Y_test.values, y_proba, k=3)
        top3_accuracies.append(top3_acc)
        
        logger.info(f"Fold {fold+1}: Accuracy = {accuracy:.2%}, Top-3 = {top3_acc:.2%}")
    
    tiempo_entrenamiento = datetime.now() - start_time
    avg_accuracy = np.mean(accuracies)
    avg_top3 = np.mean(top3_accuracies)
    
    print(f"\n🎯 RESULTADOS {modelo_nombre}:")
    print(f"   • Accuracy Promedio: {avg_accuracy:.2%}")
    print(f"   • Top-3 Accuracy: {avg_top3:.2%}")
    print(f"   • Tiempo entrenamiento: {tiempo_entrenamiento}")
    print(f"   • Mejor Fold: {max(accuracies):.2%}")
    
    logger.info(f"Entrenamiento completado: {avg_accuracy:.2%} accuracy, {avg_top3:.2%} top-3")
    
    return modelo_optimizado

# -----------------------------------------------------------
# GUARDADO Y CARGA DE MODELOS
# -----------------------------------------------------------

def guardar_modelo(modelo, le_y, metricas, nombre_modelo):
    """Guarda el modelo entrenado y sus métricas"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modelo_dir = f"modelos/{nombre_modelo}_{timestamp}"
    
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
    if not os.path.exists(modelo_dir):
        os.makedirs(modelo_dir)
    
    modelo_path = f"{modelo_dir}/modelo.pkl"
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo, f)
    
    le_path = f"{modelo_dir}/label_encoder.pkl"
    with open(le_path, 'wb') as f:
        pickle.dump(le_y, f)
    
    metricas_path = f"{modelo_dir}/metricas.json"
    with open(metricas_path, 'w') as f:
        json.dump(metricas, f, indent=2)
    
    info = {
        'nombre_modelo': nombre_modelo,
        'fecha_entrenamiento': timestamp,
        'caracteristicas_usadas': list(metricas.get('caracteristicas', [])),
        'num_muestras': metricas.get('num_muestras', 0)
    }
    
    info_path = f"{modelo_dir}/info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Modelo guardado en: {modelo_dir}")
    return modelo_dir

def cargar_modelo(modelo_dir):
    """Carga un modelo previamente guardado"""
    try:
        with open(f"{modelo_dir}/modelo.pkl", 'rb') as f:
            modelo = pickle.load(f)
        
        with open(f"{modelo_dir}/label_encoder.pkl", 'rb') as f:
            le_y = pickle.load(f)
        
        with open(f"{modelo_dir}/metricas.json", 'r') as f:
            metricas = json.load(f)
        
        logger.info(f"Modelo cargado desde: {modelo_dir}")
        return modelo, le_y, metricas
    
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return None, None, None

def cargar_ultimo_modelo(tipo_modelo):
    """Carga el último modelo entrenado de un tipo específico"""
    if not os.path.exists('modelos'):
        print("❌ No hay modelos guardados")
        return None, None, None
    
    modelos_dir = []
    for dir_name in os.listdir('modelos'):
        if dir_name.startswith(tipo_modelo):
            modelos_dir.append(dir_name)
    
    if not modelos_dir:
        print(f"❌ No se encontraron modelos de tipo: {tipo_modelo}")
        return None, None, None
    
    modelos_dir.sort(reverse=True)
    ultimo_modelo_dir = os.path.join('modelos', modelos_dir[0])
    
    modelo, le_y, metricas = cargar_modelo(ultimo_modelo_dir)
    
    if modelo:
        print(f"✅ Modelo cargado: {ultimo_modelo_dir}")
        print(f"   • Fecha entrenamiento: {metricas.get('fecha_entrenamiento', 'N/A')}")
        print(f"   • Muestras: {metricas.get('num_muestras', 'N/A')}")
        print(f"   • Accuracy: {metricas.get('accuracy_promedio', 'N/A'):.2%}")
    
    return modelo, le_y, metricas

# -----------------------------------------------------------
# FUNCIONES DE PREDICCIÓN PRINCIPALES
# -----------------------------------------------------------

def random_forest_prediction(datos):
    """Opción 7: Entrena y genera la matriz de predicción con Random Forest."""
    X, Y, le_y, numeric_features, categorical_features, all_categories = preparar_datos_ml(datos)
    
    if X.empty:
        print("**Error:** No hay suficientes datos limpios para entrenar el modelo.")
        return None
    
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    
    pipeline_entrenado = entrenar_modelo_ml(
        X, Y, modelo_rf, "Random Forest", numeric_features, categorical_features
    )
    
    matriz_prediccion = predecir_top_k_por_hora(pipeline_entrenado, le_y, datos.copy(), k=10)
    
    return matriz_prediccion

def xgboost_prediction(datos):
    """Opción 8: Entrena y genera la matriz de predicción con XGBoost."""
    X, Y, le_y, numeric_features, categorical_features, all_categories = preparar_datos_ml(datos)
    
    if X.empty:
        print("**Error:** No hay suficientes datos limpios para entrenar el modelo.")
        return None
    
    modelo_xgb = XGBClassifier(
        n_estimators=100, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42, 
        n_jobs=-1, 
        max_depth=5
    )
    
    pipeline_entrenado = entrenar_modelo_ml(
        X, Y, modelo_xgb, "XGBoost", numeric_features, categorical_features
    )
    
    matriz_prediccion = predecir_top_k_por_hora(pipeline_entrenado, le_y, datos.copy(), k=10)
    
    return matriz_prediccion

def random_forest_optimizado(datos):
    """Random Forest con optimización automática"""
    try:
        logger.info("Ejecutando Random Forest Optimizado")
        
        datos_con_features = agregar_caracteristicas_avanzadas(datos.copy())
        X, Y, le_y, numeric_features, categorical_features, available_features = preparar_datos_ml_completo(datos_con_features)
        
        if len(X) < 50:
            logger.warning(f"Datos insuficientes: {len(X)} muestras (mínimo 50 recomendado)")
            print("❌ Se recomiendan al menos 50 muestras para optimización")
            return None
        
        modelo_optimizado = entrenar_modelo_con_optimizacion(
            X, Y, 'rf', numeric_features, categorical_features
        )
        
        matriz_prediccion = predecir_top_k_por_hora(
            modelo_optimizado, le_y, datos_con_features.copy(), k=10
        )
        
        metricas = {
            'accuracy_promedio': np.mean([modelo_optimizado.score(X, Y)]),
            'num_muestras': len(X),
            'caracteristicas': available_features,
            'fecha_entrenamiento': datetime.now().isoformat()
        }
        
        modelo_dir = guardar_modelo(modelo_optimizado, le_y, metricas, "random_forest")
        
        print(f"✅ Modelo optimizado guardado en: {modelo_dir}")
        return matriz_prediccion
        
    except Exception as e:
        logger.error(f"Error en Random Forest optimizado: {e}")
        print(f"❌ Error: {e}")
        return None

def xgboost_optimizado(datos):
    """XGBoost con optimización automática"""
    try:
        logger.info("Ejecutando XGBoost Optimizado")
        
        datos_con_features = agregar_caracteristicas_avanzadas(datos.copy())
        X, Y, le_y, numeric_features, categorical_features, available_features = preparar_datos_ml_completo(datos_con_features)
        
        if len(X) < 50:
            logger.warning(f"Datos insuficientes: {len(X)} muestras")
            print("❌ Se recomiendan al menos 50 muestras para optimización")
            return None
        
        modelo_optimizado = entrenar_modelo_con_optimizacion(
            X, Y, 'xgb', numeric_features, categorical_features
        )
        
        matriz_prediccion = predecir_top_k_por_hora(
            modelo_optimizado, le_y, datos_con_features.copy(), k=10
        )
        
        metricas = {
            'accuracy_promedio': np.mean([modelo_optimizado.score(X, Y)]),
            'num_muestras': len(X),
            'caracteristicas': available_features,
            'fecha_entrenamiento': datetime.now().isoformat()
        }
        
        modelo_dir = guardar_modelo(modelo_optimizado, le_y, metricas, "xgboost")
        
        print(f"✅ Modelo optimizado guardado en: {modelo_dir}")
        return matriz_prediccion
        
    except Exception as e:
        logger.error(f"Error en XGBoost optimizado: {e}")
        print(f"❌ Error: {e}")
        return None

# -----------------------------------------------------------
# FUNCIONES DE EVALUACIÓN DE ESTRATEGIA
# -----------------------------------------------------------

def evaluacion_estrategia_frecuencia(datos):
    """
    Opción 9: Evalúa la estrategia de apuesta dinámica (Observar Mañana, Apostar Tarde)
    utilizando el Top-10 de Frecuencia Histórica (La línea base).
    """
    print("\n--- 🧠 Evaluación Estrategia DINÁMICA (BASE: Frecuencia Histórica) ---")
    
    frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
    
    top_10_map = {}
    for hora_24h in frecuencia_completa['Hora'].unique():
        top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
        top_10_map[hora_24h] = top_10_lista
    
    print("Lista Top-10 generada para todas las horas.")

    simular_estrategia(datos, top_10_map)

def evaluacion_estrategia_ia(datos, matriz_prediccion_ia):
    """
    Opción 10: Evalúa la estrategia dinámica utilizando la matriz de predicción del ML.
    """
    print("\n--- 🧠 Evaluación Estrategia DINÁMICA (OPTIMIZADA: Predicción de IA) ---")
    
    top_10_map = matriz_prediccion_ia
    print(f"Matriz de predicción cargada con {len(top_10_map)} horas.")

    simular_estrategia(datos, top_10_map)

def simular_estrategia(datos, top_10_map):
    """Función unificada que ejecuta la simulación de la estrategia dinámica - VERSIÓN CORREGIDA"""
    
    HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
    HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
    
    GASTO_TARDE = 1200.0
    GANANCIA_POR_ACIERTO = 600.0

    # Asegurar que tenemos la columna Fecha
    if 'Fecha' not in datos.columns:
        datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
    
    resultados_simulacion = []
    
    for fecha, df_dia in datos.groupby('Fecha'):
        aciertos_manana = 0
        
        df_manana = df_dia[df_dia['Hora'].isin(HORAS_MANANA)].copy()
        
        for _, row in df_manana.iterrows():
            hora_filtro = row['Hora']
            animal_salio = row['Animal']
            
            if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                aciertos_manana += 1
        
        jugar_tarde = (aciertos_manana <= 1)
        
        aciertos_tarde = 0
        ganancia_bruta_tarde = 0
        gasto_tarde = 0
        
        df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)].copy()

        if jugar_tarde:
            gasto_tarde = GASTO_TARDE
            
            for _, row in df_tarde.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    aciertos_tarde += 1
            
            ganancia_bruta_tarde = aciertos_tarde * GANANCIA_POR_ACIERTO

        # CREAR TODAS LAS COLUMNAS NECESARIAS
        resultados_simulacion.append({
            'Fecha': fecha,
            'Aciertos_Manana': aciertos_manana,
            'Jugar_Tarde': 'SÍ' if jugar_tarde else 'NO',  # Esta es la columna crítica
            'Aciertos_Tarde': aciertos_tarde,
            'Gasto': gasto_tarde,
            'Ganancia_Bruta': ganancia_bruta_tarde,
            'Ganancia_Neta': ganancia_bruta_tarde - gasto_tarde
        })
    
    df_resultados = pd.DataFrame(resultados_simulacion)
    
    # VERIFICAR que tenemos todas las columnas esperadas
    columnas_requeridas = ['Aciertos_Tarde', 'Jugar_Tarde', 'Ganancia_Neta']
    for col in columnas_requeridas:
        if col not in df_resultados.columns:
            print(f"❌ ERROR CRÍTICO: Columna '{col}' no se creó en los resultados")
            print("Columnas disponibles:", list(df_resultados.columns))
            return None
    
    # Continuar con el análisis normal...
    dias_completos = df_resultados[df_resultados['Aciertos_Tarde'].notna()]
    
    total_dias = len(dias_completos)
    total_dias_jugados = len(dias_completos[dias_completos['Jugar_Tarde'] == 'SÍ'])
    
    gasto_total = df_resultados['Gasto'].sum()
    ganancia_bruta_total = df_resultados['Ganancia_Bruta'].sum()
    ganancia_neta_total = df_resultados['Ganancia_Neta'].sum()
    
    print("\n" + "="*70)
    print("        🚀 RESUMEN DE LA EVALUACIÓN DE LA ESTRATEGIA DINÁMICA")
    print("="*70)
    print(f"✅ Días Completos Analizados: {total_dias}")
    print(f"✅ Días Jugados (con regla 0 o 1 acierto mañana): {total_dias_jugados}")
    print("-" * 70)
    print(f"💵 Gasto Total (solo en la Tarde): {gasto_total:,.2f} Bs")
    print(f"💰 Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
    print(f"📊 **GANANCIA/PÉRDIDA NETA TOTAL:** {ganancia_neta_total:,.2f} Bs")
    print("-" * 70)
    
    if gasto_total > 0:
        roi = (ganancia_neta_total / gasto_total) * 100
        print(f"📈 **Retorno de la Inversión (ROI):** {roi:,.2f}%")
        
    if ganancia_neta_total > 0:
        print("\n🎉 ¡Felicidades! La estrategia dinámica generó ganancias en la simulación.")
    elif ganancia_neta_total < 0:
        print("\n⚠️ Advertencia: La estrategia dinámica generó pérdidas en la simulación.")
    else:
        print("\n🟡 Resultado: Punto de Equilibrio (Ganancia Neta = 0).")
        
    print("\n--- Auditoría Diaria de Días Jugados ---")
    df_jugados = df_resultados[df_resultados['Jugar_Tarde'] == 'SÍ']

    if not df_jugados.empty:
        top_10_mejor = df_jugados.sort_values(by='Ganancia_Neta', ascending=False).head(10)
        top_10_peor = df_jugados.sort_values(by='Ganancia_Neta', ascending=True).head(10)

        print("\n✅ TOP 10 Días con Mayor Ganancia Neta:")
        print(top_10_mejor[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']].to_string(index=False))

        print("\n❌ TOP 10 Días con Mayor Pérdida Neta:")
        print(top_10_peor[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']].to_string(index=False))

    else:
        print("No hubo días suficientes en el historial para aplicar la estrategia.")
    
    return df_resultados

def mostrar_matriz_prediccion(matriz_prediccion):
    """Muestra la matriz de predicciones por hora"""
    print("\n🎯 MATRIZ DE PREDICCIÓN XGBoost - TOP 10 POR HORA")
    print("=" * 60)
    for hora, animales in sorted(matriz_prediccion.items()):
        print(f"🕐 {hora}:")
        for i, animal in enumerate(animales[:10], 1):  # Mostrar solo Top 10
            print(f"    {i:2d}. {animal}")
        print()

# -----------------------------------------------------------
# FUNCIONES DE PREDICCIÓN Y ANÁLISIS (FRECUENCIA Y MÁRKOV)
# -----------------------------------------------------------

def generar_matriz_probabilidad(datos):
    """Función auxiliar para generar la matriz de probabilidad."""
    datos['Animal_Siguiente'] = datos['Animal'].shift(-1)
    datos['Solo_Fecha'] = datos['Timestamp'].dt.date
    datos['Es_Ultimo_Sorteo_del_Dia'] = datos.groupby('Solo_Fecha')['Timestamp'].transform('max') == datos['Timestamp']
    
    df_transiciones = datos[datos['Es_Ultimo_Sorteo_del_Dia'] == False].copy()

    matriz_conteo = pd.crosstab(df_transiciones['Animal'], df_transiciones['Animal_Siguiente'], normalize=False)
    matriz_probabilidad = matriz_conteo.div(matriz_conteo.sum(axis=1), axis=0) * 100
    return matriz_probabilidad.fillna(0)

def matriz_probabilidad_transicion(datos):
    """Muestra la Matriz de Probabilidad de Transición de animales (Márkov)."""
    print("\n--- 📊 Matriz de Probabilidad de Transición ---")
    
    matriz = generar_matriz_probabilidad(datos.copy())
    
    print("La matriz muestra la probabilidad (%) de que la columna siga a la fila:")
    print(matriz.round(2))

def mejor_prediccion_siguiente(datos):
    """
    Opción 4: Muestra los 5 animales más probables para el siguiente sorteo, 
    basado en la Matriz de Márkov.
    """
    print("\n--- 🔮 Predicción Siguiente en Tiempo Real (TOP-5 Márkov) ---")
    
    animal_actual = input("Ingresa el **Animal** que acaba de salir (ej: PERRO): ").strip().upper()
    matriz_probabilidad = generar_matriz_probabilidad(datos.copy())
    
    if animal_actual not in matriz_probabilidad.index:
        print(f"**Error:** El animal '{animal_actual}' no se encontró en el historial de transiciones.")
        return

    fila_prediccion = matriz_probabilidad.loc[animal_actual].sort_values(ascending=False)
    top_5 = fila_prediccion.head(5) 
    
    print(f"\n--- Resultado de la Predicción (TOP 5) ---")
    print(f"Si acaba de salir **{animal_actual}**, los 5 más probables son:")
    
    resultados = []
    for animal, prob in top_5.items():
        resultados.append({'Animal': animal, 'Probabilidad (%)': f"{prob:.2f}"})
        
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

    mejor_animal = top_5.index[0]
    probabilidad_max = top_5.iloc[0]
    print(f"\n🥇 Máxima probabilidad individual: **{mejor_animal}** ({probabilidad_max:.2f}%)")

def probabilidad_maxima_por_hora(datos):
    """
    Opción 5: Muestra los 10 animales más probables de salir en cada franja horaria histórica.
    """
    print("\n--- 📈 Análisis de Frecuencia Histórica por Hora (TOP-10) ---")
    
    frecuencia_completa = datos.groupby('Solo_hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
    
    total_sorteos_por_hora = datos.groupby('Solo_hora').size().reset_index(name='Total_Sorteos')

    horas_unicas = frecuencia_completa['Solo_hora'].unique()
    
    print("\n--- Top 10 Animales por Cada Hora (Para Apuesta Diaria) ---")
    
    for hora in sorted(horas_unicas):
        df_hora = frecuencia_completa[frecuencia_completa['Solo_hora'] == hora].copy()
        
        total_sorteos = total_sorteos_por_hora[total_sorteos_por_hora['Solo_hora'] == hora]['Total_Sorteos'].iloc[0]

        top_10 = df_hora.sort_values(by='Probabilidad', ascending=False).head(10)
        
        print(f"\n⏰ **HORA: {hora}** (Total Sorteos: {total_sorteos})")
        print(top_10[['Animal', 'Probabilidad']].to_string(index=False, float_format="%.2f%%"))

def validar_modelo_markov(datos, porcentaje_entrenamiento=0.8, top_k=5):
    """
    Opción 2: Realiza una validación cruzada simple (Holdout) del modelo de Márkov, 
    calculando la Precisión Top-K.
    """
    print(f"\n--- 🔬 Validación Cruzada del Modelo de Márkov (TOP-{top_k}) ---")
    
    total_sorteos = len(datos)
    corte = int(total_sorteos * porcentaje_entrenamiento)
    
    df_entrenamiento = datos.iloc[:corte].copy()
    df_prueba = datos.iloc[corte:].copy()
    
    print(f"Total de sorteos: {total_sorteos}")
    print(f"Sorteos de Entrenamiento ({porcentaje_entrenamiento*100:.0f}%): {len(df_entrenamiento)}")
    print(f"Sorteos de Prueba ({100-porcentaje_entrenamiento*100:.0f}%): {len(df_prueba)}")
    
    if len(df_prueba) < 2:
        print("**Error:** No hay suficientes datos para la prueba.")
        return

    print("\nEntrenando la Matriz con los datos históricos...")
    matriz_entrenada = generar_matriz_probabilidad(df_entrenamiento)
    
    aciertos_top_k = 0
    predicciones_totales = 0
    
    for i in range(len(df_prueba) - 1):
        animal_actual = df_prueba.iloc[i]['Animal']
        animal_siguiente_real = df_prueba.iloc[i + 1]['Animal']
        
        if animal_actual in matriz_entrenada.index:
            predicciones_totales += 1
            
            fila_prediccion = matriz_entrenada.loc[animal_actual].sort_values(ascending=False)
            top_k_predichos = fila_prediccion.head(top_k).index.tolist()
            
            if animal_siguiente_real in top_k_predichos:
                aciertos_top_k += 1

    if predicciones_totales > 0:
        precision_top_k = (aciertos_top_k / predicciones_totales) * 100
        
        num_clases = len(matriz_entrenada.columns)
        probabilidad_azar = (top_k / num_clases) * 100
        
        print("\n--- ✅ Resultados de la Precisión (Validación) ---")
        print(f"Predicciones realizadas: {predicciones_totales}")
        print(f"Aciertos del Modelo (Top-{top_k}): {aciertos_top_k}")
        print(f"**Precisión TOP-{top_k} del Modelo: {precision_top_k:.2f}%**")
        print(f"\n*Nota: La precisión esperada al azar ({top_k}/{num_clases}) es de **{probabilidad_azar:.2f}%***")
        
        if precision_top_k > probabilidad_azar + 5:
            print("Resultado: ¡El modelo es **significativamente mejor** que el azar para el TOP-5!")
        elif precision_top_k > probabilidad_azar:
            print("Resultado: El modelo tiene un rendimiento **ligeramente superior** al azar para el TOP-5.")
        else:
            print("Resultado: El rendimiento está **cerca o por debajo del azar**. Considera usar más datos o un modelo diferente.")
    else:
        print("No se pudieron realizar predicciones válidas en el conjunto de prueba.")

def prediccion_por_hora_especifica(datos):
    """
    Opción 6: Solicita una hora específica y predice el animal con la mayor 
    probabilidad histórica de salir en esa franja horaria.
    """
    print("\n--- 🎯 Predicción Histórica por Hora Específica ---")
    
    while True:
        hora_str = input("Ingresa la **hora del sorteo** para predecir (ej: 11:00 AM o 14:00): ").strip()
        try:
            hora_dt = pd.to_datetime(hora_str, format='%H:%M', errors='coerce')
            if pd.isna(hora_dt):
                hora_dt = pd.to_datetime(hora_str, format='%I:%M %p', errors='coerce')
            
            if pd.isna(hora_dt):
                raise ValueError
            
            solo_hora_buscada = hora_dt.strftime('%I:%M %p') 
            break
        except ValueError:
            print("**Error:** Formato de hora inválido. Usa HH:MM (24h) o HH:MM AM/PM.")

    df_filtrado = datos[datos['Solo_hora'] == solo_hora_buscada].copy()
    
    if df_filtrado.empty:
        print(f"\n❌ No se encontraron datos históricos para la hora: **{solo_hora_buscada}**.")
        print("Asegúrate de que la hora esté escrita exactamente como aparece en tu historial (ej: 09:00 AM).")
        return

    frecuencia_animal = df_filtrado['Animal'].value_counts().reset_index()
    frecuencia_animal.columns = ['Animal', 'Conteo']
    
    total_sorteos_hora = len(df_filtrado)
    frecuencia_animal['Probabilidad'] = (frecuencia_animal['Conteo'] / total_sorteos_hora) * 100

    prediccion_maxima = frecuencia_animal.iloc[0]
    
    print(f"\n--- Resultados para la hora: **{solo_hora_buscada}** (Histórico) ---")
    print(f"Total de sorteos históricos analizados en esta hora: **{total_sorteos_hora}**")
    print("-" * 50)
    print(f"🥇 Animal con mayor probabilidad: **{prediccion_maxima['Animal']}**")
    print(f"   Probabilidad: **{prediccion_maxima['Probabilidad']:.2f}%**")
    print(f"   Veces que salió a esta hora: {prediccion_maxima['Conteo']}")
    print("-" * 50)
    
    print("\nTop 10 de animales en esta hora:")
    print(frecuencia_animal[['Animal', 'Probabilidad']].head(10).to_string(index=False))

# -----------------------------------------------------------
# FUNCIONALIDAD: INGRESAR DATOS
# -----------------------------------------------------------

def agregar_datos_al_excel(datos_df, nombre_archivo):
    """
    Opción 1: Permite al usuario ingresar los datos de un sorteo y guardarlos en el DataFrame 
    y en el archivo Excel.
    """
    print("\n--- ✍️ Ingreso de Nuevo Sorteo ---")
    
    while True:
        hora_str = input("Ingresa la hora del sorteo (ej: 09:00 AM o 14:00): ").strip()
        try:
            hora_dt = pd.to_datetime(hora_str, format='%H:%M', errors='coerce')
            if pd.isna(hora_dt):
                hora_dt = pd.to_datetime(hora_str, format='%I:%M %p', errors='coerce')
            
            if pd.isna(hora_dt):
                raise ValueError
            
            hora_final = hora_dt.strftime('%H:%M')
            solo_hora_final = hora_dt.strftime('%I:%M %p')
            break
        except ValueError:
            print("Formato de hora inválido. Usa HH:MM (24h) o HH:MM AM/PM.")

    while True:
        try:
            animal = input("Ingresa el nombre del animal (ej: PERRO): ").strip().upper()
            animal = validar_animal(animal)
            break
        except ValueError as e:
            print(f"Error: {e}")
    
    while True:
        numero_str = input("Ingresa el número (ej: 01): ").strip()
        try:
            if numero_str.isdigit():
                numero = int(numero_str)
                numero = validar_numero(numero)
                break
            else:
                print("Número inválido. Debe ser un número entre 00 y 37.")
        except ValueError as e:
            print(f"Error: {e}")

    fecha_hoy = date.today()  # Usa date directamente
    timestamp = pd.to_datetime(f"{fecha_hoy} {hora_final}")
    
    nueva_fila = pd.DataFrame([{
        'Fecha': fecha_hoy,
        'Hora': hora_final,
        'Animal': animal,
        'Numero': numero,
        'Timestamp': timestamp,
        'Solo_hora': solo_hora_final
    }])
    
    datos_df = pd.concat([datos_df, nueva_fila], ignore_index=True)
    
    try:
        datos_df[['Fecha', 'Hora', 'Animal', 'Numero']].to_excel(nombre_archivo, index=False)
        print("\n✅ ¡Sorteo agregado y archivo Excel actualizado exitosamente!")
        print(f"   Animal: {animal}, Número: {numero}, Hora: {solo_hora_final}")
        return datos_df
    except Exception as e:
        print(f"\n❌ Error al guardar en el archivo Excel: {e}")
        return datos_df

def evaluacion_estrategia_solo_manana(datos, hora_corte='13:00:00'):
    """
    Evalúa estrategia de jugar solo en la mañana (hasta hora específica)
    """
    print(f"\n--- 🌅 EVALUACIÓN ESTRATEGIA SOLO MAÑANA (Hasta {hora_corte}) ---")
    
    # 1. GENERAR MATRIZ DE FRECUENCIA PARA MAÑANA
    horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
    
    # Filtrar solo horas de mañana para el cálculo de frecuencia
    datos_manana = datos[datos['Hora'].isin(horas_manana)].copy()
    
    frecuencia_manana = datos_manana.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
    
    top_10_map_manana = {}
    for hora_24h in frecuencia_manana['Hora'].unique():
        top_10_lista = frecuencia_manana[frecuencia_manana['Hora'] == hora_24h].head(15)['Animal'].tolist()
        top_10_map_manana[hora_24h] = top_10_lista
    
    print(f"✅ Matriz de frecuencia generada para {len(top_10_map_manana)} horas de mañana")

    # 2. PARÁMETROS FINANCIEROS
    APUESTA_POR_HORA = 500.0  # 10 animales * 50 Bs cada uno
    GANANCIA_POR_ACIERTO = 150.0  # 18x premio (aprox)
    
    # Determinar horas a jugar basado en la hora de corte
    horas_a_jugar = [h for h in horas_manana if h <= hora_corte]
    total_horas = len(horas_a_jugar)
    GASTO_DIARIO = total_horas * APUESTA_POR_HORA

    # 3. SIMULACIÓN
    datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
    
    resultados_simulacion = []
    
    for fecha, df_dia in datos.groupby('Fecha'):
        aciertos_totales = 0
        gasto_dia = 0
        ganancia_bruta_dia = 0
        
        # Jugar solo horas de mañana hasta la hora de corte
        df_manana_jugar = df_dia[df_dia['Hora'].isin(horas_a_jugar)].copy()
        
        for _, row in df_manana_jugar.iterrows():
            hora_filtro = row['Hora']
            animal_salio = row['Animal']
            
            if hora_filtro in top_10_map_manana and animal_salio in top_10_map_manana[hora_filtro]:
                aciertos_totales += 1
        
        gasto_dia = GASTO_DIARIO
        ganancia_bruta_dia = aciertos_totales * GANANCIA_POR_ACIERTO
        
        resultados_simulacion.append({
            'Fecha': fecha,
            'Horas_Jugadas': total_horas,
            'Aciertos_Manana': aciertos_totales,
            'Gasto': gasto_dia,
            'Ganancia_Bruta': ganancia_bruta_dia,
            'Ganancia_Neta': ganancia_bruta_dia - gasto_dia
        })
    
    # 4. REPORTE FINAL
    df_resultados = pd.DataFrame(resultados_simulacion)
    dias_completos = df_resultados[df_resultados['Aciertos_Manana'].notna()]
    
    total_dias = len(dias_completos)
    gasto_total = df_resultados['Gasto'].sum()
    ganancia_bruta_total = df_resultados['Ganancia_Bruta'].sum()
    ganancia_neta_total = df_resultados['Ganancia_Neta'].sum()
    
    aciertos_promedio = df_resultados['Aciertos_Manana'].mean()
    
    print("\n" + "="*70)
    print(f"     📊 RESUMEN ESTRATEGIA SOLO MAÑANA (Hasta {hora_corte})")
    print("="*70)
    print(f"✅ Días Analizados: {total_dias}")
    print(f"✅ Horas jugadas por día: {total_horas} ({', '.join(horas_a_jugar)})")
    print(f"✅ Aciertos promedio por día: {aciertos_promedio:.2f}")
    print("-" * 70)
    print(f"💵 Gasto Total: {gasto_total:,.2f} Bs")
    print(f"💰 Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
    print(f"📊 **GANANCIA/PÉRDIDA NETA TOTAL:** {ganancia_neta_total:,.2f} Bs")
    print("-" * 70)
    
    if gasto_total > 0:
        roi = (ganancia_neta_total / gasto_total) * 100
        print(f"📈 **Retorno de la Inversión (ROI):** {roi:,.2f}%")
    
    # Análisis de rentabilidad
    if ganancia_neta_total > 0:
        print("\n🎉 ¡La estrategia SOLO MAÑANA generó ganancias!")
        print(f"💰 Ganancia promedio por día: {ganancia_neta_total/total_dias:,.2f} Bs")
    elif ganancia_neta_total < 0:
        print(f"\n⚠️  La estrategia generó pérdidas.")
        print(f"📉 Pérdida promedio por día: {ganancia_neta_total/total_dias:,.2f} Bs")
    else:
        print("\n🟡 Resultado: Punto de Equilibrio.")
    
    # Mostrar distribución de aciertos
    print(f"\n📋 DISTRIBUCIÓN DE ACIERTOS POR DÍA:")
    distribucion = df_resultados['Aciertos_Manana'].value_counts().sort_index()
    for aciertos, conteo in distribucion.items():
        porcentaje = (conteo / total_dias) * 100
        print(f"   • {aciertos} aciertos: {conteo} días ({porcentaje:.1f}%)")
    
    return df_resultados

def evaluacion_estrategia_filtrada(datos, filtro_ganancia=True):
    """
    Evalúa la estrategia dinámica filtrando días con ganancia o aplicando umbrales
    """
    print("\n--- 🧠 Evaluación Estrategia DINÁMICA FILTRADA ---")
    
    # DEBUG: Verificar datos de entrada
    print("🔍 DEBUG - Columnas en datos de entrada:", list(datos.columns))
    print("🔍 DEBUG - Primeras filas:")
    print(datos[['Timestamp', 'Hora', 'Animal']].head(3) if 'Timestamp' in datos.columns else "No hay Timestamp")
    
    # Calcular matriz de frecuencia (base)
    try:
        frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        print(f"✅ Matriz de frecuencia calculada: {len(frecuencia_completa)} registros")
    except Exception as e:
        print(f"❌ Error calculando frecuencia: {e}")
        return None
    
    top_10_map = {}
    horas_con_datos = frecuencia_completa['Hora'].unique()
    print(f"🔍 Horas con datos: {sorted(horas_con_datos)}")
    
    for hora_24h in horas_con_datos:
        top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
        top_10_map[hora_24h] = top_10_lista
    
    print(f"✅ Top-10 map creado: {len(top_10_map)} horas")
    
    # Parámetros de la estrategia
    HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
    HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
    
    GASTO_TARDE = 3000.0
    GANANCIA_POR_ACIERTO = 1500.0

    # Asegurar que tenemos la columna Fecha
    if 'Fecha' not in datos.columns:
        if 'Timestamp' in datos.columns:
            datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
            print("✅ Columna Fecha creada desde Timestamp")
        else:
            print("❌ ERROR: No hay columna Timestamp ni Fecha")
            return None
    
    # Verificar agrupación por fecha
    fechas_unicas = datos['Fecha'].nunique()
    print(f"🔍 Fechas únicas encontradas: {fechas_unicas}")
    
    resultados_simulacion = []
    dias_procesados = 0
    
    for fecha, df_dia in datos.groupby('Fecha'):
        dias_procesados += 1
        print(f"🔍 Procesando día {dias_procesados}: {fecha} - {len(df_dia)} registros")
        
        aciertos_manana = 0
        
        # Filtrar horas de la mañana
        df_manana = df_dia[df_dia['Hora'].isin(HORAS_MANANA)].copy()
        print(f"   - Registros en mañana: {len(df_manana)}")
        
        for _, row in df_manana.iterrows():
            hora_filtro = row['Hora']
            animal_salio = row['Animal']
            
            if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                aciertos_manana += 1
        
        print(f"   - Aciertos en mañana: {aciertos_manana}")
        
        # REGLA PRINCIPAL: Solo jugar si 0 o 1 aciertos en la mañana
        jugar_tarde = (aciertos_manana <= 1)
        
        aciertos_tarde = 0
        ganancia_bruta_tarde = 0
        gasto_tarde = 0
        
        # Filtrar horas de la tarde
        df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)].copy()
        print(f"   - Registros en tarde: {len(df_tarde)}")

        if jugar_tarde:
            gasto_tarde = GASTO_TARDE
            
            for _, row in df_tarde.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    aciertos_tarde += 1
            
            ganancia_bruta_tarde = aciertos_tarde * GANANCIA_POR_ACIERTO
            print(f"   - Aciertos en tarde: {aciertos_tarde}")

        ganancia_neta = ganancia_bruta_tarde - gasto_tarde
        
        # CREAR LAS COLUMNAS CORRECTAMENTE
        resultados_simulacion.append({
            'Fecha': fecha,
            'Aciertos_Manana': aciertos_manana,
            'Jugar_Tarde': 'SÍ' if jugar_tarde else 'NO',
            'Aciertos_Tarde': aciertos_tarde,
            'Gasto': gasto_tarde,
            'Ganancia_Bruta': ganancia_bruta_tarde,
            'Ganancia_Neta': ganancia_neta
        })
    
    print(f"✅ Días procesados: {dias_procesados}")
    print(f"✅ Resultados en simulación: {len(resultados_simulacion)}")
    
    if len(resultados_simulacion) == 0:
        print("❌ ERROR: No se crearon resultados. Posibles causas:")
        print("   - No hay datos en las horas especificadas")
        print("   - Problema con las columnas Hora o Fecha")
        print("   - No hay coincidencias en el top_10_map")
        return None
    
    df_resultados = pd.DataFrame(resultados_simulacion)
    
    # VERIFICAR que las columnas se crearon
    print("🔍 Columnas creadas en resultados:", list(df_resultados.columns))
    print("🔍 Primeras filas de resultados:")
    print(df_resultados.head(3))
    
    # FILTRADO POR GANANCIA
    if filtro_ganancia:
        # Verificar que las columnas existen antes de filtrar
        if 'Jugar_Tarde' not in df_resultados.columns or 'Ganancia_Neta' not in df_resultados.columns:
            print("❌ ERROR: Columnas necesarias no se crearon correctamente")
            print("Columnas disponibles:", list(df_resultados.columns))
            return None
            
        df_filtrado = df_resultados[
            (df_resultados['Jugar_Tarde'] == 'SÍ') & 
            (df_resultados['Ganancia_Neta'] > 0)
        ].copy()
        
        print(f"🎯 FILTRO APLICADO: Solo días con GANANCIA NETA POSITIVA")
        print(f"   - Días antes del filtro: {len(df_resultados)}")
        print(f"   - Días después del filtro: {len(df_filtrado)}")
    else:
        if 'Jugar_Tarde' not in df_resultados.columns:
            print("❌ ERROR: Columna 'Jugar_Tarde' no encontrada")
            return None
        df_filtrado = df_resultados[df_resultados['Jugar_Tarde'] == 'SÍ'].copy()
    
    if len(df_filtrado) == 0:
        print("❌ No hay días que cumplan los criterios de filtrado")
        return None
    
    # CÁLCULOS FINALES
    total_dias_jugados = len(df_filtrado)
    gasto_total = df_filtrado['Gasto'].sum()
    ganancia_bruta_total = df_filtrado['Ganancia_Bruta'].sum()
    ganancia_neta_total = df_filtrado['Ganancia_Neta'].sum()
    
    print("\n" + "="*70)
    print("        🚀 RESUMEN ESTRATEGIA FILTRADA")
    print("="*70)
    print(f"✅ Días Completos Analizados: {len(df_resultados)}")
    print(f"✅ Días Jugados (con regla 0 o 1 acierto mañana): {len(df_resultados[df_resultados['Jugar_Tarde'] == 'SÍ'])}")
    print(f"🎯 Días FILTRADOS con Ganancia: {total_dias_jugados}")
    print("-" * 70)
    print(f"💵 Gasto Total: {gasto_total:,.2f} Bs")
    print(f"💰 Ganancia Bruta Total: {ganancia_bruta_total:,.2f} Bs")
    print(f"📊 **GANANCIA NETA TOTAL:** {ganancia_neta_total:,.2f} Bs")
    print("-" * 70)
    
    if gasto_total > 0:
        roi = (ganancia_neta_total / gasto_total) * 100
        print(f"📈 **Retorno de la Inversión (ROI):** {roi:,.2f}%")
    
    # ANÁLISIS DETALLADO
    if total_dias_jugados > 0:
        print(f"\n📊 DISTRIBUCIÓN DE RESULTADOS:")
        distribucion = df_filtrado['Ganancia_Neta'].value_counts().sort_index(ascending=False)
        for ganancia, conteo in distribucion.items():
            porcentaje = (conteo / total_dias_jugados) * 100
            print(f"   • {ganancia:+,.0f} Bs: {conteo} días ({porcentaje:.1f}%)")
    
    # DÍAS CON PÉRDIDA (para referencia)
    dias_con_perdida = len(df_resultados[
        (df_resultados['Jugar_Tarde'] == 'SÍ') & 
        (df_resultados['Ganancia_Neta'] < 0)
    ])
    
    print(f"\n⚠️  Días eliminados con pérdida: {dias_con_perdida}")
    
    # MOSTRAR LOS MEJORES DÍAS
    if len(df_filtrado) > 0:
        print(f"\n🏆 TOP 5 MEJORES DÍAS (con filtro):")
        top_5 = df_filtrado.nlargest(5, 'Ganancia_Neta')[['Fecha', 'Aciertos_Manana', 'Aciertos_Tarde', 'Ganancia_Neta']]
        print(top_5.to_string(index=False))
    
    return df_filtrado


def analisis_estadistico_avanzado(datos):
    """
    Análisis avanzado para encontrar patrones en días ganadores vs perdedores
    """
    print("\n--- 📊 ANÁLISIS ESTADÍSTICO AVANZADO ---")
    
    # VERIFICACIÓN COMPLETA DE DATOS
    print("🔍 DIAGNÓSTICO COMPLETO:")
    print(f"   • Total de registros en datos: {len(datos)}")
    print(f"   • Columnas disponibles: {list(datos.columns)}")
    
    # Verificar si hay datos reales
    if len(datos) == 0:
        print("❌ ERROR: El DataFrame está completamente vacío")
        return None
    
    # Verificar columnas críticas
    columnas_criticas = ['Timestamp', 'Hora', 'Animal']
    for col in columnas_criticas:
        if col not in datos.columns:
            print(f"❌ ERROR: Columna crítica '{col}' no encontrada")
            return None
    
    # Verificar valores en columnas críticas
    print(f"🔍 Valores en columnas críticas:")
    print(f"   • Timestamp: {datos['Timestamp'].notna().sum()} valores no nulos")
    print(f"   • Hora: {datos['Hora'].notna().sum()} valores no nulos") 
    print(f"   • Animal: {datos['Animal'].notna().sum()} valores no nulos")
    
    # Mostrar algunas filas reales
    print(f"🔍 Primeras 3 filas REALES:")
    print(datos[['Timestamp', 'Hora', 'Animal']].head(3))
    
    # Verificar formato de horas
    print(f"🔍 Valores únicos en Hora: {sorted(datos['Hora'].unique())[:10]}")  # Primeros 10
    
    try:
        # Calcular matriz de frecuencia
        frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
        print(f"✅ Matriz de frecuencia: {len(frecuencia_completa)} registros")
        
        if len(frecuencia_completa) == 0:
            print("❌ La matriz de frecuencia está vacía - revisar formato de Hora y Animal")
            return None
            
        top_10_map = {}
        horas_con_datos = frecuencia_completa['Hora'].unique()
        print(f"🔍 Horas con datos en frecuencia: {sorted(horas_con_datos)}")
        
        for hora_24h in horas_con_datos:
            top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
            top_10_map[hora_24h] = top_10_lista

        HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
        
        GASTO_TARDE = 1200.0
        GANANCIA_POR_ACIERTO = 600.0

        # Preparar datos
        datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
        datos['Dia_Semana'] = datos['Timestamp'].dt.day_name()
        datos['Mes'] = datos['Timestamp'].dt.month
        datos['Dia_Mes'] = datos['Timestamp'].dt.day
        
        print(f"🔍 Fechas únicas después de procesar: {datos['Fecha'].nunique()}")
        
        resultados = []
        dias_procesados = 0
        
        for fecha, df_dia in datos.groupby('Fecha'):
            dias_procesados += 1
            
            # Calcular aciertos en la mañana
            aciertos_manana = 0
            df_manana = df_dia[df_dia['Hora'].isin(HORAS_MANANA)]
            
            for _, row in df_manana.iterrows():
                hora_filtro = row['Hora']
                animal_salio = row['Animal']
                
                if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                    aciertos_manana += 1
            
            # DECISIÓN: ¿Jugar en la tarde?
            jugar_tarde = (aciertos_manana <= 1)
            
            # Calcular resultados de la tarde (si se juega)
            aciertos_tarde = 0
            if jugar_tarde:
                df_tarde = df_dia[df_dia['Hora'].isin(HORAS_TARDE)]
                for _, row in df_tarde.iterrows():
                    hora_filtro = row['Hora']
                    animal_salio = row['Animal']
                    
                    if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                        aciertos_tarde += 1
            
            ganancia_neta = (aciertos_tarde * GANANCIA_POR_ACIERTO) - (GASTO_TARDE if jugar_tarde else 0)
            
            resultados.append({
                'Fecha': fecha,
                'Dia_Semana': df_dia['Dia_Semana'].iloc[0],
                'Mes': df_dia['Mes'].iloc[0],
                'Dia_Mes': df_dia['Dia_Mes'].iloc[0],
                'Aciertos_Manana': aciertos_manana,
                'Jugar_Tarde': jugar_tarde,
                'Aciertos_Tarde': aciertos_tarde,
                'Ganancia_Neta': ganancia_neta
            })
        
        print(f"✅ Días procesados: {dias_procesados}")
        print(f"✅ Resultados creados: {len(resultados)}")
        
        if len(resultados) == 0:
            print("❌ No se crearon resultados - posiblemente no hay datos en las horas especificadas")
            print(f"   Horas mañana buscadas: {HORAS_MANANA}")
            print(f"   Horas tarde buscadas: {HORAS_TARDE}")
            return None
        
        df_analisis = pd.DataFrame(resultados)
        print("✅ Análisis completado - Columnas creadas:", list(df_analisis.columns))
        
        # Continuar con el análisis normal...
        df_jugados = df_analisis[df_analisis['Jugar_Tarde'] == True]
        
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   • Total días analizados: {len(df_analisis)}")
        print(f"   • Días que cumplen regla (0-1 aciertos mañana): {len(df_jugados)}")
        
        if len(df_jugados) > 0:
            # ... resto del análisis igual
            print("📈 ANÁLISIS POR DÍA DE LA SEMANA:")
            analisis_dia_semana = df_jugados.groupby('Dia_Semana').agg({
                'Ganancia_Neta': ['count', 'sum', 'mean'],
                'Aciertos_Tarde': 'mean'
            }).round(2)
            print(analisis_dia_semana)
        else:
            print("❌ No hay días que cumplan la condición para jugar en la tarde")
        
        return df_analisis
        
    except Exception as e:
        print(f"❌ Error en análisis estadístico: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def patrones_dias_rentables(datos):
    """
    Identifica patrones específicos que predicen días rentables vs no rentables
    """
    print("\n--- 🔍 ANÁLISIS DE PATRONES PARA DÍAS RENTABLES ---")
    
    # Calcular matriz de frecuencia
    frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
    
    top_10_map = {}
    for hora_24h in frecuencia_completa['Hora'].unique():
        top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(10)['Animal'].tolist()
        top_10_map[hora_24h] = top_10_lista

    HORAS_MANANA = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
    HORAS_TARDE = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
    
    GASTO_TARDE = 3000.0
    GANANCIA_POR_ACIERTO = 1500.0

    datos['Fecha'] = pd.to_datetime(datos['Timestamp']).dt.date
    datos['Dia_Semana'] = datos['Timestamp'].dt.day_name()
    datos['Mes'] = datos['Timestamp'].dt.month
    datos['Dia_Mes'] = datos['Timestamp'].dt.day
    
    resultados_detallados = []
    
    for fecha, df_dia in datos.groupby('Fecha'):
        # Análisis por hora de la mañana
        aciertos_por_hora_manana = {}
        animales_por_hora_manana = {}
        
        for hora in HORAS_MANANA:
            df_hora = df_dia[df_dia['Hora'] == hora]
            if not df_hora.empty:
                animal_salio = df_hora['Animal'].iloc[0]
                acierto = 1 if (hora in top_10_map and animal_salio in top_10_map[hora]) else 0
                aciertos_por_hora_manana[hora] = acierto
                animales_por_hora_manana[hora] = animal_salio
        
        aciertos_manana = sum(aciertos_por_hora_manana.values())
        jugar_tarde = (aciertos_manana <= 1)
        
        # Resultados de la tarde
        aciertos_tarde = 0
        if jugar_tarde:
            for _, row in df_dia[df_dia['Hora'].isin(HORAS_TARDE)].iterrows():
                if row['Hora'] in top_10_map and row['Animal'] in top_10_map[row['Hora']]:
                    aciertos_tarde += 1
        
        ganancia_neta = (aciertos_tarde * GANANCIA_POR_ACIERTO) - (GASTO_TARDE if jugar_tarde else 0)
        
        resultados_detallados.append({
            'Fecha': fecha,
            'Dia_Semana': df_dia['Dia_Semana'].iloc[0],
            'Mes': df_dia['Mes'].iloc[0],
            'Dia_Mes': df_dia['Dia_Mes'].iloc[0],
            'Aciertos_Manana': aciertos_manana,
            'Aciertos_Tarde': aciertos_tarde,
            'Ganancia_Neta': ganancia_neta,
            'Jugar_Tarde': jugar_tarde,
            # Patrones específicos por hora
            'Acierto_8am': aciertos_por_hora_manana.get('08:00:00', 0),
            'Acierto_9am': aciertos_por_hora_manana.get('09:00:00', 0),
            'Acierto_10am': aciertos_por_hora_manana.get('10:00:00', 0),
            'Acierto_11am': aciertos_por_hora_manana.get('11:00:00', 0),
            'Acierto_12pm': aciertos_por_hora_manana.get('12:00:00', 0),
            'Acierto_1pm': aciertos_por_hora_manana.get('13:00:00', 0),
            # Animales que salieron (podrían tener patrones)
            'Animal_8am': animales_por_hora_manana.get('08:00:00', 'N/A'),
            'Animal_9am': animales_por_hora_manana.get('09:00:00', 'N/A'),
            'Animal_10am': animales_por_hora_manana.get('10:00:00', 'N/A'),
            'Animal_11am': animales_por_hora_manana.get('11:00:00', 'N/A'),
            'Animal_12pm': animales_por_hora_manana.get('12:00:00', 'N/A'),
            'Animal_1pm': animales_por_hora_manana.get('13:00:00', 'N/A')
        })
    
    df_analisis = pd.DataFrame(resultados_detallados)
    df_jugados = df_analisis[df_analisis['Jugar_Tarde'] == True]
    
    print("🎯 PATRONES ENCONTRADOS:")
    print("="*60)
    
    # 1. ANÁLISIS POR DÍA DE LA SEMANA
    print("\n📅 1. RENTABILIDAD POR DÍA DE LA SEMANA:")
    analisis_dias = df_jugados.groupby('Dia_Semana').agg({
        'Ganancia_Neta': ['count', 'sum', 'mean'],
        'Aciertos_Tarde': 'mean'
    }).round(2)
    
    print(analisis_dias)
    
    # 2. PATRONES POR HORA DE LA MAÑANA
    print("\n⏰ 2. PATRONES POR COMPORTAMIENTO EN MAÑANA:")
    
    # Combinaciones de aciertos por hora que predicen éxito
    print("Combinaciones que PREDICEN ÉXITO en la tarde:")
    combinaciones_exitosas = df_jugados[df_jugados['Ganancia_Neta'] > 0]
    if len(combinaciones_exitosas) > 0:
        patron_exito = combinaciones_exitosas[['Acierto_8am', 'Acierto_9am', 'Acierto_10am', 
                                             'Acierto_11am', 'Acierto_12pm', 'Acierto_1pm']].mean()
        print("Aciertos promedio por hora en días GANADORES:")
        for hora, prob in patron_exito.items():
            print(f"   {hora}: {prob:.2%}")
    
    print("\nCombinaciones que PREDICEN FRACASO en la tarde:")
    combinaciones_fracaso = df_jugados[df_jugados['Ganancia_Neta'] < 0]
    if len(combinaciones_fracaso) > 0:
        patron_fracaso = combinaciones_fracaso[['Acierto_8am', 'Acierto_9am', 'Acierto_10am', 
                                              'Acierto_11am', 'Acierto_12pm', 'Acierto_1pm']].mean()
        print("Aciertos promedio por hora en días PERDEDORES:")
        for hora, prob in patron_fracaso.items():
            print(f"   {hora}: {prob:.2%}")
    
    # 3. REGLAS DE DECISIÓN SIMPLES
    print("\n🎲 3. REGLAS DE DECISIÓN (CUÁNDO APOSTAR):")
    
    # Regla 1: Días específicos más rentables
    dias_recomendados = df_jugados.groupby('Dia_Semana')['Ganancia_Neta'].mean()
    dias_recomendados = dias_recomendados[dias_recomendados > 0].index.tolist()
    
    if dias_recomendados:
        print(f"   ✅ PRIORIZAR estos días: {', '.join(dias_recomendados)}")
    
    # Regla 2: Patrones horarios específicos
    print("\n   🔍 PATRONES HORARIOS RECOMENDADOS:")
    
    # Buscar patrones específicos
    for patron in [
        {'Acierto_8am': 0, 'Acierto_9am': 0, 'Acierto_10am': 0},  # Mal inicio
        {'Acierto_11am': 1, 'Acierto_12pm': 0},  # Acierto a media mañana
        {'Acierto_1pm': 1}  # Acierto justo antes del corte
    ]:
        # Simplificado - en implementación real harías filtros más complejos
        pass
    
    # 4. SISTEMA DE ALERTA EN TIEMPO REAL
    print("\n🚨 4. SISTEMA DE ALERTA EN TIEMPO REAL:")
    print("   (Basado en tus datos históricos)")
    
    # Umbrales de seguridad
    print(f"   • Días totales analizados: {len(df_jugados)}")
    print(f"   • Días con ganancia: {len(df_jugados[df_jugados['Ganancia_Neta'] > 0])}")
    print(f"   • Días con pérdida: {len(df_jugados[df_jugados['Ganancia_Neta'] < 0])}")
    print(f"   • Probabilidad de ganar: {(len(df_jugados[df_jugados['Ganancia_Neta'] > 0]) / len(df_jugados)):.1%}")
    
    return df_analisis

def predictor_dia_actual(datos):
    """
    Predice si HOY es un día seguro para apostar en base a patrones históricos
    """
    print("\n--- 🔮 PREDICCIÓN PARA HOY ---")
    
    # Obtener el análisis completo
    df_analisis = patrones_dias_rentables(datos)
    
    # Obtener día actual
    from datetime import datetime
    hoy = datetime.now().date()
    dia_semana_hoy = hoy.strftime("%A")
    
    print(f"\n📅 HOY ES: {hoy} ({dia_semana_hoy})")
    
    # Filtrar datos históricos para este día de la semana
    df_mismo_dia = df_analisis[df_analisis['Dia_Semana'] == dia_semana_hoy]
    df_mismo_dia_jugado = df_mismo_dia[df_mismo_dia['Jugar_Tarde'] == True]
    
    if len(df_mismo_dia_jugado) == 0:
        print("❌ No hay suficientes datos históricos para este día")
        return
    
    # Calcular métricas
    total_dias = len(df_mismo_dia_jugado)
    dias_ganadores = len(df_mismo_dia_jugado[df_mismo_dia_jugado['Ganancia_Neta'] > 0])
    dias_perdedores = len(df_mismo_dia_jugado[df_mismo_dia_jugado['Ganancia_Neta'] < 0])
    
    probabilidad_ganar = dias_ganadores / total_dias if total_dias > 0 else 0
    ganancia_promedio = df_mismo_dia_jugado['Ganancia_Neta'].mean()
    
    print(f"📊 ESTADÍSTICAS PARA LOS {dia_semana_hoy}s:")
    print(f"   • Días jugados históricamente: {total_dias}")
    print(f"   • Días con ganancia: {dias_ganadores} ({probabilidad_ganar:.1%})")
    print(f"   • Días con pérdida: {dias_perdedores}")
    print(f"   • Ganancia promedio: {ganancia_promedio:,.0f} Bs")
    
    # RECOMENDACIÓN
    print(f"\n🎯 RECOMENDACIÓN PARA HOY:")
    
    if probabilidad_ganar >= 0.6 and ganancia_promedio > 500:
        print("   ✅ ¡CONFIANZA ALTA! Es un buen día para apostar en la tarde")
        print(f"   📈 Probabilidad histórica de ganar: {probabilidad_ganar:.1%}")
        print(f"   💰 Ganancia promedio esperada: {ganancia_promedio:,.0f} Bs")
    
    elif probabilidad_ganar >= 0.4:
        print("   ⚠️  CONFIANZA MEDIA - Considera apostar moderadamente")
        print(f"   📊 Probabilidad histórica: {probabilidad_ganar:.1%}")
        
    else:
        print("   ❌ CONFIANZA BAJA - Mejor no apostar hoy")
        print(f"   📉 Probabilidad histórica: {probabilidad_ganar:.1%}")
    
    # MOSTRAR PATRONES ESPECÍFICOS PARA HOY
    print(f"\n🔍 PATRONES ESPECÍFICOS A OBSERVAR HOY:")
    
    if len(df_mismo_dia_jugado) > 0:
        # Patrones de aciertos en mañana para días exitosos
        dias_exitosos = df_mismo_dia_jugado[df_mismo_dia_jugado['Ganancia_Neta'] > 0]
        if len(dias_exitosos) > 0:
            print("   En días EXITOSOS de este día, los patrones fueron:")
            for col in ['Acierto_8am', 'Acierto_9am', 'Acierto_10am', 'Acierto_11am', 'Acierto_12pm', 'Acierto_1pm']:
                if col in dias_exitosos.columns:
                    prob = dias_exitosos[col].mean()
                    print(f"     • {col}: {prob:.0%} de aciertos")
    
    return probabilidad_ganar, ganancia_promedio

def ver_ultimos_registros_y_faltantes(datos):
    """
    Muestra los últimos registros y qué animales faltan por agregar hoy
    """
    print("\n--- 📋 ÚLTIMOS REGISTROS Y ANÁLISIS DEL DÍA ---")
    
    # Verificar que tenemos datos
    if len(datos) == 0:
        print("❌ No hay registros en la base de datos")
        return
    
    # Ordenar por fecha y hora más reciente
    datos_ordenados = datos.sort_values('Timestamp', ascending=False)
    
    # Mostrar últimos 10 registros
    print("\n📊 ÚLTIMOS 10 REGISTROS (más recientes primero):")
    print("="*60)
    columnas_mostrar = ['Fecha', 'Hora', 'Animal', 'Numero']
    ultimos_10 = datos_ordenados[columnas_mostrar].head(10)
    print(ultimos_10.to_string(index=False))
    
    # Encontrar la fecha más reciente
    fecha_mas_reciente = datos_ordenados['Fecha'].iloc[0]
    print(f"\n🗓️  FECHA MÁS RECIENTE EN DATOS: {fecha_mas_reciente}")
    
    # Filtrar datos de hoy (si existen)
    from datetime import date
    hoy = date.today()
    datos_hoy = datos[datos['Fecha'] == hoy]
    
    if len(datos_hoy) > 0:
        print(f"\n✅ REGISTROS DE HOY ({hoy}):")
        print("="*40)
        registros_hoy = datos_hoy[['Hora', 'Animal', 'Numero']].sort_values('Hora')
        print(registros_hoy.to_string(index=False))
        
        # Horas posibles del día
        horas_posibles = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', 
                         '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', 
                         '18:00:00', '19:00:00']
        
        # Encontrar horas faltantes de hoy
        horas_registradas_hoy = datos_hoy['Hora'].unique()
        horas_faltantes = [h for h in horas_posibles if h not in horas_registradas_hoy]
        
        if horas_faltantes:
            print(f"\n⏰ HORAS FALTANTES POR AGREGAR HOY:")
            for hora in horas_faltantes:
                print(f"   • {hora}")
        else:
            print(f"\n🎉 ¡TODAS LAS HORAS DE HOY ESTÁN COMPLETAS!")
            
    else:
        print(f"\n📝 HOY NO HAY REGISTROS ({hoy})")
        print("   Usa la Opción 1 para agregar el primer sorteo del día")
    
    # Análisis de animales recientes
    print(f"\n🔍 ANÁLISIS DE ANIMALES RECIENTES:")
    print("="*40)
    
    # Últimos 20 animales (sin duplicados consecutivos)
    ultimos_20 = datos_ordenados.head(20)
    animales_recientes = ultimos_20['Animal'].unique()
    
    print(f"Animales en últimos 20 sorteos ({len(animales_recientes)} únicos):")
    for i, animal in enumerate(animales_recientes, 1):
        print(f"   {i:2d}. {animal}")
    
    # Frecuencia de animales en últimos 50 sorteos
    ultimos_50 = datos_ordenados.head(50)
    frecuencia_reciente = ultimos_50['Animal'].value_counts().head(10)
    
    print(f"\n🏆 TOP 10 ANIMALES MÁS FRECUENTES (últimos 50 sorteos):")
    for animal, conteo in frecuencia_reciente.items():
        porcentaje = (conteo / len(ultimos_50)) * 100
        print(f"   • {animal}: {conteo} veces ({porcentaje:.1f}%)")
    
    return datos_ordenados.head(10)

def ver_estado_actual_dia(datos):
    """
    Función rápida para ver solo el estado del día actual
    """
    from datetime import date
    hoy = date.today()
    
    print(f"\n--- 📅 ESTADO ACTUAL - {hoy} ---")
    
    datos_hoy = datos[datos['Fecha'] == hoy]
    
    if len(datos_hoy) == 0:
        print("❌ No hay registros para hoy")
        print("   Usa la Opción 1 para agregar sorteos")
        return
    
    # Mostrar registros de hoy ordenados por hora
    registros_hoy = datos_hoy[['Hora', 'Animal', 'Numero']].sort_values('Hora')
    print(f"\n✅ SORTEOS REGISTRADOS HOY ({len(registros_hoy)}):")
    print(registros_hoy.to_string(index=False))
    
    # Horas esperadas vs actuales
    horas_manana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
    horas_tarde = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
    
    horas_registradas = datos_hoy['Hora'].tolist()
    
    manana_registradas = [h for h in horas_registradas if h in horas_manana]
    tarde_registradas = [h for h in horas_registradas if h in horas_tarde]
    
    print(f"\n⏰ RESUMEN HORARIO:")
    print(f"   • Mañana (8am-1pm): {len(manana_registradas)}/{len(horas_manana)} horas")
    print(f"   • Tarde (2pm-7pm): {len(tarde_registradas)}/{len(horas_tarde)} horas")
    
    # Horas faltantes
    todas_horas = horas_manana + horas_tarde
    horas_faltantes = [h for h in todas_horas if h not in horas_registradas]
    
    if horas_faltantes:
        print(f"\n📝 HORAS FALTANTES:")
        for hora in horas_faltantes:
            print(f"   • {hora}")
    else:
        print(f"\n🎉 ¡DÍA COMPLETO! Todas las horas registradas")

def analizar_rachas_tempranas(datos, horas_evaluacion=3, umbral_aciertos=3):
    """
    Analiza si las rachas tempranas predicen días exitosos
    """
    print(f"\n--- 🔍 ANÁLISIS DE RACHAS TEMPRANAS ---")
    print(f"Buscando: {umbral_aciertos}+ aciertos en primeras {horas_evaluacion} horas")
    
    # Horarios de evaluación
    if horas_evaluacion == 2:
        horas_a_evaluar = ['08:00:00', '09:00:00']
    elif horas_evaluacion == 3:
        horas_a_evaluar = ['08:00:00', '09:00:00', '10:00:00']
    elif horas_evaluacion == 4:
        horas_a_evaluar = ['08:00:00', '09:00:00', '10:00:00', '11:00:00']
    
    # Calcular matriz de frecuencia para el top-10
    frecuencia_completa = datos.groupby('Hora')['Animal'].value_counts(normalize=True).mul(100).rename('Probabilidad').reset_index()
    
    top_10_map = {}
    for hora_24h in frecuencia_completa['Hora'].unique():
        top_10_lista = frecuencia_completa[frecuencia_completa['Hora'] == hora_24h].head(15)['Animal'].tolist()
        top_10_map[hora_24h] = top_10_lista
    
    resultados = []
    
    for fecha, df_dia in datos.groupby('Fecha'):
        # Contar aciertos en horas tempranas
        aciertos_tempranos = 0
        df_temprano = df_dia[df_dia['Hora'].isin(horas_a_evaluar)]
        
        for _, row in df_temprano.iterrows():
            hora_filtro = row['Hora']
            animal_salio = row['Animal']
            
            if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                aciertos_tempranos += 1
        
        # Calcular aciertos en todo el día (mañana + tarde)
        aciertos_mañana = 0
        aciertos_tarde = 0
        
        horas_mañana = ['08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00']
        horas_tarde = ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00']
        
        for _, row in df_dia.iterrows():
            hora_filtro = row['Hora']
            animal_salio = row['Animal']
            
            if hora_filtro in top_10_map and animal_salio in top_10_map[hora_filtro]:
                if hora_filtro in horas_mañana:
                    aciertos_mañana += 1
                elif hora_filtro in horas_tarde:
                    aciertos_tarde += 1
        
        resultados.append({
            'Fecha': fecha,
            'Aciertos_Tempranos': aciertos_tempranos,
            'Aciertos_Mañana_Total': aciertos_mañana,
            'Aciertos_Tarde_Total': aciertos_tarde,
            'Aciertos_Dia_Completo': aciertos_mañana + aciertos_tarde,
            'Tiene_Racha_Temprana': aciertos_tempranos >= umbral_aciertos
        })
    
    df_analisis = pd.DataFrame(resultados)
    
    # Análisis de efectividad
    print(f"\n📈 RESULTADOS PARA {umbral_aciertos}+ ACIERTOS EN {horas_evaluacion} HORAS:")
    
    dias_con_racha = df_analisis[df_analisis['Tiene_Racha_Temprana'] == True]
    dias_sin_racha = df_analisis[df_analisis['Tiene_Racha_Temprana'] == False]
    
    if len(dias_con_racha) > 0:
        print(f"✅ DÍAS CON RACHA TEMPRANA ({len(dias_con_racha)} días):")
        print(f"   • Aciertos mañana promedio: {dias_con_racha['Aciertos_Mañana_Total'].mean():.2f}")
        print(f"   • Aciertos tarde promedio: {dias_con_racha['Aciertos_Tarde_Total'].mean():.2f}")
        print(f"   • Aciertos día completo: {dias_con_racha['Aciertos_Dia_Completo'].mean():.2f}")
        
        # Rentabilidad estimada (asumiendo 20 Bs por animal, 30x premio)
        ganancia_promedio_racha = (dias_con_racha['Aciertos_Dia_Completo'].mean() * 580) - (12 * 20)
        print(f"   • Ganancia neta estimada: {ganancia_promedio_racha:+.0f} Bs")
    
    if len(dias_sin_racha) > 0:
        print(f"\n📊 DÍAS SIN RACHA TEMPRANA ({len(dias_sin_racha)} días):")
        print(f"   • Aciertos mañana promedio: {dias_sin_racha['Aciertos_Mañana_Total'].mean():.2f}")
        print(f"   • Aciertos tarde promedio: {dias_sin_racha['Aciertos_Tarde_Total'].mean():.2f}")
        print(f"   • Aciertos día completo: {dias_sin_racha['Aciertos_Dia_Completo'].mean():.2f}")
        
        ganancia_promedio_sin_racha = (dias_sin_racha['Aciertos_Dia_Completo'].mean() * 580) - (12 * 20)
        print(f"   • Ganancia neta estimada: {ganancia_promedio_sin_racha:+.0f} Bs")
    
    return df_analisis

def probar_umbrales_rachas(datos):
    """
    Prueba diferentes umbrales para encontrar el óptimo
    """
    print("\n--- 🎯 OPTIMIZACIÓN DE UMBRALES DE RACHA ---")
    
    combinaciones = [
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3), (3, 4), 
        (4, 2), (4, 3), (4, 4)
    ]
    
    resultados_umbrales = []
    
    for horas, umbral in combinaciones:
        df_temp = analizar_rachas_tempranas(datos, horas_evaluacion=horas, umbral_aciertos=umbral)
        
        dias_con_racha = df_temp[df_temp['Tiene_Racha_Temprana'] == True]
        
        if len(dias_con_racha) > 0:
            efectividad = dias_con_racha['Aciertos_Dia_Completo'].mean()
            resultados_umbrales.append({
                'Horas_Evaluacion': horas,
                'Umbral_Aciertos': umbral,
                'Dias_Con_Racha': len(dias_con_racha),
                'Aciertos_Promedio_Dia': efectividad,
                'Ganancia_Estimada': (efectividad * 580) - (12 * 20)
            })
    
    df_umbrales = pd.DataFrame(resultados_umbrales)
    
    print(f"\n🏆 MEJORES COMBINACIONES:")
    mejores = df_umbrales.nlargest(3, 'Ganancia_Estimada')
    print(mejores[['Horas_Evaluacion', 'Umbral_Aciertos', 'Dias_Con_Racha', 'Ganancia_Estimada']].to_string(index=False))
    
    return df_umbrales

# -----------------------------------------------------------
# MENÚ PRINCIPAL
# -----------------------------------------------------------

def main_menu(datos, datosLotto):
    """Menú principal con todas las mejoras"""
    opciones = [
        "Ingresar Sorteo del Día (Actualizar Excel) ✍️",
        "Validación Cruzada de Precisión del Modelo (Márkov) 🔬", 
        "Mostrar Matriz de Probabilidad de Transición 📊", 
        "Predicción Siguiente (Basado en Último Animal - Márkov) 🔮", 
        "Probabilidad Máxima Histórica por Hora (Tabla Completa - TOP-10) 📈", 
        "Predicción Histórica por Hora Específica (TOP-10) ⏰", 
        "**ENTRENAR: Random Forest Optimizado (Auto-tuning)** 🌳",
        "**ENTRENAR: XGBoost Optimizado (Auto-tuning)** 🚀",
        "**CARGAR Modelo Pre-entrenado** 💾",
        "Evaluar Estrategia Dinámica (Frecuencia Histórica) 💰",
        "**Evaluar Estrategia Dinámica (Predicción de IA/ML)** 🧠",
        "Evaluar Estrategia Solo Mañana (Hasta Hora Específica) 🌅",
        "Evaluar Estrategia Dinámica Filtrada (Ganancia Neta Positiva) 🧠",
        "Análisis Estadístico Avanzado (Días Ganadores vs Perdedoras) 📊",
        "Análisis de Patrones para Días Rentables vs No Rentables 🔍",
        "Predicción para Hoy (¿Es un día seguro para apostar?) 🔮",
        "Ver Últimos Registros y Análisis del Día 📋",
        "Ver Estado Rápido del Día Actual 📅",
        "Análisis de Rachas Tempranas y su Impacto en el Día Completo 📈",
        "Salir del Programa"
    ]
    
    matriz_ia_entrenada = None
    modelo_cargado = None
    le_y_cargado = None
    
    while True:
        opcion_elegida = mostrar_menu("Programa de Predicción - Menú Principal", opciones)
        
        if opcion_elegida == 1:
            datos = agregar_datos_al_excel(datos, datosLotto)
            datos = agregar_caracteristicas_avanzadas(datos.copy())
            logger.info("Nuevo sorteo ingresado y características actualizadas")
        elif opcion_elegida == 2:
            validar_modelo_markov(datos.copy()) 
        elif opcion_elegida == 3:
            matriz_probabilidad_transicion(datos.copy())
        elif opcion_elegida == 4:
            mejor_prediccion_siguiente(datos.copy())
        elif opcion_elegida == 5:
            probabilidad_maxima_por_hora(datos.copy())
        elif opcion_elegida == 6: 
            prediccion_por_hora_especifica(datos.copy())
        
        # --- NUEVAS OPCIONES DE ML MEJORADAS ---
        elif opcion_elegida == 7: # Random Forest Optimizado
            matriz_ia_entrenada = random_forest_optimizado(datos.copy())
            if matriz_ia_entrenada:
                print("\n✅ Matriz de predicción Random Forest Optimizado generada.")
        
        elif opcion_elegida == 8: # XGBoost Optimizado
            matriz_ia_entrenada = xgboost_optimizado(datos.copy())
            if matriz_ia_entrenada:
                print("\n✅ Matriz de predicción XGBoost Optimizado generada.")
        
        elif opcion_elegida == 9: # Cargar Modelo
            print("\n--- 💾 CARGAR MODELO PRE-ENTRENADO ---")
            print("1. Cargar Random Forest")
            print("2. Cargar XGBoost")
            
            sub_opcion = input("Selecciona tipo de modelo: ").strip()
            if sub_opcion == '1':
                modelo_cargado, le_y_cargado, metricas = cargar_ultimo_modelo("random_forest")
                if modelo_cargado:
                    print("✅ Random Forest cargado - Listo para predicciones")
            elif sub_opcion == '2':
                modelo_cargado, le_y_cargado, metricas = cargar_ultimo_modelo("xgboost")
                if modelo_cargado:
                    print("✅ XGBoost cargado - Listo para predicciones")
        
        # --- OPCIONES DE EVALUACIÓN DINÁMICA ---
        elif opcion_elegida == 10: # Frecuencia Histórica
            evaluacion_estrategia_frecuencia(datos.copy())
        
        elif opcion_elegida == 11: # Predicción de IA
            if matriz_ia_entrenada is None and modelo_cargado is None:
                print("\n⚠️ **ADVERTENCIA:** Primero debes ejecutar la Opción 7, 8 o 9.")
            elif modelo_cargado:
                datos_con_features = agregar_caracteristicas_avanzadas(datos.copy())
                matriz_prediccion = predecir_top_k_por_hora(
                    modelo_cargado, le_y_cargado, datos_con_features.copy(), k=10
                )
                # 🆕 AGREGA ESTA LÍNEA:
                mostrar_matriz_prediccion(matriz_prediccion)
                evaluacion_estrategia_ia(datos.copy(), matriz_prediccion)
            else:
                # 🆕 AGREGA ESTA LÍNEA:
                mostrar_matriz_prediccion(matriz_ia_entrenada)
                evaluacion_estrategia_ia(datos.copy(), matriz_ia_entrenada)
        elif opcion_elegida == 12:  # Nueva opción
            print("\n🌅 ESTRATEGIA SOLO MAÑANA")
            print("1. Jugar hasta las 12:00 (4 horas)")
            print("2. Jugar hasta las 13:00 (5 horas)")
            
            sub_opcion = input("Selecciona horario: ").strip()
            if sub_opcion == '1':
                evaluacion_estrategia_solo_manana(datos.copy(), '12:00:00')
            elif sub_opcion == '2':
                evaluacion_estrategia_solo_manana(datos.copy(), '13:00:00')
        # En el main_menu, después de las opciones existentes, agrega:
        elif opcion_elegida == 13:  # Nueva opción para estrategia filtrada
            evaluacion_estrategia_filtrada(datos.copy(), filtro_ganancia=True)

        elif opcion_elegida == 14:  # Análisis estadístico
            analisis_estadistico_avanzado(datos.copy())
        # En el main_menu, agrega:
        elif opcion_elegida == 15:  # Análisis de patrones
            patrones_dias_rentables(datos.copy())

        elif opcion_elegida == 16:  # Predicción para hoy
            predictor_dia_actual(datos.copy())
        elif opcion_elegida == 17:  # Ver últimos registros y análisis
            ver_ultimos_registros_y_faltantes(datos.copy())
        elif opcion_elegida == 18:  # Ver estado rápido del día actual
            ver_estado_actual_dia(datos.copy())
        elif opcion_elegida == 19:  # Análisis de rachas tempranas
            print("\n🔍 OPCIONES DE ANÁLISIS DE RACHAS:")
            print("1. Análisis con 3+ aciertos en primeras 3 horas")
            print("2. Probar diferentes umbrales")
            
            sub_opcion = input("Selecciona: ").strip()
            if sub_opcion == '1':
                analizar_rachas_tempranas(datos.copy(), horas_evaluacion=3, umbral_aciertos=3)
            elif sub_opcion == '2':
                probar_umbrales_rachas(datos.copy())
        elif opcion_elegida == 20: # Salir
            print("\n👋 ¡Gracias por usar el programa de predicción! Saliendo...")
            break
        
        if opcion_elegida != 1:
            input("\nPresiona **Enter** para volver al menú...")

# -----------------------------------------------------------
# BLOQUE DE EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------

if __name__ == "__main__":
    datosLotto = "LottoActivo.xlsx"
    datos = None
    

    print(f"🐍 Python version: {sys.version}")
    
    # VERIFICAR DICCIONARIO PRIMERO
    if not verificar_diccionario_animales():
        print("❌ ERROR CRÍTICO: El diccionario de animales tiene problemas.")
        sys.exit(1)
    
    try:
        datos = pd.read_excel(datosLotto)
        print(f"✅ Archivo cargado: {len(datos)} registros")
        
        # 1. Limpieza de Animales y Números
        datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
        datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce') 
        numeros_invalidos = datos['Numero'].isna().sum()
        if numeros_invalidos > 0:
            print(f"⚠️  {numeros_invalidos} registros con números inválidos")
        
        # 2. Limpieza de Tiempos
        datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
        datos['Hora'] = datos['Hora'].astype(str).str.strip() 
        
        datos['Timestamp'] = pd.to_datetime(datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce')
        
        tiempos_invalidos = datos['Timestamp'].isna().sum()
        if tiempos_invalidos > 0:
            print(f"⚠️  {tiempos_invalidos} registros con timestamp inválido")
        
        datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
        datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip() 
        
        # 3. Ordenar
        datos = datos.sort_values(by='Timestamp').reset_index(drop=True)

        # --- APLICAR INGENIERÍA DE CARACTERÍSTICAS MEJORADA ---
        datos = agregar_caracteristicas_avanzadas(datos)
        
        # --- PREPARACIÓN FINAL ---
        print("\n📊 RESUMEN DE DATOS:")
        print(f"   • Total registros: {len(datos)}")
        print(f"   • Rango fechas: {datos['Timestamp'].min()} a {datos['Timestamp'].max()}")
        print(f"   • Animales únicos: {datos['Animal'].nunique()}")
        print(f"   • Horas únicas: {datos['Solo_hora'].nunique()}")
        
        print("\n🔍 Primeras filas con características:")
        columnas_mostrar = ['Timestamp','Animal','Numero', 'Diferencia_Ciclica', 'Posicion_Previo']
        columnas_disponibles = [col for col in columnas_mostrar if col in datos.columns]
        print(datos[columnas_disponibles].head())

        # Iniciar el menú
        main_menu(datos, datosLotto) 

    except FileNotFoundError:
        print("❌ Archivo no encontrado. Creando archivo de ejemplo...")
        datos = pd.DataFrame(columns=['Fecha', 'Hora', 'Animal', 'Numero'])
        datos.to_excel(datosLotto, index=False)
        print(f"✅ Se creó '{datosLotto}'. Agrega datos y ejecuta nuevamente.")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()