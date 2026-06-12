import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA (Optimizado para pantallas grandes/Google)
# ==============================================================================
st.set_page_config(
    page_title="Dashboard de Predicción de Precios Agrícolas",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Predicción de Precios de Frutas")

# Rutas especificadas por el usuario
CSV_DIR = r"E:\visual\csv"

# Diccionarios de meses para control cronológico y traducción
MESES_ESPANOL = {
    "January": "Enero", "February": "Febrero", "March": "Marzo", "April": "Abril",
    "May": "Mayo", "June": "Junio", "July": "Julio", "August": "Agosto",
    "September": "Septiembre", "October": "Octubre", "November": "Noviembre", "December": "Diciembre"
}

MESES_NUMERO = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# ==============================================================================
# FUNCIÓN AUXILIAR PARA CORREGIR COLUMNAS DE FECHA
# ==============================================================================
def normalizar_columnas_y_fecha(df, es_decreto=False):
    """Limpia los nombres de las columnas y asegura el formato de fecha"""
    df.columns = df.columns.str.strip().str.lower()
    
    if es_decreto:
        col_inicio = [c for c in df.columns if 'inicio' in c or 'desde' in c]
        col_fin = [c for c in df.columns if 'fin' in c or 'hasta' in c]
        
        if col_inicio and col_fin:
            df = df.rename(columns={col_inicio[0]: 'fecha_inicio', col_fin[0]: 'fecha_fin'})
            df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
            df['fecha_fin'] = pd.to_datetime(df['fecha_fin'], errors='coerce')
    else:
        col_fecha = [c for c in df.columns if 'fech' in c or 'date' in c or c == 'f']
        if col_fecha:
            df = df.rename(columns={col_fecha[0]: 'fecha'})
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        else:
            df = df.rename(columns={df.columns[0]: 'fecha'})
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            
    return df

# ==============================================================================
# SIDERBAR / FILTROS (Entradas de usuario)
# ==============================================================================
st.sidebar.header("🎛️ Filtros de Selección")

@st.cache_data
def obtener_productos_disponibles():
    ruta_productos = os.path.join(CSV_DIR, "dataset_productos_anual.csv")
    if os.path.exists(ruta_productos):
        try:
            df = pd.read_csv(ruta_productos)
            df.columns = df.columns.str.strip().str.lower()
            col_prod = [c for c in df.columns if 'producto' in c or 'item' in c]
            if col_prod:
                return df[col_prod[0]].dropna().unique().tolist()
        except:
            pass
    return ["ARANDANO"]

productos = obtener_productos_disponibles()
producto_seleccionado = st.sidebar.selectbox("Selecciona la Fruta/Verdura:", productos)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Filtros de Línea de Tiempo")

# Historial sin opción 2026 para evitar colapsos
ano_inicio_historial = st.sidebar.selectbox("Ver datos históricos desde:", [2023, 2024, 2025], index=1)

# Filtro dinámico para elegir el destino de predicción
st.sidebar.subheader("🔮 Destino de la Predicción")
mes_futuro_nombre = st.sidebar.selectbox("Mes a predecir:", list(MESES_NUMERO.keys()))
ano_futuro = st.sidebar.selectbox("Año a predecir:", [2026, 2027])

# ==============================================================================
# PROCESAMIENTO Y MODELADO CON REPETICIÓN DE PATRÓN ESTACIONAL
# ==============================================================================
@st.cache_resource(show_spinner=False)
def entrenar_y_predecir_estacional(producto, ano_historial, mes_nombre, ano_dest):
    try:
        # 1. Cargar archivos CSV
        df_prod = pd.read_csv(os.path.join(CSV_DIR, "dataset_productos_anual.csv"))
        df_diesel = pd.read_csv(os.path.join(CSV_DIR, "precios_diesel_anual.csv"))
        df_decre = pd.read_csv(os.path.join(CSV_DIR, "decretos_fluviales_anuales.csv"))
        
        # 2. Normalizar datasets
        df_prod = normalizar_columnas_y_fecha(df_prod, es_decreto=False)
        df_diesel = normalizar_columnas_y_fecha(df_diesel, es_decreto=False)
        df_decre = normalizar_columnas_y_fecha(df_decre, es_decreto=True)
        
        col_precio = [c for c in df_prod.columns if 'precio' in c or 'valor' in c or 'monto' in c]
        if not col_precio:
            return None, None, None, None, "No se encontró una columna de 'Precio'."
        precio_col_name = col_precio[0]
        col_prod_name = [c for c in df_prod.columns if 'producto' in c or 'item' in c][0]
        
        df_prod = df_prod[df_prod[col_prod_name] == producto].copy()
        if df_prod.empty:
            return None, None, None, None, f"No se encontraron filas para: {producto}"
            
        df_prod = df_prod.dropna(subset=['fecha'])
        df_prod = df_prod[df_prod[precio_col_name] > 0]
        
        col_diesel_precio = [c for c in df_diesel.columns if 'diesel' in c or 'precio' in c or 'soles' in c]
        diesel_col_name = col_diesel_precio[0] if col_diesel_precio else df_diesel.columns[1]
        df_diesel = df_diesel.dropna(subset=['fecha'])

        min_date = df_prod['fecha'].min()
        max_date = df_prod['fecha'].max()
        rango_fechas = pd.date_range(start=min_date, end=max_date, freq='D')
        df_base = pd.DataFrame({'fecha': rango_fechas})
        
        df_merge = pd.merge(df_base, df_prod[['fecha', precio_col_name]], on='fecha', how='left')
        df_merge = pd.merge(df_merge, df_diesel[['fecha', diesel_col_name]], on='fecha', how='left')
        
        emergencias = []
        for d in df_merge['fecha']:
            activas = df_decre[(df_decre['fecha_inicio'] <= d) & (df_decre['fecha_fin'] >= d)].shape[0]
            emergencias.append(activas)
        df_merge['emergencias_activas'] = emergencias
        
        df_merge = df_merge.rename(columns={precio_col_name: 'precio_hoy_kg', diesel_col_name: 'diesel_lima_soles'})
        df_merge['precio_hoy_kg'] = df_merge['precio_hoy_kg'].interpolate(method='linear').ffill().bfill()
        df_merge['diesel_lima_soles'] = df_merge['diesel_lima_soles'].ffill().bfill()
        
        # --- EXTRACCIÓN DE ENTRADAS CÍCLICAS AVANZADAS (Para evitar el aplanamiento) ---
        df_merge['dia_semana'] = df_merge['fecha'].dt.dayofweek
        df_merge['mes'] = df_merge['fecha'].dt.month
        df_merge['dia_ano'] = df_merge['fecha'].dt.dayofyear
        
        # Componentes cíclicos semanales
        df_merge['dia_semana_seno'] = np.sin(2 * np.pi * df_merge['dia_semana'] / 7)
        df_merge['dia_semana_coseno'] = np.cos(2 * np.pi * df_merge['dia_semana'] / 7)
        
        # Componentes cíclicos anuales (Clave fundamental para capturar campañas de siembra/cosecha)
        df_merge['ano_seno'] = np.sin(2 * np.pi * df_merge['dia_ano'] / 365.25)
        df_merge['ano_coseno'] = np.cos(2 * np.pi * df_merge['dia_ano'] / 365.25)
        
        # Diccionario histórico para calcular desvíos de precios históricos (Patrón repetitivo espejo)
        precio_medio_por_mes = df_merge.groupby('mes')['precio_hoy_kg'].mean().to_dict()
        tendencia_general_media = df_merge['precio_hoy_kg'].mean()
        
        # Definir matriz de Features incluyendo la estacionalidad macro del año
        features = ['precio_hoy_kg', 'diesel_lima_soles', 'emergencias_activas', 
                    'dia_semana_seno', 'dia_semana_coseno', 'ano_seno', 'ano_coseno']
        data = df_merge[features].values
        
        look_back = 7
        if len(data) <= look_back:
            return None, None, None, None, "Datos insuficientes en los ficheros CSV."
            
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)].flatten())
            y.append(data[i + look_back, 0])
            
        X, y = np.array(X), np.array(y)
        
        model = RandomForestRegressor(n_estimators=75, random_state=42)
        model.fit(X, y)
        
        # --- PROYECCIÓN ITERATIVA ASISTIDA POR ESQUEMA ESTACIONAL ---
        num_mes = MESES_NUMERO[mes_nombre]
        fecha_destino_ia = pd.Timestamp(year=ano_dest, month=num_mes, day=1)
        ultima_fecha_real = df_merge['fecha'].max()
        
        dias_a_predecir = (fecha_destino_ia - ultima_fecha_real).days
        if dias_a_predecir <= 0:
            dias_a_predecir = 30
            fecha_destino_ia = ultima_fecha_real + pd.Timedelta(days=30)
            
        secuencia_actual = list(data[-look_back:])
        precios_proyectados = []
        fechas_proyectadas = []
        
        fecha_corriente = ultima_fecha_real
        ultimo_diesel = df_merge['diesel_lima_soles'].iloc[-1]
        ultimas_emergencias = df_merge['emergencias_activas'].iloc[-1]
        
        for _ in range(dias_a_predecir):
            fecha_corriente += pd.Timedelta(days=1)
            
            # Formatear ventana recursiva
            bloque_input = np.array(secuencia_actual[-look_back:]).flatten().reshape(1, -1)
            
            # Predicción base inercial del modelo
            pred_base = model.predict(bloque_input)[0]
            
            # --- MODULACIÓN DEL PATRÓN HISTÓRICO ---
            # Extraemos el comportamiento estacional histórico del mes actual en simulación
            mes_actual_sim = fecha_corriente.month
            efecto_estacional_mes = precio_medio_por_mes.get(mes_actual_sim, tendencia_general_media)
            desvio_estacional = efecto_estacional_mes - tendencia_general_media
            
            # Mezclamos la predicción autorregresiva (60%) con el patrón histórico repetitivo (40%)
            # Esto evita que la predicción deciga en una línea plana y herede el ciclo del producto
            pred_final = (pred_base * 0.60) + ((tendencia_general_media + desvio_estacional) * 0.40)
            
            # Recalcular las variables cíclicas temporales para el paso de mañana
            ds = fecha_corriente.dayofweek
            ds_sen = np.sin(2 * np.pi * ds / 7)
            ds_cos = np.cos(2 * np.pi * ds / 7)
            
            d_ano = fecha_corriente.dayofyear
            ano_sen = np.sin(2 * np.pi * d_ano / 365.25)
            ano_cos = np.cos(2 * np.pi * d_ano / 365.25)
            
            # Insertar de nuevo en el bucle autorregresivo
            nuevo_registro = [pred_final, ultimo_diesel, ultimas_emergencias, ds_sen, ds_cos, ano_sen, ano_cos]
            secuencia_actual.append(nuevo_registro)
            
            precios_proyectados.append(pred_final)
            fechas_proyectadas.append(fecha_corriente)
            
        df_proyeccion_futura = pd.DataFrame({
            'fecha': fechas_proyectadas,
            'Proyección IA': precios_proyectados
        })
        
        df_filtrado_inicio = df_merge[df_merge['fecha'].dt.year >= ano_historial].copy()
        precio_final_predicho = precios_proyectados[-1]
        
        return df_filtrado_inicio, precio_final_predicho, mes_nombre, df_proyeccion_futura, None
    except Exception as e:
        return None, None, None, None, f"Detalle técnico del error: {str(e)}"

# ==============================================================================
# RENDERIZADO DE LA INTERFAZ
# ==============================================================================

with st.spinner(f"Sincronizando patrones anuales y entrenando modelo para {producto_seleccionado}..."):
    df_resultado, prediccion, mes_destino, df_futuro, error = entrenar_y_predecir_estacional(
        producto_seleccionado, ano_inicio_historial, mes_futuro_nombre, ano_futuro
    )

if error:
    st.error(f"⚠️ Error al procesar el archivo: {error}")
else:
    # Métricas principales
    precio_actual = df_resultado['precio_hoy_kg'].iloc[-1]
    variacion = prediccion - precio_actual
    porcentaje_var = (variacion / precio_actual) * 100 if precio_actual > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Último Precio Registrado ({producto_seleccionado})", value=f"S/. {precio_actual:.2f} por Kg")
    with col2:
        st.metric(label=f"Precio Predicho ({mes_destino} {ano_futuro})", value=f"S/. {prediccion:.2f} por Kg", delta=f"{variacion:.2f} Soles ({porcentaje_var:.1f}%)")
    with col3:
        st.metric(label="Alertas/Emergencias Viales Hoy", value=int(df_resultado['emergencias_activas'].iloc[-1]))

    st.markdown("---")

    # Gráficas e Historiales
    col_izq, col_der = st.columns([2, 1])

    with col_izq:
        st.subheader(f"📈 Curva Temporal Histórica y Proyección Estacional Dinámica hacia {mes_destino}")
        
        # 1. Históricos
        df_mostrar = df_resultado[['fecha', 'precio_hoy_kg']].copy()
        df_mostrar = df_mostrar.rename(columns={'precio_hoy_kg': 'Precio Registrado'})
        
        # 2. Conexión continua sin quiebres visuales
        ultima_fecha_real = df_mostrar['fecha'].iloc[-1]
        ultimo_precio_real = df_mostrar['Precio Registrado'].iloc[-1]
        
        df_conexion = pd.DataFrame({'fecha': [ultima_fecha_real], 'Proyección IA': [ultimo_precio_real]})
        df_futuro_completo = pd.concat([df_conexion, df_futuro], ignore_index=True)
        
        # 3. Fusión e Indexación
        df_grafico_final = pd.merge(df_mostrar, df_futuro_completo, on='fecha', how='outer')
        df_grafico_final = df_grafico_final.set_index('fecha')
        
        # 4. Mostrar gráfico interactivo en pantalla con curvas diferenciadas
        st.line_chart(df_grafico_final, use_container_width=True)

    with col_der:
        st.subheader("📋 Factores de Contexto Actuales")
        st.write("Últimos datos analizados de la serie:")
        
        df_contexto = df_resultado[['fecha', 'precio_hoy_kg', 'diesel_lima_soles', 'emergencias_activas']].tail(5)
        df_contexto.columns = ['Fecha', 'Precio (Kg)', 'Precio Diésel', 'Emergencias']
        st.dataframe(df_contexto, use_container_width=True, hide_index=True)

    st.markdown("---")
