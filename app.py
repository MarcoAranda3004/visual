import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# Importamos la página de administración
from admin import mostrar_admin

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA (¡OBLIGATORIO AL INICIO PARA EL ANCHO COMPLETO!)
# ==============================================================================
st.set_page_config(
    page_title="Dashboard de Predicción de Precios Agrícolas",
    layout="wide",
    initial_sidebar_state="expanded"
)

def mostrar_dashboard():
    st.title("📊 Predicción de Precios de Frutas y Verduras")

    # RUTA DINÁMICA ABSOLUTA: Detecta automáticamente la carpeta del proyecto
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DIR = os.path.join(BASE_DIR, "csv")

    MESES_NUMERO = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
        "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }

    # ==============================================================================
    # 1. LIMPIEZA Y NORMALIZACIÓN DE DATOS (Robusta)
    # ==============================================================================
    def normalizar_columnas_y_fecha(df, es_decreto=False):
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
            else:
                df = df.rename(columns={df.columns[0]: 'fecha'})
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                
        return df

    # ==============================================================================
    # SIDEBAR / FILTROS Y CONTROLES DE ESCENARIO
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
                    return sorted(df[col_prod[0]].astype(str).str.strip().str.upper().dropna().unique().tolist())
            except:
                pass
        return ["ARÁNDANO"]

    productos = obtener_productos_disponibles()
    producto_seleccionado = st.sidebar.selectbox("Selecciona la Fruta/Verdura:", productos)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Filtros de Línea de Tiempo")

    ano_inicio_historial = st.sidebar.selectbox("Ver datos históricos desde:", [2023, 2024, 2025], index=0)

    st.sidebar.subheader("🔮 Destino de la Predicción")
    mes_futuro_nombre = st.sidebar.selectbox("Mes a predecir:", list(MESES_NUMERO.keys()))
    ano_futuro = st.sidebar.selectbox("Año a predecir:", [2026, 2027])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Simulación de Factores")
    val_sim_diesel = st.sidebar.slider("Precio proyectado Diésel (S/.)", min_value=10.0, max_value=30.0, value=17.5, step=0.1)

    val_sim_emergencias = 0

    # ==============================================================================
    # FUNCIÓN AUXILIAR: DETERMINAR ESTACIONES EN PERÚ
    # ==============================================================================
    def obtener_color_estacion_peru(fecha):
        mes = fecha.month
        dia = fecha.day
        
        if (mes == 12 and dia >= 21) or (mes in [1, 2]) or (mes == 3 and dia <= 20):
            return "rgba(255, 99, 71, 0.15)"   
        elif (mes == 3 and dia >= 21) or (mes in [4, 5]) or (mes == 6 and dia <= 21):
            return "rgba(218, 165, 32, 0.15)"  
        elif (mes == 6 and dia >= 22) or (mes in [7, 8]) or (mes == 9 and dia <= 22):
            return "rgba(30, 144, 255, 0.15)"  
        else:
            return "rgba(46, 139, 87, 0.15)"   

    # ==============================================================================
    # PROCESAMIENTO Y MODELADO
    # ==============================================================================
    @st.cache_resource(show_spinner=False)
    def entrenar_y_predecir_estacional(producto, ano_historial, mes_nombre, ano_dest, sim_diesel, sim_emergencias):
        try:
            df_prod = pd.read_csv(os.path.join(CSV_DIR, "dataset_productos_anual.csv"))
            df_diesel = pd.read_csv(os.path.join(CSV_DIR, "precios_diesel_anual.csv"))
            df_decre = pd.read_csv(os.path.join(CSV_DIR, "decretos_fluviales_anuales.csv"))
            
            df_prod = normalizar_columnas_y_fecha(df_prod, es_decreto=False)
            df_diesel = normalizar_columnas_y_fecha(df_diesel, es_decreto=False)
            df_decre = normalizar_columnas_y_fecha(df_decre, es_decreto=True)
            
            precio_col_name = [c for c in df_prod.columns if 'precio' in c or 'valor' in c or 'monto' in c][0]
            col_prod_name = [c for c in df_prod.columns if 'producto' in c or 'item' in c][0]
            
            df_prod[col_prod_name] = df_prod[col_prod_name].astype(str).str.strip().str.upper()
            df_prod = df_prod[df_prod[col_prod_name] == str(producto).strip().upper()].copy()
            
            if df_prod.empty:
                return None, None, None, None, f"No hay datos para: {producto}"
                
            df_prod = df_prod.dropna(subset=['fecha'])
            df_prod = df_prod[df_prod[precio_col_name] > 0]
            
            col_diesel_precio = [c for c in df_diesel.columns if 'diesel' in c or 'precio' in c or 'soles' in c]
            diesel_col_name = col_diesel_precio[0] if col_diesel_precio else df_diesel.columns[1]

            min_date, max_date = df_prod['fecha'].min(), df_prod['fecha'].max()
            df_base = pd.DataFrame({'fecha': pd.date_range(start=min_date, end=max_date, freq='D')})
            
            df_merge = pd.merge(df_base, df_prod[['fecha', precio_col_name]], on='fecha', how='left')
            df_merge = pd.merge(df_merge, df_diesel[['fecha', diesel_col_name]], on='fecha', how='left')
            
            emergencias = []
            for d in df_merge['fecha']:
                activas = df_decre[(df_decre['fecha_inicio'] <= d) & (df_decre['fecha_fin'] >= d)].shape[0]
                emergencias.append(activas)
            df_merge['emergencias_activas'] = emergencias
            
            df_merge = df_merge.rename(columns={precio_col_name: 'precio_hoy_kg', diesel_col_name: 'diesel_lima_soles'})
            df_merge['precio_hoy_kg'] = df_merge['precio_hoy_kg'].interpolate(method='linear').ffill().bfill()
            df_merge['diesel_lima_soles'] = df_merge['diesel_lima_soles'].interpolate(method='linear').ffill().bfill()
            
            diesel_medio_hist = df_merge['diesel_lima_soles'].dropna().mean()
            if pd.isna(diesel_medio_hist) or diesel_medio_hist == 0:
                diesel_medio_hist = 17.5  
            
            df_merge['dia_semana'] = df_merge['fecha'].dt.dayofweek
            df_merge['mes'] = df_merge['fecha'].dt.month
            df_merge['dia_ano'] = df_merge['fecha'].dt.dayofyear
            df_merge['dia_semana_seno'] = np.sin(2 * np.pi * df_merge['dia_semana'] / 7)
            df_merge['dia_semana_coseno'] = np.cos(2 * np.pi * df_merge['dia_semana'] / 7)
            df_merge['ano_seno'] = np.sin(2 * np.pi * df_merge['dia_ano'] / 365.25)
            df_merge['ano_coseno'] = np.cos(2 * np.pi * df_merge['dia_ano'] / 365.25)
            
            precio_medio_por_mes = df_merge.groupby('mes')['precio_hoy_kg'].mean().to_dict()
            tendencia_general_media = df_merge['precio_hoy_kg'].mean()
            
            features = ['precio_hoy_kg', 'diesel_lima_soles', 'emergencias_activas', 
                        'dia_semana_seno', 'dia_semana_coseno', 'ano_seno', 'ano_coseno']
            data = df_merge[features].values
            
            look_back = 7
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:(i + look_back)].flatten())
                y.append(data[i + look_back, 0])
            
            model = RandomForestRegressor(n_estimators=75, random_state=42)
            model.fit(np.array(X), np.array(y))
            
            num_mes = MESES_NUMERO[mes_nombre]
            fecha_destino_ia = pd.Timestamp(year=ano_dest, month=num_mes, day=1)
            ultima_fecha_real = df_merge['fecha'].max()
            
            dias_a_predecir = (fecha_destino_ia - ultima_fecha_real).days
            if dias_a_predecir <= 0: dias_a_predecir = 30
                
            secuencia_actual = list(data[-look_back:])
            precios_proyectados, fechas_proyectadas = [], []
            fecha_corriente = ultima_fecha_real
            
            for _ in range(dias_a_predecir):
                fecha_corriente += pd.Timedelta(days=1)
                bloque_input = np.array(secuencia_actual[-look_back:]).flatten().reshape(1, -1)
                pred_base = model.predict(bloque_input)[0]
                
                mes_actual_sim = fecha_corriente.month
                efecto_estacional_mes = precio_medio_por_mes.get(mes_actual_sim, tendencia_general_media)
                desvio_estacional = efecto_estacional_mes - tendencia_general_media
                
                desvio_diesel = (sim_diesel - diesel_medio_hist) / diesel_medio_hist
                impacto_escenario = (desvio_diesel * 0.20) + (sim_emergencias * 0.05)
                
                pred_base_ajustada = pred_base * (1 + impacto_escenario)
                pred_final = (pred_base_ajustada * 0.60) + ((tendencia_general_media + desvio_estacional) * 0.40)
                
                ds = fecha_corriente.dayofweek
                nuevo_registro = [pred_final, sim_diesel, sim_emergencias, 
                                  np.sin(2*np.pi*ds/7), np.cos(2*np.pi*ds/7),
                                  np.sin(2*np.pi*fecha_corriente.dayofyear/365.25), np.cos(2*np.pi*fecha_corriente.dayofyear/365.25)]
                secuencia_actual.append(nuevo_registro)
                precios_proyectados.append(pred_final)
                fechas_proyectadas.append(fecha_corriente)
                
            df_proyeccion_futura = pd.DataFrame({'fecha': fechas_proyectadas, 'precio_ia': precios_proyectados})
            df_filtrado_inicio = df_merge[df_merge['fecha'].dt.year >= ano_historial].copy()
            
            return df_filtrado_inicio, precios_proyectados[-1], mes_nombre, df_proyeccion_futura, None
        except Exception as e:
            return None, None, None, None, f"Detalle técnico: {str(e)}"

    # ==============================================================================
    # INTERFAZ Y GRÁFICO AVANZADO (PLOTLY)
    # ==============================================================================
    with st.spinner("Procesando datos bajo normativas y estacionalidad de Perú..."):
        df_resultado, prediccion, mes_destino, df_futuro, error = entrenar_y_predecir_estacional(
            producto_seleccionado, ano_inicio_historial, mes_futuro_nombre, ano_futuro, val_sim_diesel, val_sim_emergencias
        )

    if error:
        st.error(f"⚠️ Error: {error}")
    else:
        precio_actual = df_resultado['precio_hoy_kg'].iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric(f"Último Real ({producto_seleccionado})", f"S/. {precio_actual:.2f}")
        col2.metric(f"IA Escenario Proyectado", f"S/. {prediccion:.2f}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_resultado['fecha'], y=df_resultado['precio_hoy_kg'],
            mode='lines', name='Precio Histórico Real',
            line=dict(color='#2b5c8f', width=2.5)
        ))

        fechas_ia = [df_resultado['fecha'].iloc[-1]] + list(df_futuro['fecha'])
        precios_ia = [df_resultado['precio_hoy_kg'].iloc[-1]] + list(df_futuro['precio_ia'])
        
        fig.add_trace(go.Scatter(
            x=fechas_ia, y=precios_ia,
            mode='lines', name='Proyección IA Simulada',
            line=dict(color='#e06666', width=3, dash='dash')
        ))

        umbral_precio_alto = df_resultado['precio_hoy_kg'].quantile(0.85)
        df_alertas = df_resultado[
            (df_resultado['precio_hoy_kg'] >= umbral_precio_alto) & 
            ((df_resultado['diesel_lima_soles'] > df_resultado['diesel_lima_soles'].median()) | 
             (df_resultado['emergencias_activas'] > 0))
        ]

        fig.add_trace(go.Scatter(
            x=df_alertas['fecha'], y=df_alertas['precio_hoy_kg'],
            mode='markers', name='Picos Históricos por Diésel/Emergencias',
            marker=dict(color='red', size=8, symbol='triangle-up'),
            hovertemplate="<b>Alerta de Factor</b><br>Precio: S/. %{y:.2f}<br>Fecha: %{x}<extra></extra>"
        ))

        todas_las_fechas = pd.concat([df_resultado['fecha'], df_futuro['fecha']])
        rango_meses = pd.date_range(start=todas_las_fechas.min(), end=todas_las_fechas.max(), freq='MS')
        
        for i in range(len(rango_meses)-1):
            fecha_bucle = rango_meses[i]
            color_fondo = obtener_color_estacion_peru(fecha_bucle)
            fig.add_vrect(
                x0=rango_meses[i], x1=rango_meses[i+1],
                fillcolor=color_fondo, opacity=1,
                layer="below",
                line_width=0
            )

        fig.update_layout(
            title=f"Análisis Histórico Integrado (Desde {ano_inicio_historial}) y Proyección de {producto_seleccionado}",
            xaxis_title="Línea de Tiempo",
            yaxis_title="Precio por Unidad de Medida (S/.)", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            """
            <div style="display: flex; gap: 15px; font-size: 13px; margin-top: 5px; align-items: center;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 12px; height: 12px; background-color: rgba(255, 99, 71, 0.4); border: 1px solid tomato;"></div> Verano
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 12px; height: 12px; background-color: rgba(218, 165, 32, 0.4); border: 1px solid goldenrod;"></div> Otoño
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 12px; height: 12px; background-color: rgba(30, 144, 255, 0.4); border: 1px solid dodgerblue;"></div> Invierno
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 12px; height: 12px; background-color: rgba(46, 139, 87, 0.4); border: 1px solid seagreen;"></div> Primavera
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ==============================================================================
# ENRUTADOR DE NAVEGACIÓN (Controla las pestañas de la app)
# ==============================================================================
pg = st.navigation([
    st.Page(mostrar_dashboard, title="📊 Dashboard Principal", url_path="dashboard"),
    st.Page(mostrar_admin, title="🔐 Acceso Administrativo", url_path="admin")
])
pg.run()