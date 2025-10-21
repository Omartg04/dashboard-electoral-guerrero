import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Definir rutas de archivos
csv_path = 'data/secciones.csv'  # Ajusta según la ubicación real
shp_path = 'data/secciones.geojson'  # Ajusta según la ubicación real
sample_path = 'data/secciones_muestra.csv'  # Ajusta según la ubicación real

# Cargar datos usando la función
@st.cache_data
def load_data():
    # Cargar datos
    try:
        df = pd.read_csv(csv_path)
        gdf = gpd.read_file(shp_path)
        df_sample = pd.read_csv(sample_path)
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}. Asegúrate de que los archivos estén en la carpeta 'data'.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

    # Asegurar que la base principal no tenga duplicados
    df.drop_duplicates(subset=['SECCIÓN'], keep='first', inplace=True)
    
    # Asegurar tipos consistentes
    df['SECCIÓN'] = df['SECCIÓN'].astype(str).str.strip()
    gdf['SECCIÓN'] = gdf['SECCIÓN'].astype(str).str.strip()  # Usar SECCIÓN en lugar de SECCION
    df_sample['SECCIÓN'] = df_sample['SECCIÓN'].astype(str).str.strip()
    
    # Reproyectar a EPSG:4326 para Folium
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Merge CSV con shapefile
    merged_gdf = gdf.merge(df, left_on='SECCIÓN', right_on='SECCIÓN', how='left')
    
    # Convertir a GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry', crs="EPSG:4326")
    
    # Crear la columna 'is_sampled'
    secciones_en_muestra_ids = df_sample['SECCIÓN'].unique()
    merged_gdf['is_sampled'] = merged_gdf['SECCIÓN'].isin(secciones_en_muestra_ids)

    # ===== DATOS SIMULADOS PARA MOCKUP =====
    np.random.seed(42)

    # Debugging: Verificar que df, df_sample, y merged_gdf son válidos
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df no es un DataFrame válido.")
    if not isinstance(df_sample, pd.DataFrame):
        raise ValueError("df_sample no es un DataFrame válido.")
    if not isinstance(merged_gdf, gpd.GeoDataFrame):
        raise ValueError("merged_gdf no es un GeoDataFrame válido.")

    if 'SECCIÓN' not in df.columns:
        raise ValueError("La columna 'SECCIÓN' no existe en df.")
    if 'SECCIÓN' not in df_sample.columns:
        raise ValueError("La columna 'SECCIÓN' no existe en df_sample.")
    if 'SECCIÓN' not in merged_gdf.columns:
        raise ValueError("La columna 'SECCIÓN' no existe en merged_gdf.")

    # Verificar duplicados en SECCIÓN
    if df['SECCIÓN'].duplicated().sum() > 0:
        raise ValueError(f"Hay {df['SECCIÓN'].duplicated().sum()} valores duplicados en SECCIÓN en df.")
    if df_sample['SECCIÓN'].duplicated().sum() > 0:
        raise ValueError(f"Hay {df_sample['SECCIÓN'].duplicated().sum()} valores duplicados en SECCIÓN en df_sample.")
    if merged_gdf['SECCIÓN'].duplicated().sum() > 0:
        raise ValueError(f"Hay {merged_gdf['SECCIÓN'].duplicated().sum()} valores duplicados en SECCIÓN en merged_gdf.")

    # Verificar que SECCIÓN en df_sample y merged_gdf es subconjunto de df
    if not df_sample['SECCIÓN'].isin(df['SECCIÓN']).all():
        raise ValueError(f"Algunas SECCIÓN en df_sample no están en df. Matches: {len(set(df_sample['SECCIÓN']).intersection(set(df['SECCIÓN'])))}/{len(df_sample)}")
    if not merged_gdf['SECCIÓN'].isin(df['SECCIÓN']).all():
        st.write("SECCIÓN no coincidentes en merged_gdf:", merged_gdf[~merged_gdf['SECCIÓN'].isin(df['SECCIÓN'])]['SECCIÓN'].tolist())
        raise ValueError(f"Algunas SECCIÓN en merged_gdf no están en df. Matches: {len(set(merged_gdf['SECCIÓN']).intersection(set(df['SECCIÓN'])))}/{len(merged_gdf)}")

    # Simulación para global en df (todas las 765 secciones)
    n_secciones_totales = len(df)
    encuestas_por_seccion_global = 20000 // n_secciones_totales  # ~26 encuestas por sección
    resto_global = 20000 % n_secciones_totales
    df['ENCUESTAS_ASIGNADAS_GLOBAL'] = encuestas_por_seccion_global
    df.iloc[:resto_global, df.columns.get_loc('ENCUESTAS_ASIGNADAS_GLOBAL')] += 1

    estados_captura_global = np.random.choice(
        ['Completada', 'En Proceso', 'Pendiente'], 
        size=n_secciones_totales,
        p=[0.60, 0.25, 0.15]
    )
    df['STATUS_CAPTURA_GLOBAL'] = estados_captura_global

    df['ENCUESTAS_REALIZADAS_GLOBAL'] = df.apply(
        lambda x: x['ENCUESTAS_ASIGNADAS_GLOBAL'] if x['STATUS_CAPTURA_GLOBAL'] == 'Completada' 
        else int(x['ENCUESTAS_ASIGNADAS_GLOBAL'] * np.random.uniform(0.3, 0.9)) if x['STATUS_CAPTURA_GLOBAL'] == 'En Proceso' 
        else 0,
        axis=1
    )

    # Simulación para muestral en df_sample (400 secciones)
    n_secciones = len(df_sample)
    df_sample['ENCUESTAS_ASIGNADAS_MUESTRAL'] = 10  # 10 encuestas por sección, sumando 4,000

    estados_captura = np.random.choice(
        ['Completada', 'En Proceso', 'Pendiente'], 
        size=n_secciones,
        p=[0.60, 0.25, 0.15]
    )
    df_sample['STATUS_CAPTURA'] = estados_captura

    df_sample['ENCUESTAS_REALIZADAS_MUESTRAL'] = df_sample.apply(
        lambda x: x['ENCUESTAS_ASIGNADAS_MUESTRAL'] if x['STATUS_CAPTURA'] == 'Completada' 
        else int(x['ENCUESTAS_ASIGNADAS_MUESTRAL'] * np.random.uniform(0.3, 0.9)) if x['STATUS_CAPTURA'] == 'En Proceso' 
        else 0,
        axis=1
    )

    # Fecha de última actualización
    fecha_base = datetime.now()
    df_sample['ULTIMA_ACTUALIZACION'] = [
        fecha_base - timedelta(days=np.random.randint(0, 8)) for _ in range(n_secciones)
    ]

    # Validación de contactos
    df_sample['PCT_EMAIL_VALIDO'] = np.random.uniform(70, 95, n_secciones).round(1)
    df_sample['PCT_CELULAR_VALIDO'] = np.random.uniform(75, 98, n_secciones).round(1)
    df_sample['EMAILS_VALIDOS'] = (df_sample['ENCUESTAS_REALIZADAS_MUESTRAL'] * df_sample['PCT_EMAIL_VALIDO'] / 100).astype(int)
    df_sample['CELULARES_VALIDOS'] = (df_sample['ENCUESTAS_REALIZADAS_MUESTRAL'] * df_sample['PCT_CELULAR_VALIDO'] / 100).astype(int)

    # Encuestadores
    encuestadores = ['Juan Pérez', 'María González', 'Carlos Ramírez', 'Ana López', 
                     'Luis Martínez', 'Sofia Torres', 'Diego Hernández', 'Laura Sánchez']
    df_sample['ENCUESTADOR'] = np.random.choice(encuestadores, n_secciones)

    # Calidad y tiempo
    df_sample['CALIDAD_DATOS'] = np.random.uniform(60, 100, n_secciones).round(1)
    df_sample['TIEMPO_PROMEDIO_MIN'] = np.random.uniform(8, 25, n_secciones).round(1)

    # Merge global y muestral en merged_gdf
    merged_gdf_temp = merged_gdf.copy()
    merged_gdf_temp = merged_gdf_temp.merge(
        df[['SECCIÓN', 'ENCUESTAS_ASIGNADAS_GLOBAL', 'ENCUESTAS_REALIZADAS_GLOBAL', 'STATUS_CAPTURA_GLOBAL']], 
        on='SECCIÓN', 
        how='left',
        validate='1:1'
    )
    merged_gdf_temp = merged_gdf_temp.merge(
        df_sample[['SECCIÓN', 'STATUS_CAPTURA', 'ENCUESTAS_ASIGNADAS_MUESTRAL', 'ENCUESTAS_REALIZADAS_MUESTRAL', 'ENCUESTADOR']], 
        on='SECCIÓN', 
        how='left',
        validate='1:1'
    )

    # Debugging: Verificar columnas después del merge
    if 'ENCUESTAS_REALIZADAS_GLOBAL' not in merged_gdf_temp.columns:
        st.write("Columnas en df:", df.columns.tolist())
        st.write("Columnas en merged_gdf antes del merge:", merged_gdf.columns.tolist())
        st.write("SECCIÓN matches (df vs merged_gdf):", len(set(df['SECCIÓN']).intersection(set(merged_gdf['SECCIÓN']))), "/", len(merged_gdf))
        raise ValueError(f"Failed to add 'ENCUESTAS_REALIZADAS_GLOBAL' to merged_gdf. SECCIÓN matches: {len(set(df['SECCIÓN']).intersection(set(merged_gdf['SECCIÓN'])))}/{len(merged_gdf)}")
    if 'ENCUESTAS_ASIGNADAS_MUESTRAL' not in merged_gdf_temp.columns:
        raise ValueError(f"Failed to add 'ENCUESTAS_ASIGNADAS_MUESTRAL' to merged_gdf. SECCIÓN matches: {len(set(df_sample['SECCIÓN']).intersection(set(merged_gdf['SECCIÓN'])))}/{len(merged_gdf)}")

    return df, df_sample, merged_gdf_temp

# Cargar los datos
try:
    df, df_sample, merged_gdf = load_data()
except Exception as e:
    st.error(f"Error en load_data: {str(e)}")
    st.stop()

# Definir filtros después de cargar los datos
distritos_unicos = ['Todos'] + sorted(df['Distrito'].unique())
municipios_unicos = ['Todos'] + sorted(df['MUNICIPIOS'].unique())
status_options = ['Completada', 'En Proceso', 'Pendiente']

# Filtros en la barra lateral
st.sidebar.header("Filtros")
selected_distrito = st.sidebar.selectbox("Distrito", distritos_unicos)
selected_municipio = st.sidebar.selectbox("Municipio", municipios_unicos)
status_filter = st.sidebar.multiselect("Estado de Captura", status_options, default=status_options)
show_sampled = st.sidebar.checkbox("Mostrar solo secciones muestreadas", value=False)

# ===== FILTROS =====
filtered_gdf = merged_gdf.copy()
if selected_distrito != 'Todos':
    filtered_gdf = filtered_gdf[filtered_gdf['Distrito'] == selected_distrito]
if selected_municipio != 'Todos':
    filtered_gdf = filtered_gdf[filtered_gdf['MUNICIPIOS'] == selected_municipio]

filtered_sample = df_sample.copy()
if selected_distrito != 'Todos':
    filtered_sample = filtered_sample[filtered_sample['Distrito'] == selected_distrito]
if selected_municipio != 'Todos':
    filtered_sample = filtered_sample[filtered_sample['MUNICIPIOS'] == selected_municipio]
if status_filter:
    filtered_sample = filtered_sample[filtered_sample['STATUS_CAPTURA'].isin(status_filter)]

# Verificación de columnas para depuración
if 'ENCUESTAS_ASIGNADAS_MUESTRAL' not in filtered_sample.columns:
    st.error("Error: La columna 'ENCUESTAS_ASIGNADAS_MUESTRAL' no existe en filtered_sample. Revisa la función load_data.")
    st.stop()
if 'ENCUESTAS_REALIZADAS_MUESTRAL' not in filtered_sample.columns:
    st.error("Error: La columna 'ENCUESTAS_REALIZADAS_MUESTRAL' no existe en filtered_sample. Revisa la función load_data.")
    st.stop()
if 'ENCUESTAS_REALIZADAS_GLOBAL' not in filtered_gdf.columns:
    st.error("Error: La columna 'ENCUESTAS_REALIZADAS_GLOBAL' no existe en filtered_gdf. Revisa la función load_data.")
    st.stop()

# Calcular métricas globales y muestrales
total_encuestas_asignadas_muestral = filtered_sample['ENCUESTAS_ASIGNADAS_MUESTRAL'].sum()
total_encuestas_realizadas_muestral = filtered_sample['ENCUESTAS_REALIZADAS_MUESTRAL'].sum()
progreso_captura_muestral = (total_encuestas_realizadas_muestral / total_encuestas_asignadas_muestral * 100) if total_encuestas_asignadas_muestral > 0 else 0
avance_muestral = (total_encuestas_realizadas_muestral / 4000 * 100) if 4000 > 0 else 0

total_encuestas_realizadas_global = filtered_gdf['ENCUESTAS_REALIZADAS_GLOBAL'].sum()
avance_global = (total_encuestas_realizadas_global / 20000 * 100) if 20000 > 0 else 0

# Mantener compatibilidad con el resto del código
total_encuestas_realizadas = total_encuestas_realizadas_muestral
progreso_captura = progreso_captura_muestral
tasa_muestreo = (len(df_sample) / len(df) * 100) if len(df) > 0 else 0

# NUEVAS MÉTRICAS DE PROGRESO
total_encuestas_asignadas = filtered_sample['ENCUESTAS_ASIGNADAS'].sum()
total_encuestas_realizadas = filtered_sample['ENCUESTAS_REALIZADAS'].sum()
progreso_captura = (total_encuestas_realizadas / total_encuestas_asignadas * 100) if total_encuestas_asignadas > 0 else 0

# Título principal
st.title("📊 Dashboard Electoral - Guerrero")
st.markdown(f"**Vista actual:** {selected_distrito} {'→ ' + selected_municipio if selected_municipio != 'Todos' else ''}")

# DISCLAIMER MOCKUP
st.info("ℹ️ **Dashboard demostrativo con datos simulados.** Los datos reales se integrarán desde el portal de captura en producción.")

# Sistema de Tabs - ACTUALIZADO
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Instrucciones", 
    "📈 Resumen Ejecutivo", 
    "🗺️ Mapa Interactivo", 
    "📊 Análisis de Cobertura",
    "📱 Validación de Contactos",
    "👥 Control de Calidad",
    "💾 Datos y Descarga"
])

# ==================== TAB 1: INSTRUCCIONES (ACTUALIZADO) ====================
with tab1:
    st.header("📋 Instrucciones de Uso")
    
    # NUEVO: Timeline del proyecto
    st.subheader("🎯 Flujo del Proyecto Electoral")
    
    # Crear visualización más clara del timeline
    fases = [
        {"fase": "Diseño de Muestra", "num": "1", "status": "Completado", "color": "#28a745", "icon": "✅"},
        {"fase": "Captura en Campo", "num": "2", "status": "En Progreso", "color": "#ffc107", "icon": "🔄"},
        {"fase": "Base de Datos", "num": "3", "status": "En Progreso", "color": "#ffc107", "icon": "🔄"},
        {"fase": "Validación Contactos", "num": "4", "status": "En Progreso", "color": "#ffc107", "icon": "🔄"},
        {"fase": "Auditoría Calidad", "num": "5", "status": "Pendiente", "color": "#6c757d", "icon": "⏳"},
        {"fase": "Perfilamiento", "num": "6", "status": "Pendiente", "color": "#6c757d", "icon": "⏳"}
    ]
    
    # Crear gráfica de timeline más visual
    timeline_fig = go.Figure()
    
    # Agregar línea conectora
    timeline_fig.add_trace(go.Scatter(
        x=list(range(len(fases))),
        y=[0.5] * len(fases),
        mode='lines',
        line=dict(color='lightgray', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Agregar puntos de cada fase
    for i, fase_info in enumerate(fases):
        timeline_fig.add_trace(go.Scatter(
            x=[i], y=[0.5],
            mode='markers+text',
            marker=dict(size=40, color=fase_info['color'], line=dict(width=2, color='white')),
            text=fase_info['num'],
            textposition="middle center",
            textfont=dict(size=16, color='white', family='Arial Black'),
            name=f"{fase_info['icon']} {fase_info['num']}. {fase_info['fase']}",
            hovertemplate=f"<b>Fase {fase_info['num']}: {fase_info['fase']}</b><br>" +
                         f"Estado: {fase_info['status']}<br>" +
                         f"{fase_info['icon']}<extra></extra>",
            showlegend=True
        ))
    
    timeline_fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11)
        ),
        height=250,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, len(fases)-0.5]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=200, t=20, b=20),
        hovermode='closest'
    )
    
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Leyenda más clara
    col_ley1, col_ley2, col_ley3 = st.columns(3)
    with col_ley1:
        st.markdown("✅ **Completado:** Fase finalizada")
    with col_ley2:
        st.markdown("🔄 **En Progreso:** Actualmente trabajando")
    with col_ley3:
        st.markdown("⏳ **Pendiente:** Por iniciar")
    
    st.markdown("---")
    
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        st.subheader("🎯 Objetivo del Dashboard")
        st.markdown("""
        Este dashboard permite visualizar y analizar:
        - **Distribución electoral** en el estado de Guerrero
        - **Plan de muestreo** de 400 secciones para encuestas
        - **Progreso en tiempo real** de la captura de datos
        - **Validación de contactos** (emails y celulares)
        - **Control de calidad** por encuestador
        - **Cobertura geográfica y poblacional** de la muestra
        """)
        
        st.subheader("🔍 Cómo usar los filtros")
        st.markdown("""
        **Barra lateral izquierda (Filtros):**
        
        1. **Distrito:** Selecciona un distrito específico o "Todos" para ver todo el estado
           - Ejemplo: Si seleccionas "Distrito 1", solo verás datos de ese distrito
           
        2. **Municipio:** Filtra por municipio específico
           - Nota: Los municipios disponibles cambian según el distrito seleccionado
           - Si Distrito = "Todos", verás todos los municipios
           
        3. **Estado de Captura:** Filtra las secciones según su progreso
           - Completada: Secciones donde se terminó el trabajo de campo
           - En Proceso: Secciones con encuestas parciales
           - Pendiente: Secciones aún no iniciadas
           - Puedes seleccionar múltiples estados
        
        4. **Mostrar solo secciones en muestra:** 
           - Activado: Muestra únicamente las 400 secciones seleccionadas para encuestar
           - Desactivado: Muestra todas las secciones del estado
        
        Los filtros se aplican automáticamente a todas las pestañas del dashboard.
        """)
        
        # Ejemplo visual de cómo funcionan los filtros
        with st.expander("Ver ejemplo de uso de filtros"):
            st.markdown("""
            **Caso de uso 1: Quiero ver el progreso solo del Distrito 5**
            - Selecciona: Distrito = "5"
            - Municipio = "Todos"
            - Estado = [Todos seleccionados]
            
            **Caso de uso 2: Quiero ver qué secciones están pendientes en Acapulco**
            - Distrito = "Todos" o el correspondiente
            - Municipio = "Acapulco de Juárez"
            - Estado = Solo "Pendiente"
            
            **Caso de uso 3: Ver todas las secciones completadas de la muestra**
            - Distrito = "Todos"
            - Municipio = "Todos"  
            - Estado = Solo "Completada"
            - Activar "Mostrar solo secciones en muestra"
            """)
        
    
    with col_inst2:
        st.subheader("🔑 Navegación por pestañas")
        st.markdown("""
        **📈 Resumen Ejecutivo**
        - Métricas clave del proyecto
        - Progreso de captura en tiempo real
        - Indicadores de cobertura
        
        **🗺️ Mapa Interactivo**
        - Visualización geográfica de secciones
        - Capas intercambiables
        - Estado de captura por color
        
        **📊 Análisis de Cobertura**
        - Gráficas de distribución de muestra
        - Cobertura por municipio
        - Validación de asignación de encuestas
        
        **📱 Validación de Contactos (NUEVO)**
        - Calidad de emails y celulares
        - Tasas de verificación por sección
        - Directorio confiable
        
        **👥 Control de Calidad (NUEVO)**
        - Desempeño de encuestadores
        - Métricas de productividad
        - Alertas de bajo rendimiento
        
        **💾 Datos y Descarga**
        - Tabla completa de secciones
        - Exportación a CSV
        """)
    
    st.subheader("📊 Parámetros del Estudio")
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    param_col1.metric("Secciones totales", df['SECCIÓN'].nunique())
    param_col2.metric("Secciones en muestra", df_sample['SECCIÓN'].nunique())
    param_col3.metric("Total de encuestas", int(df_sample['ENCUESTAS_ASIGNADAS'].sum()))
    param_col4.metric("Progreso general", f"{progreso_captura:.1f}%")
    
    st.markdown("**Nivel de confianza:** 95% | **Margen de error:** ±1.55%")

# ===== TAB 2: RESUMEN EJECUTIVO =====
with tab2:
    st.header("📈 Métricas Principales")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Secciones Totales", 765)
    col2.metric("Encuestas Realizadas Global", int(total_encuestas_realizadas_global), 
                delta=f"{avance_global:.1f}% de 20,000")
    col3.metric("Secciones en Muestra", len(filtered_sample))
    col4.metric("Encuestas Asignadas Muestral", int(total_encuestas_asignadas_muestral))
    col5.metric("Encuestas Realizadas Muestral", int(total_encuestas_realizadas_muestral), 
                delta=f"{progreso_captura_muestral:.1f}%")
    col6.metric("Avance Muestral", f"{int(total_encuestas_realizadas_muestral):,}", 
                delta=f"{avance_muestral:.1f}% de 4,000")
    
    st.markdown("---")
    
    st.subheader("🎯 Indicadores de Progreso en Tiempo Real")
    
    col_prog1, col_prog2, col_prog3 = st.columns(3)
    
    with col_prog1:
        fig_prog = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=progreso_captura_muestral,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Progreso de Captura Muestral (%)"},
            delta={'reference': 100, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig_prog.update_layout(height=250)
        st.plotly_chart(fig_prog, use_container_width=True)
    
    with col_prog2:
        fig_gauge1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=tasa_muestreo,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Cobertura Geográfica (% Secciones)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 15], 'color': "gray"}]}))
        fig_gauge1.update_layout(height=250)
        st.plotly_chart(fig_gauge1, use_container_width=True)
    
    with col_prog3:
        calidad_promedio = filtered_sample['CALIDAD_DATOS'].mean()
        fig_calidad = go.Figure(go.Indicator(
            mode="gauge+number",
            value=calidad_promedio,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calidad de Datos (Score)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 60], 'color': "lightcoral"},
                    {'range': [60, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightgreen"}]}))
        fig_calidad.update_layout(height=250)
        st.plotly_chart(fig_calidad, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Estado de Captura por Distrito")
    
    status_by_distrito = filtered_sample.groupby(['Distrito', 'STATUS_CAPTURA']).size().reset_index(name='count')
    
    fig_status = px.bar(
        status_by_distrito,
        x='Distrito',
        y='count',
        color='STATUS_CAPTURA',
        title="Distribución de Secciones por Estado de Captura",
        color_discrete_map={
            'Completada': 'green',
            'En Proceso': 'orange',
            'Pendiente': 'red'
        },
        barmode='stack'
    )
    st.plotly_chart(fig_status, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔍 Distribución de la Muestra por Distrito")
    sample_distrito = filtered_sample.groupby('Distrito').agg({
        'SECCIÓN': 'count',
        'ENCUESTAS_ASIGNADAS_MUESTRAL': 'sum',
        'ENCUESTAS_REALIZADAS_MUESTRAL': 'sum'
    }).reset_index()
    sample_distrito.columns = ['Distrito', 'Secciones', 'Encuestas Asignadas Muestral', 'Encuestas Realizadas Muestral']
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        fig_distrito = px.bar(
            sample_distrito,
            x='Distrito',
            y='Secciones',
            title="Secciones Muestreadas por Distrito",
            color='Secciones',
            color_continuous_scale='Blues'
        )
        fig_distrito.update_layout(showlegend=False)
        st.plotly_chart(fig_distrito, use_container_width=True)
    
    with col_dist2:
        fig_encuestas = px.bar(
            sample_distrito,
            x='Distrito',
            y=['Encuestas Asignadas Muestral', 'Encuestas Realizadas Muestral'],
            title="Encuestas Muestral: Asignadas vs Realizadas",
            barmode='group'
        )
        st.plotly_chart(fig_encuestas, use_container_width=True)

# ==================== TAB 3: MAPA INTERACTIVO ====================
with tab3:
    st.header("🗺️ Mapa de Secciones Electorales")
    
    col_map1, col_map2 = st.columns([3, 1])
    with col_map2:
        st.markdown("**Controles:**")
        st.markdown("- Usa el control de capas en el mapa")
        st.markdown("- Haz clic en las secciones para ver detalles")
        st.markdown("- Colores indican estado de captura")
    
    map_data = filtered_gdf[filtered_gdf['geometry'].notna()].copy()
    
    if show_sampled:
        map_data = map_data[map_data['is_sampled']]
    
    if map_data.empty:
        st.warning("⚠️ No hay datos geográficos disponibles para los filtros seleccionados.")
    else:
        try:
            bounds = map_data.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=9, 
                tiles="CartoDB positron"
            )
            
            # Capa base con todos los polígonos (sin filtro de estado, para contexto)
            folium.Choropleth(
                geo_data=map_data,
                name="📊 Total Lista Nominal",
                data=map_data,
                columns=['SECCIÓN', 'TOTAL LISTA NOMINAL'],
                key_on='feature.properties.SECCIÓN',
                fill_color='BuGn',
                fill_opacity=0.4,
                line_opacity=0.2,
                legend_name='Total Lista Nominal',
                show=True
            ).add_to(m)
            
            # NUEVO: Capas por estado de captura
            sampled_sections = map_data[map_data['is_sampled']].copy()
            
            if not sampled_sections.empty:
                # Merge con datos de encuestas
                sampled_sections = sampled_sections.merge(
                    df_sample[['SECCIÓN', 'ENCUESTAS_ASIGNADAS', 'STATUS_CAPTURA', 'ENCUESTADOR']], 
                    on='SECCIÓN', 
                    how='left',
                    suffixes=('', '_sample')
                )
                
                # CAMBIO: Aplicar filtro de estado aquí
                if status_filter:
                    sampled_sections = sampled_sections[sampled_sections['STATUS_CAPTURA'].isin(status_filter)]
                
                # Si después del filtro no hay datos, no agregar la capa
                if not sampled_sections.empty:
                    # Función de estilo según estado
                    def style_function(feature):
                        status = feature['properties'].get('STATUS_CAPTURA', 'Pendiente')
                        color_map = {
                            'Completada': '#28a745',
                            'En Proceso': '#ffc107',
                            'Pendiente': '#dc3545'
                        }
                        return {
                            'fillColor': color_map.get(status, '#6c757d'),
                            'fillOpacity': 0.7,
                            'color': 'black',
                            'weight': 2
                        }
                    
                    folium.GeoJson(
                        sampled_sections,
                        name="🎯 Estado de Captura",
                        style_function=style_function,
                        tooltip=folium.GeoJsonTooltip(
                            fields=['SECCIÓN', 'MUNICIPIOS', 'STATUS_CAPTURA', 'ENCUESTADOR', 'ENCUESTAS_ASIGNADAS'],
                            aliases=['Sección', 'Municipio', 'Estado', 'Encuestador', 'Encuestas Asignadas'],
                            localize=True
                        ),
                        show=True
                    ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            sw = [bounds[1], bounds[0]]
            ne = [bounds[3], bounds[2]]
            m.fit_bounds([sw, ne])
            
            st_folium(m, width=1400, height=700, returned_objects=[])
            
            # CAMBIO: Estadísticas del área visible, ahora usando sampled_sections filtrado
            st.markdown("---")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            col_stat1.metric("Secciones visibles", len(map_data))
            col_stat2.metric("Completadas", len(sampled_sections[sampled_sections['STATUS_CAPTURA'] == 'Completada']) if not sampled_sections.empty else 0)
            col_stat3.metric("En Proceso", len(sampled_sections[sampled_sections['STATUS_CAPTURA'] == 'En Proceso']) if not sampled_sections.empty else 0)
            col_stat4.metric("Pendientes", len(sampled_sections[sampled_sections['STATUS_CAPTURA'] == 'Pendiente']) if not sampled_sections.empty else 0)
                
        except Exception as e:
            st.error(f"❌ Error al crear el mapa: {str(e)}")

# ==================== TAB 4: ANÁLISIS DE COBERTURA ====================
with tab4:
    st.header("📊 Análisis de Cobertura de Muestra")
    
    st.subheader("Cobertura de Secciones por Municipio")
    
    munic_stats = filtered_gdf.groupby('MUNICIPIOS').agg({
        'SECCIÓN': 'nunique'
    }).reset_index()
    munic_stats.columns = ['Municipio', 'Total_Secciones']
    
    munic_sample = filtered_gdf[filtered_gdf['SECCIÓN'].isin(df_sample['SECCIÓN'])].groupby('MUNICIPIOS').agg({
        'SECCIÓN': 'nunique'
    }).reset_index()
    munic_sample.columns = ['Municipio', 'Secciones_Muestreadas']
    
    munic_comparison = munic_stats.merge(munic_sample, on='Municipio', how='left')
    munic_comparison['Secciones_Muestreadas'] = munic_comparison['Secciones_Muestreadas'].fillna(0)
    munic_comparison = munic_comparison.sort_values('Total_Secciones', ascending=True)
    
    fig_cobertura = go.Figure()
    fig_cobertura.add_trace(go.Bar(
        y=munic_comparison['Municipio'],
        x=munic_comparison['Total_Secciones'],
        name='Total Secciones',
        orientation='h',
        marker=dict(color='lightblue')
    ))
    fig_cobertura.add_trace(go.Bar(
        y=munic_comparison['Municipio'],
        x=munic_comparison['Secciones_Muestreadas'],
        name='Secciones Muestreadas',
        orientation='h',
        marker=dict(color='darkblue')
    ))
    fig_cobertura.update_layout(
        barmode='overlay',
        title="Secciones Totales vs Muestreadas por Municipio",
        xaxis_title="Número de Secciones",
        yaxis_title="Municipio",
        height=max(400, len(munic_comparison) * 20),
        showlegend=True
    )
    st.plotly_chart(fig_cobertura, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Validación de Asignación de Encuestas")
    
    # Merge correcto evitando duplicados de columnas
    scatter_data = filtered_gdf[filtered_gdf['SECCIÓN'].isin(df_sample['SECCIÓN'])].copy()
    scatter_data = scatter_data.merge(
        df_sample[['SECCIÓN', 'ENCUESTAS_ASIGNADAS', 'ENCUESTAS_REALIZADAS']], 
        on='SECCIÓN', 
        how='left',
        suffixes=('', '_sample')
    )
    
    # Si hay columnas duplicadas, usar las del sample
    if 'ENCUESTAS_ASIGNADAS_sample' in scatter_data.columns:
        scatter_data['ENCUESTAS_ASIGNADAS'] = scatter_data['ENCUESTAS_ASIGNADAS_sample']
        scatter_data['ENCUESTAS_REALIZADAS'] = scatter_data['ENCUESTAS_REALIZADAS_sample']
    
    fig_scatter = px.scatter(
        scatter_data,
        x='TOTAL LISTA NOMINAL',
        y='ENCUESTAS_ASIGNADAS',
        color='Distrito',
        size='ENCUESTAS_REALIZADAS',
        hover_data=['SECCIÓN', 'MUNICIPIOS', 'ENCUESTAS_REALIZADAS'],
        title="Relación entre Tamaño de Sección y Encuestas Asignadas",
        labels={
            'TOTAL LISTA NOMINAL': 'Total Lista Nominal (Tamaño de Sección)',
            'ENCUESTAS_ASIGNADAS': 'Número de Encuestas Asignadas'
        }
    )
    fig_scatter.update_traces(marker=dict(opacity=0.7))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Intensidad de Encuestamiento")
    st.markdown("Encuestas por cada 1,000 habitantes en la lista nominal")
    
    scatter_data['intensidad'] = (scatter_data['ENCUESTAS_ASIGNADAS'] / scatter_data['TOTAL LISTA NOMINAL']) * 1000
    
    fig_intensidad = px.bar(
        scatter_data.sort_values('intensidad', ascending=False).head(20),
        x='SECCIÓN',
        y='intensidad',
        color='Distrito',
        title="Top 20 Secciones por Intensidad de Encuestamiento",
        labels={'intensidad': 'Encuestas por 1,000 habitantes'}
    )
    fig_intensidad.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_intensidad, use_container_width=True)

# ==================== TAB 5: VALIDACIÓN DE CONTACTOS (NUEVO) ====================
with tab5:
    st.header("📱 Validación de Contactos")
    st.markdown("**Fase 4:** Verificación de calidad de emails y celulares para crear directorio confiable")
    
    # Métricas generales
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    
    total_contactos = filtered_sample['ENCUESTAS_REALIZADAS'].sum()
    emails_validos_total = filtered_sample['EMAILS_VALIDOS'].sum()
    celulares_validos_total = filtered_sample['CELULARES_VALIDOS'].sum()
    
    col_v1.metric("Contactos Totales", int(total_contactos))
    col_v2.metric("Emails Válidos", int(emails_validos_total), 
                  delta=f"{(emails_validos_total/total_contactos*100):.1f}%" if total_contactos > 0 else "0%")
    col_v3.metric("Celulares Válidos", int(celulares_validos_total),
                  delta=f"{(celulares_validos_total/total_contactos*100):.1f}%" if total_contactos > 0 else "0%")
    col_v4.metric("Tasa Validación", 
                  f"{filtered_sample['PCT_EMAIL_VALIDO'].mean():.1f}%")
    
    st.markdown("---")
    
    # Gráficas de validación
    st.subheader("📊 Análisis de Calidad de Contactos")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Distribución de calidad de emails
        fig_email = px.histogram(
            filtered_sample,
            x='PCT_EMAIL_VALIDO',
            nbins=20,
            title="Distribución de Calidad de Emails por Sección",
            labels={'PCT_EMAIL_VALIDO': '% Emails Válidos', 'count': 'Número de Secciones'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_email.add_vline(x=filtered_sample['PCT_EMAIL_VALIDO'].mean(), 
                           line_dash="dash", line_color="red",
                           annotation_text="Promedio")
        st.plotly_chart(fig_email, use_container_width=True)
    
    with col_chart2:
        # Distribución de calidad de celulares
        fig_cel = px.histogram(
            filtered_sample,
            x='PCT_CELULAR_VALIDO',
            nbins=20,
            title="Distribución de Calidad de Celulares por Sección",
            labels={'PCT_CELULAR_VALIDO': '% Celulares Válidos', 'count': 'Número de Secciones'},
            color_discrete_sequence=['#2ca02c']
        )
        fig_cel.add_vline(x=filtered_sample['PCT_CELULAR_VALIDO'].mean(), 
                         line_dash="dash", line_color="red",
                         annotation_text="Promedio")
        st.plotly_chart(fig_cel, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla de secciones con baja calidad
    st.subheader("⚠️ Secciones con Baja Calidad de Contactos")
    st.markdown("Secciones que requieren revisión (< 80% validación)")
    
    low_quality = filtered_sample[
        (filtered_sample['PCT_EMAIL_VALIDO'] < 80) | 
        (filtered_sample['PCT_CELULAR_VALIDO'] < 80)
    ][['SECCIÓN', 'MUNICIPIOS', 'ENCUESTADOR', 'PCT_EMAIL_VALIDO', 'PCT_CELULAR_VALIDO', 'STATUS_CAPTURA']].sort_values('PCT_EMAIL_VALIDO')
    
    if len(low_quality) > 0:
        st.dataframe(
            low_quality,
            use_container_width=True,
            height=300
        )
        st.warning(f"⚠️ {len(low_quality)} secciones requieren atención especial")
    else:
        st.success("✅ Todas las secciones tienen calidad de contactos aceptable")
    
    st.markdown("---")
    
    # Comparación por distrito
    st.subheader("📍 Calidad de Contactos por Distrito")
    
    distrito_validation = filtered_sample.groupby('Distrito').agg({
        'PCT_EMAIL_VALIDO': 'mean',
        'PCT_CELULAR_VALIDO': 'mean',
        'EMAILS_VALIDOS': 'sum',
        'CELULARES_VALIDOS': 'sum'
    }).reset_index()
    
    fig_distrito_val = go.Figure()
    fig_distrito_val.add_trace(go.Bar(
        x=distrito_validation['Distrito'],
        y=distrito_validation['PCT_EMAIL_VALIDO'],
        name='% Email Válido',
        marker_color='#1f77b4'
    ))
    fig_distrito_val.add_trace(go.Bar(
        x=distrito_validation['Distrito'],
        y=distrito_validation['PCT_CELULAR_VALIDO'],
        name='% Celular Válido',
        marker_color='#2ca02c'
    ))
    fig_distrito_val.update_layout(
        title="Promedio de Validación por Distrito",
        barmode='group',
        yaxis_title="Porcentaje de Validación (%)"
    )
    st.plotly_chart(fig_distrito_val, use_container_width=True)

# ==================== TAB 6: CONTROL DE CALIDAD (NUEVO) ====================
with tab6:
    st.header("👥 Control de Calidad - Auditoría de Encuestadores")
    st.markdown("**Fase 5:** Análisis de desempeño y calidad de datos por encuestador")
    
    # Métricas generales del equipo
    col_eq1, col_eq2, col_eq3, col_eq4 = st.columns(4)
    
    num_encuestadores = filtered_sample['ENCUESTADOR'].nunique()
    encuestas_por_encuestador = filtered_sample.groupby('ENCUESTADOR')['ENCUESTAS_REALIZADAS'].sum().mean()
    calidad_promedio = filtered_sample['CALIDAD_DATOS'].mean()
    tiempo_promedio = filtered_sample['TIEMPO_PROMEDIO_MIN'].mean()
    
    col_eq1.metric("Encuestadores Activos", num_encuestadores)
    col_eq2.metric("Encuestas/Encuestador", f"{encuestas_por_encuestador:.0f}")
    col_eq3.metric("Calidad Promedio", f"{calidad_promedio:.1f}/100")
    col_eq4.metric("Tiempo Promedio", f"{tiempo_promedio:.1f} min")
    
    st.markdown("---")
    
    # Ranking de encuestadores
    st.subheader("🏆 Ranking de Encuestadores")
    
    encuestador_stats = filtered_sample.groupby('ENCUESTADOR').agg({
        'SECCIÓN': 'count',
        'ENCUESTAS_REALIZADAS': 'sum',
        'CALIDAD_DATOS': 'mean',
        'TIEMPO_PROMEDIO_MIN': 'mean',
        'PCT_EMAIL_VALIDO': 'mean',
        'PCT_CELULAR_VALIDO': 'mean'
    }).reset_index()
    
    encuestador_stats.columns = [
        'Encuestador', 'Secciones Asignadas', 'Encuestas Realizadas', 
        'Calidad Promedio', 'Tiempo Promedio (min)', 
        '% Email Válido', '% Celular Válido'
    ]
    
    # Calcular score compuesto
    encuestador_stats['Score Global'] = (
        encuestador_stats['Calidad Promedio'] * 0.4 +
        encuestador_stats['% Email Válido'] * 0.3 +
        encuestador_stats['% Celular Válido'] * 0.3
    ).round(1)
    
    encuestador_stats = encuestador_stats.sort_values('Score Global', ascending=False)
    
    # Visualización del ranking
    fig_ranking = px.bar(
        encuestador_stats,
        x='Encuestador',
        y='Score Global',
        color='Score Global',
        title="Score de Desempeño por Encuestador",
        color_continuous_scale='RdYlGn',
        text='Score Global'
    )
    fig_ranking.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_ranking.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_ranking, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla detallada
    st.subheader("📋 Detalle de Desempeño por Encuestador")
    
    # Mostrar tabla simple
    st.dataframe(
        encuestador_stats.round(2),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # Análisis de productividad vs calidad
    st.subheader("⚖️ Balance Productividad vs Calidad")
    
    fig_balance = px.scatter(
        encuestador_stats,
        x='Encuestas Realizadas',
        y='Calidad Promedio',
        size='Score Global',
        color='Encuestador',
        title="Relación entre Productividad y Calidad",
        labels={
            'Encuestas Realizadas': 'Productividad (Encuestas Realizadas)',
            'Calidad Promedio': 'Calidad de Datos (Score)'
        },
        hover_data=['Secciones Asignadas', 'Tiempo Promedio (min)']
    )
    fig_balance.update_traces(marker=dict(size=encuestador_stats['Score Global']*0.5, opacity=0.7))
    st.plotly_chart(fig_balance, use_container_width=True)
    
    st.markdown("---")
    
    # Alertas de bajo desempeño
    st.subheader("🚨 Alertas de Desempeño")
    
    col_alert1, col_alert2 = st.columns(2)
    
    with col_alert1:
        bajo_score = encuestador_stats[encuestador_stats['Score Global'] < 75]
        if len(bajo_score) > 0:
            st.warning(f"⚠️ {len(bajo_score)} encuestadores con score bajo (<75)")
            st.dataframe(bajo_score[['Encuestador', 'Score Global', 'Calidad Promedio']], 
                        use_container_width=True)
        else:
            st.success("✅ Todos los encuestadores con desempeño aceptable")
    
    with col_alert2:
        tiempo_alto = encuestador_stats[encuestador_stats['Tiempo Promedio (min)'] > 20]
        if len(tiempo_alto) > 0:
            st.info(f"ℹ️ {len(tiempo_alto)} encuestadores con tiempo alto (>20 min)")
            st.dataframe(tiempo_alto[['Encuestador', 'Tiempo Promedio (min)', 'Encuestas Realizadas']], 
                        use_container_width=True)
        else:
            st.success("✅ Tiempos de encuesta dentro del rango óptimo")

# ==================== TAB 7: DATOS Y DESCARGA ====================
with tab7:
    st.header("💾 Tabla de Secciones Muestreadas")
    
    # Preparar tabla completa con todos los datos
    sample_display = filtered_sample[[
        'SECCIÓN', 'MUNICIPIOS', 'Distrito', 
        'TOTAL PADRÓN', 'TOTAL LISTA NOMINAL',
        'ENCUESTAS_ASIGNADAS', 'ENCUESTAS_REALIZADAS',
        'STATUS_CAPTURA', 'ENCUESTADOR',
        'PCT_EMAIL_VALIDO', 'PCT_CELULAR_VALIDO',
        'CALIDAD_DATOS', 'ULTIMA_ACTUALIZACION'
    ]].copy()
    
    sample_display['ULTIMA_ACTUALIZACION'] = sample_display['ULTIMA_ACTUALIZACION'].dt.strftime('%Y-%m-%d')
    
    # Opciones de visualización
    col_table1, col_table2, col_table3 = st.columns(3)
    with col_table1:
        search_seccion = st.text_input("🔍 Buscar Sección", "")
    with col_table2:
        sort_by = st.selectbox("Ordenar por", [
            'SECCIÓN', 'MUNICIPIOS', 'Distrito', 
            'ENCUESTAS_REALIZADAS', 'CALIDAD_DATOS', 'STATUS_CAPTURA'
        ])
    with col_table3:
        sort_order = st.radio("Orden", ['Ascendente', 'Descendente'], horizontal=True)
    
    # Aplicar filtros de búsqueda
    if search_seccion:
        sample_display = sample_display[sample_display['SECCIÓN'].str.contains(search_seccion, case=False)]
    
    # Aplicar ordenamiento
    sample_display = sample_display.sort_values(sort_by, ascending=(sort_order == 'Ascendente'))
    
    # Mostrar tabla simple sin estilos que requieran matplotlib
    st.dataframe(
        sample_display,
        use_container_width=True, 
        height=500
    )
    
    # Estadísticas de la tabla
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    col_stats1.metric("Registros mostrados", len(sample_display))
    col_stats2.metric("Total Lista Nominal", f"{sample_display['TOTAL LISTA NOMINAL'].sum():,}")
    col_stats3.metric("Encuestas Realizadas", int(sample_display['ENCUESTAS_REALIZADAS'].sum()))
    col_stats4.metric("Calidad Promedio", f"{sample_display['CALIDAD_DATOS'].mean():.1f}")
    
    # Botón de descarga
    st.markdown("---")
    csv_buffer = io.StringIO()
    sample_display.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Descargar tabla completa como CSV",
        data=csv_buffer.getvalue(),
        file_name=f"secciones_detalle_{selected_distrito}_{selected_municipio}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Botón de descarga de reporte de encuestadores
    st.markdown("### 📊 Reportes Adicionales")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        encuestador_buffer = io.StringIO()
        encuestador_stats.to_csv(encuestador_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar Reporte de Encuestadores",
            data=encuestador_buffer.getvalue(),
            file_name=f"reporte_encuestadores_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        contactos_buffer = io.StringIO()
        contactos_export = filtered_sample[['SECCIÓN', 'EMAILS_VALIDOS', 'CELULARES_VALIDOS', 'PCT_EMAIL_VALIDO', 'PCT_CELULAR_VALIDO']]
        contactos_export.to_csv(contactos_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar Reporte de Contactos",
            data=contactos_buffer.getvalue(),
            file_name=f"reporte_contactos_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer con información técnica
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption(f"📅 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with col_footer2:
    unmatched = set(df['SECCIÓN'].astype(str)) - set(merged_gdf['SECCIÓN'])
    st.caption(f"ℹ️ {len(unmatched)} secciones sin información geográfica")

with col_footer3:
    st.caption("🔒 Dashboard con datos simulados para demostración")