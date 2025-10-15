import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import io

# Configuración de la página
st.set_page_config(page_title="Dashboard Electoral Guerrero", layout="wide", initial_sidebar_state="expanded")

# Rutas de archivos
csv_path = "/Users/omartellez/guerrero/consolidado_seleccion.csv"
shp_path = "/Users/omartellez/guerrero/SECCION.shp"
sample_path = "/Users/omartellez/guerrero/plan_de_campo.csv"

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(shp_path)
    df_sample = pd.read_csv(sample_path)
    # Asegurar tipos consistentes
    df['SECCIÓN'] = df['SECCIÓN'].astype(str)
    gdf['SECCION'] = gdf['SECCION'].astype(str)
    df_sample['SECCIÓN'] = df_sample['SECCIÓN'].astype(str)
    # Merge CSV con shapefile
    merged_gdf = gdf.merge(df, left_on='SECCION', right_on='SECCIÓN', how='inner')
    # Marcar secciones muestreadas y agregar número de encuestas
    merged_gdf['is_sampled'] = merged_gdf['SECCIÓN'].isin(df_sample['SECCIÓN'])
    merged_gdf = merged_gdf.merge(df_sample[['SECCIÓN', 'ENCUESTAS_ASIGNADAS']], on='SECCIÓN', how='left')
    merged_gdf['ENCUESTAS_ASIGNADAS'] = merged_gdf['ENCUESTAS_ASIGNADAS'].fillna(0).astype(int)
    # Asegurar proyección EPSG:4326
    if merged_gdf.crs != "EPSG:4326":
        merged_gdf = merged_gdf.to_crs(epsg=4326)
    return df, merged_gdf, df_sample

df, merged_gdf, df_sample = load_data()

# Barra lateral para filtros
st.sidebar.title("🔍 Filtros de Visualización")
distrito_options = ['Todos'] + sorted(df['Distrito'].unique().tolist())
selected_distrito = st.sidebar.selectbox("Distrito", distrito_options)

# Filtro de Municipio
if selected_distrito != 'Todos':
    municipio_options = ['Todos'] + sorted(df[df['Distrito'] == selected_distrito]['MUNICIPIOS'].unique().tolist())
else:
    municipio_options = ['Todos'] + sorted(df['MUNICIPIOS'].unique().tolist())
selected_municipio = st.sidebar.selectbox("Municipio", municipio_options)

# Filtro de secciones muestreadas
st.sidebar.markdown("---")
show_sampled = st.sidebar.checkbox("🎯 Solo secciones muestreadas", value=False)

# Filtrar datos
filtered_df = df.copy()
filtered_gdf = merged_gdf.copy()

if selected_distrito != 'Todos':
    filtered_df = filtered_df[filtered_df['Distrito'] == selected_distrito]
    filtered_gdf = filtered_gdf[filtered_gdf['Distrito'] == selected_distrito]
if selected_municipio != 'Todos':
    filtered_df = filtered_df[filtered_df['MUNICIPIOS'] == selected_municipio]
    filtered_gdf = filtered_gdf[filtered_gdf['MUNICIPIOS'] == selected_municipio]
if show_sampled:
    filtered_df = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])]
    filtered_gdf = filtered_gdf[filtered_gdf['is_sampled']]

# Calcular métricas
total_secciones = filtered_df['SECCIÓN'].nunique()
secciones_muestreadas = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])]['SECCIÓN'].nunique()
tasa_muestreo = (secciones_muestreadas / total_secciones * 100) if total_secciones > 0 else 0
total_lista = filtered_df['TOTAL LISTA NOMINAL'].sum()
lista_muestra = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])]['TOTAL LISTA NOMINAL'].sum()
cobertura_poblacional = (lista_muestra / total_lista * 100) if total_lista > 0 else 0

if show_sampled:
    total_encuestas = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])].merge(
        df_sample[['SECCIÓN', 'ENCUESTAS_ASIGNADAS']], on='SECCIÓN', how='left'
    )['ENCUESTAS_ASIGNADAS'].sum()
else:
    total_encuestas = df_sample['ENCUESTAS_ASIGNADAS'].sum()

promedio_encuestas = (total_encuestas / secciones_muestreadas) if secciones_muestreadas > 0 else 0

# Título principal
st.title("📊 Dashboard Electoral - Guerrero")
st.markdown(f"**Vista actual:** {selected_distrito} {'→ ' + selected_municipio if selected_municipio != 'Todos' else ''}")

# Sistema de Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Instrucciones", "📈 Resumen Ejecutivo", "🗺️ Mapa Interactivo", "📊 Análisis de Cobertura", "💾 Datos y Descarga"])

# ==================== TAB 1: INSTRUCCIONES ====================
with tab1:
    st.header("📋 Instrucciones de Uso")
    
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        st.subheader("🎯 Objetivo del Dashboard")
        st.markdown("""
        Este dashboard permite visualizar y analizar:
        - **Distribución electoral** en el estado de Guerrero
        - **Plan de muestreo** de 400 secciones para encuestas
        - **Cobertura geográfica y poblacional** de la muestra
        - **Asignación de encuestas** por sección
        """)
        
        st.subheader("🔍 Cómo usar los filtros")
        st.markdown("""
        **Barra lateral izquierda:**
        1. **Distrito:** Filtra por distrito electoral
        2. **Municipio:** Filtra por municipio específico
        3. **☑️ Solo secciones muestreadas:** Muestra únicamente las 400 secciones seleccionadas
        
        Los filtros se aplican automáticamente a todas las pestañas.
        """)
    
    with col_inst2:
        st.subheader("📑 Navegación por pestañas")
        st.markdown("""
        **📈 Resumen Ejecutivo**
        - Métricas clave del proyecto
        - Indicadores de cobertura
        - Estadísticas generales
        
        **🗺️ Mapa Interactivo**
        - Visualización geográfica de secciones
        - Capas intercambiables (Padrón, Lista Nominal, Muestra)
        - Tooltips con información detallada
        
        **📊 Análisis de Cobertura**
        - Gráficas de distribución de muestra
        - Cobertura por municipio
        - Validación de asignación de encuestas
        
        **💾 Datos y Descarga**
        - Tabla completa de secciones muestreadas
        - Exportación a CSV
        - Filtros adicionales
        """)
    
    st.subheader("💡 Consejos de uso")
    st.info("""
    - **Para explorar un área específica:** Usa los filtros de Distrito y Municipio
    - **Para revisar el plan de campo:** Activa "Solo secciones muestreadas" y ve a la pestaña "Datos y Descarga"
    - **Para validar cobertura:** Revisa las gráficas en "Análisis de Cobertura"
    - **En el mapa:** Usa el control de capas (esquina superior derecha) para cambiar entre vistas
    """)
    
    st.subheader("📊 Parámetros del Estudio")
    param_col1, param_col2, param_col3 = st.columns(3)
    param_col1.metric("Secciones totales", df['SECCIÓN'].nunique())
    param_col2.metric("Secciones en muestra", df_sample['SECCIÓN'].nunique())
    param_col3.metric("Total de encuestas", int(df_sample['ENCUESTAS_ASIGNADAS'].sum()))
    
    st.markdown("**Nivel de confianza:** 95% | **Margen de error:** ±1.55%")

# ==================== TAB 2: RESUMEN EJECUTIVO ====================
with tab2:
    st.header("📈 Métricas Principales")
    
    # Métricas superiores
    col1, col2, col3 = st.columns(3)
    col1.metric("Municipios", filtered_df['MUNICIPIOS'].nunique())
    col2.metric("Total Secciones", total_secciones)
    col3.metric("Total Lista Nominal", f"{total_lista:,}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Secciones Muestreadas", secciones_muestreadas, delta=f"{tasa_muestreo:.1f}% del total")
    col5.metric("Total Encuestas", int(total_encuestas))
    col6.metric("Promedio Encuestas/Sección", f"{promedio_encuestas:.1f}")
    
    st.markdown("---")
    
    # Indicadores de cobertura
    st.subheader("📊 Indicadores de Cobertura")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Tasa de muestreo
        fig_gauge1 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = tasa_muestreo,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Tasa de Muestreo (% Secciones)"},
            delta = {'reference': 10},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 15], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10}}))
        fig_gauge1.update_layout(height=300)
        st.plotly_chart(fig_gauge1, use_container_width=True)
    
    with col_b:
        # Cobertura poblacional
        fig_gauge2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cobertura_poblacional,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cobertura Poblacional (% Lista Nominal)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 15], 'color': "gray"}]}))
        fig_gauge2.update_layout(height=300)
        st.plotly_chart(fig_gauge2, use_container_width=True)
    
    st.markdown("---")
    
    # Distribución por distrito
    st.subheader("📍 Distribución de la Muestra por Distrito")
    sample_distrito = df_sample.groupby('Distrito').agg({
        'SECCIÓN': 'count',
        'ENCUESTAS_ASIGNADAS': 'sum'
    }).reset_index()
    sample_distrito.columns = ['Distrito', 'Secciones', 'Encuestas']
    
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
            y='Encuestas',
            title="Encuestas Asignadas por Distrito",
            color='Encuestas',
            color_continuous_scale='Greens'
        )
        fig_encuestas.update_layout(showlegend=False)
        st.plotly_chart(fig_encuestas, use_container_width=True)

# ==================== TAB 3: MAPA INTERACTIVO ====================
with tab3:
    st.header("🗺️ Mapa de Secciones Electorales")
    
    # Opciones de visualización del mapa
    col_map1, col_map2 = st.columns([3, 1])
    with col_map2:
        st.markdown("**Controles:**")
        st.markdown("- Usa el control de capas en el mapa")
        st.markdown("- Haz clic en las secciones para ver detalles")
        st.markdown("- Zoom con scroll o botones")
    
    # Crear mapa
    m = folium.Map(location=[17.55, -99.50], zoom_start=8, tiles="CartoDB positron")
    
    # Capa para TOTAL PADRÓN
    folium.Choropleth(
        geo_data=filtered_gdf,
        name="🔵 Total Padrón",
        data=filtered_gdf,
        columns=['SECCIÓN', 'TOTAL PADRÓN'],
        key_on='feature.properties.SECCIÓN',
        fill_color='YlOrRd',
        fill_opacity=0.6,
        line_opacity=0.3,
        legend_name='Total Padrón',
        show=True
    ).add_to(m)
    
    # Capa para TOTAL LISTA NOMINAL
    folium.Choropleth(
        geo_data=filtered_gdf,
        name="🟢 Total Lista Nominal",
        data=filtered_gdf,
        columns=['SECCIÓN', 'TOTAL LISTA NOMINAL'],
        key_on='feature.properties.SECCIÓN',
        fill_color='BuGn',
        fill_opacity=0.6,
        line_opacity=0.3,
        legend_name='Total Lista Nominal',
        show=False
    ).add_to(m)
    
    # Capa para secciones muestreadas con tooltip enriquecido
    sampled_sections = filtered_gdf[filtered_gdf['is_sampled']].copy()
    if not sampled_sections.empty:
        folium.GeoJson(
            sampled_sections,
            name="🎯 Secciones Muestreadas",
            style_function=lambda x: {
                'fillColor': '#ff4444', 
                'fillOpacity': 0.7, 
                'color': 'darkred', 
                'weight': 2
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['SECCIÓN', 'MUNICIPIOS', 'TOTAL PADRÓN', 'TOTAL LISTA NOMINAL', 'ENCUESTAS_ASIGNADAS'],
                aliases=['Sección', 'Municipio', 'Total Padrón', 'Total Lista Nominal', 'Encuestas Asignadas'],
                localize=True
            ),
            show=True
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=1400, height=700, returned_objects=[])

# ==================== TAB 4: ANÁLISIS DE COBERTURA ====================
with tab4:
    st.header("📊 Análisis de Cobertura de Muestra")
    
    # Gráfica 1: Cobertura de muestra por municipio
    st.subheader("Cobertura de Secciones por Municipio")
    
    munic_stats = filtered_df.groupby('MUNICIPIOS').agg({
        'SECCIÓN': 'nunique'
    }).reset_index()
    munic_stats.columns = ['Municipio', 'Total_Secciones']
    
    munic_sample = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])].groupby('MUNICIPIOS').agg({
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
    
    # Gráfica 2: Dispersión de asignación de encuestas
    st.subheader("Validación de Asignación de Encuestas")
    st.markdown("Esta gráfica permite verificar si la asignación de encuestas es proporcional al tamaño de cada sección.")
    
    scatter_data = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])].merge(
        df_sample[['SECCIÓN', 'ENCUESTAS_ASIGNADAS']], on='SECCIÓN', how='left'
    )
    
    fig_scatter = px.scatter(
        scatter_data,
        x='TOTAL LISTA NOMINAL',
        y='ENCUESTAS_ASIGNADAS',
        color='Distrito',
        hover_data=['SECCIÓN', 'MUNICIPIOS'],
        title="Relación entre Tamaño de Sección y Encuestas Asignadas",
        labels={
            'TOTAL LISTA NOMINAL': 'Total Lista Nominal (Tamaño de Sección)',
            'ENCUESTAS_ASIGNADAS': 'Número de Encuestas Asignadas'
        }
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Gráfica 3: Intensidad de encuestas
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

# ==================== TAB 5: DATOS Y DESCARGA ====================
with tab5:
    st.header("💾 Tabla de Secciones Muestreadas")
    
    # Preparar tabla
    sample_display = filtered_df[filtered_df['SECCIÓN'].isin(df_sample['SECCIÓN'])][
        ['SECCIÓN', 'MUNICIPIOS', 'Distrito', 'TOTAL PADRÓN', 'TOTAL LISTA NOMINAL']
    ].merge(
        df_sample[['SECCIÓN', 'ENCUESTAS_ASIGNADAS']], on='SECCIÓN', how='left'
    ).sort_values(['Distrito', 'MUNICIPIOS', 'SECCIÓN'])
    
    # Opciones de visualización
    col_table1, col_table2, col_table3 = st.columns(3)
    with col_table1:
        search_seccion = st.text_input("🔍 Buscar Sección", "")
    with col_table2:
        sort_by = st.selectbox("Ordenar por", ['SECCIÓN', 'MUNICIPIOS', 'Distrito', 'TOTAL LISTA NOMINAL', 'ENCUESTAS_ASIGNADAS'])
    with col_table3:
        sort_order = st.radio("Orden", ['Ascendente', 'Descendente'], horizontal=True)
    
    # Aplicar filtros de búsqueda
    if search_seccion:
        sample_display = sample_display[sample_display['SECCIÓN'].str.contains(search_seccion, case=False)]
    
    # Aplicar ordenamiento
    sample_display = sample_display.sort_values(sort_by, ascending=(sort_order == 'Ascendente'))
    
    # Mostrar tabla
    st.dataframe(sample_display, use_container_width=True, height=500)
    
    # Estadísticas de la tabla
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    col_stats1.metric("Registros mostrados", len(sample_display))
    col_stats2.metric("Total Lista Nominal", f"{sample_display['TOTAL LISTA NOMINAL'].sum():,}")
    col_stats3.metric("Total Encuestas", int(sample_display['ENCUESTAS_ASIGNADAS'].sum()))
    
    # Botón de descarga
    st.markdown("---")
    csv_buffer = io.StringIO()
    sample_display.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Descargar tabla completa como CSV",
        data=csv_buffer.getvalue(),
        file_name=f"secciones_muestreadas_{selected_distrito}_{selected_municipio}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer con información de secciones no mapeadas
st.markdown("---")
unmatched = set(df['SECCIÓN'].astype(str)) - set(merged_gdf['SECCIÓN'])
unmatched_sample = set(df_sample['SECCIÓN'].astype(str)) - set(merged_gdf['SECCIÓN'])
if unmatched or unmatched_sample:
    with st.expander("ℹ️ Información técnica"):
        st.caption(f"Nota: {len(unmatched)} secciones del CSV principal y {len(unmatched_sample)} de la muestra no se pudieron mapear al shapefile.")