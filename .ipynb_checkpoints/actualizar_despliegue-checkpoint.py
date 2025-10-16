# -*- coding: utf-8 -*-
import pandas as pd
import os # Se importa la librería 'os' para manejar rutas de archivo

print("Iniciando el script para actualizar el plan de campo del dashboard...")

# --- PARÁMETROS DE CONFIGURACIÓN ---
# Se define la ruta base del proyecto para que sea fácil de modificar si es necesario
RUTA_BASE = '/Users/omartellez/guerrero/dashboard-electoral-guerrero/data/'

# Se construyen las rutas completas para los archivos de entrada y salida
NOMBRE_ARCHIVO_ENTRADA = os.path.join(RUTA_BASE, 'consolidado_seleccion.csv')
NOMBRE_ARCHIVO_SALIDA = os.path.join(RUTA_BASE, 'plan_de_campo.csv')

# Parámetros del muestreo (se mantienen consistentes)
TAMANO_MUESTRA_SECCIONES = 400
ENTREVISTAS_POR_SECCION = 10
COLUMNA_SECCION_UNICA = 'SECCIÓN'
SEMILLA_ALEATORIA = 42

try:
    # --- PASO 1: CARGAR Y CONFIRMAR DATOS ÚNICOS ---
    df_original = pd.read_csv(NOMBRE_ARCHIVO_ENTRADA)
    print(f"Archivo '{NOMBRE_ARCHIVO_ENTRADA}' cargado. Contiene {len(df_original)} filas.")

    # >> Verificación de duplicados <<
    df_corregido = df_original.drop_duplicates(subset=[COLUMNA_SECCION_UNICA], keep='first')
    
    num_filas_originales = len(df_original)
    num_filas_corregidas = len(df_corregido)

    if num_filas_originales == num_filas_corregidas:
        print(f"Confirmado: La base ya no contenía duplicados. El universo es de {num_filas_corregidas} secciones únicas.")
    else:
        print(f"Se encontraron y eliminaron {num_filas_originales - num_filas_corregidas} duplicados. El universo corregido es de {num_filas_corregidas} secciones.")

    # --- PASO 2: RECALCULAR LA MUESTRA ESTRATIFICADA ---
    print(f"Calculando la muestra de {TAMANO_MUESTRA_SECCIONES} secciones...")
    
    padron_total = df_corregido['TOTAL PADRÓN'].sum()
    calculos_muestra = df_corregido.groupby('MUNICIPIOS').agg(
        padron_municipio=('TOTAL PADRÓN', 'sum'),
        total_secciones_municipio=('SECCIÓN', 'count')
    ).reset_index()

    calculos_muestra['peso_proporcional'] = calculos_muestra['padron_municipio'] / padron_total
    calculos_muestra['n_ideal'] = calculos_muestra['peso_proporcional'] * TAMANO_MUESTRA_SECCIONES
    calculos_muestra['n_final'] = calculos_muestra['n_ideal'].round().astype(int)

    diferencia = TAMANO_MUESTRA_SECCIONES - calculos_muestra['n_final'].sum()
    if diferencia != 0:
        indices_ajuste = abs(calculos_muestra['n_ideal'] - calculos_muestra['n_final']).nlargest(abs(diferencia)).index
        ajuste = 1 if diferencia > 0 else -1
        calculos_muestra.loc[indices_ajuste, 'n_final'] += ajuste
    
    calculos_muestra['n_final'] = calculos_muestra.apply(
        lambda row: min(row['n_final'], row['total_secciones_municipio']), axis=1
    )

    # --- PASO 3: GENERAR LA NUEVA MUESTRA ---
    lista_de_muestras = []
    for _, datos in calculos_muestra.iterrows():
        n_muestras = datos['n_final']
        if n_muestras > 0:
            muestra = df_corregido[df_corregido['MUNICIPIOS'] == datos['MUNICIPIOS']].sample(
                n=int(n_muestras), random_state=SEMILLA_ALEATORIA
            )
            lista_de_muestras.append(muestra)

    muestra_final = pd.concat(lista_de_muestras)

    # --- PASO 4: ASIGNAR ENTREVISTAS Y GUARDAR ---
    muestra_final['ENCUESTAS_ASIGNADAS'] = ENTREVISTAS_POR_SECCION
    
    # Guardar el archivo final, sobreescribiendo el anterior en la ruta especificada
    muestra_final.to_csv(NOMBRE_ARCHIVO_SALIDA, index=False, encoding='utf-8-sig')
    
    total_entrevistas = len(muestra_final) * ENTREVISTAS_POR_SECCION
    
    print("-" * 50)
    print("¡ÉXITO! 🚀")
    print(f"Se ha guardado y actualizado el archivo en la siguiente ruta:")
    print(f"'{NOMBRE_ARCHIVO_SALIDA}'")
    print(f"El nuevo plan contiene {len(muestra_final)} secciones y un total de {total_entrevistas} entrevistas.")
    print("Ya puedes hacer push para actualizar tu despliegue en Streamlit.")
    print("-" * 50)

except FileNotFoundError:
    print(f"🚨 ERROR: No se encontró el archivo en la ruta especificada.")
    print(f"Verifica que la ruta '{NOMBRE_ARCHIVO_ENTRADA}' sea correcta.")
except Exception as e:
    print(f"🚨 Ocurrió un error inesperado: {e}")