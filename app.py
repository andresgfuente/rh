import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
import boto3
from scipy.spatial import distance
import time 
import pandas as pd

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime',region_name='us-east-1')
#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def main():
    load_dotenv()

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("RRHH - Asistente de selección de personal...💁")
    st.subheader("Puedo ayudarte a encontrar el mejor perfil basado en tus requerimientos")

    # Dividir en dos columnas
    col1, col2 = st.columns(2)

    # Primera columna: Descripción del trabajo
    with col1:
        st.subheader("Descripción del trabajo")
        job_description = st.text_area("Escriba aquí la descripción del trabajo (obligatorio)", key="1")

    # Segunda columna: Requisitos excluyentes (opcional con valor predeterminado)
    with col2:
        st.subheader("Requisitos excluyentes")
        job_requirements = st.text_area(
            "Escriba aquí los requisitos excluyentes (opcional)", 
            key="2", 
            value="No se mencionan requerimientos excluyentes"
        )

    # Cargar archivos PDF
    pdf = st.file_uploader(
        "Suba los documentos aquí (formato PDF solamente admitido)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    # Botón para análisis
    submit = st.button("Ayúdame con el análisis")
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = pd.DataFrame(columns=[
        "Nombre", 
        "Apellido",
        "Nombre y Apellido",
        "Sexo",
        "Correo",
        "Teléfono",
        "Fecha de Nacimiento",
        "Formación Académica",
        "Experiencia Laboral",
        "Match Score (%)",
        "Fortalezas",
        "Debilidades",
        "Requisitos Excluyentes",
        "categoria"
    ])
    # Manejo del botón
    if submit:
        if not job_description.strip():
            st.error("Por favor, completa la descripción del trabajo antes de continuar.")
        else:
            requerimientos_excluyentes = job_requirements.strip() or "No se mencionan requerimientos excluyentes"
            with st.spinner('Espera un rato...'):

                # Crear un ID único para los documentos
                st.session_state['unique_id'] = uuid.uuid4().hex

                # Crear una lista de documentos a partir de los archivos PDF cargados
                final_docs_list = create_docs(pdf, st.session_state['unique_id'])

                # Mostrar la cantidad de currículums cargados
                st.write("*Se han cargado:* " + str(len(final_docs_list)) + " currículums")
                for n in range(len(final_docs_list)):
                    # Clasificar texto del currículum
                    dictionario_texto = clasificador_texto(final_docs_list[n].page_content)
                    nombre = dictionario_texto.get("Nombre", "No se menciona")
                    apellido = dictionario_texto.get("Apellido", "No se menciona")
                    nombre_apellido = dictionario_texto.get("Nombre y Apellido", "No se menciona")
                    sexo = dictionario_texto.get("Sexo", "No se menciona")
                    correo = dictionario_texto.get("Correo", "No se menciona")
                    telefono = dictionario_texto.get("Telefono", "No se menciona")
                    fecha_nacimiento = dictionario_texto.get("Fecha de Nacimiento", "No se menciona")
                    formacion_academica = dictionario_texto.get("Formacion Academica", "No se menciona")
                    experiencia_laboral = dictionario_texto.get("Experiencia Laboral", "No se menciona")
                    
                    
                    
                    #st.subheader("👉 " + dictionario_texto['Nombre y Apellido'])

                    # Mostrar archivo cargado
                    #st.write("**File** : " + str(n + 1))

                    # Calcular similitud con descripción del trabajo
                    embedding1 = embedding(final_docs_list[n].page_content)
                    embedding2 = embedding(job_description)
                    similitud = str(round((1 - distance.cosine(embedding1, embedding2)) * 100, 2))
                    #st.info("**Match Score** : " + similitud + '%')

                    with st.expander('Show me 👀'):
                        # Calcular fortalezas y debilidades
                        fortalezas, debilidades,categoria = fortalezas_y_debilidades(final_docs_list[n].page_content, job_description)
                        
                        # Analizar requisitos excluyentes
                        resultado_requerimientos = analizar_requerimientos_excluyentes(
                            texto=final_docs_list[n].page_content, 
                            requerimientos=requerimientos_excluyentes
                        )

                        
                        #st.write('**Area ideal:**')
                        #st.write(categoria)
                        #st.write('**Fortalezas del Candidato:**')
                        #st.write(fortalezas)
                        #st.write('**Debilidades del Candidato:**')
                        #st.write(debilidades)
                        #st.write('**Requisitos Excluyentes:**')
                        #st.write(resultado_requerimientos)
                        time.sleep(5)
                        
                        st.session_state['results_df'] = pd.concat([
                        st.session_state['results_df'], 
                        pd.DataFrame([{
                            "Nombre": nombre,
                            "Apellido": apellido,
                            "Nombre y Apellido": nombre_apellido,
                            "Sexo": sexo,
                            "Correo": correo,
                            "Teléfono": telefono,
                            "Fecha de Nacimiento": fecha_nacimiento,
                            "Formación Académica": formacion_academica,
                            "Experiencia Laboral": experiencia_laboral,
                            "Match Score (%)": similitud,
                            "Fortalezas": fortalezas,
                            "Debilidades": debilidades,
                            "Requisitos Excluyentes": resultado_requerimientos,
                            "categoria":categoria
                        }])
                    ], ignore_index=True)
                        
    if not st.session_state['results_df'].empty:
        st.subheader("Resultados del análisis")
        st.dataframe(st.session_state['results_df'])                     
                        
                        
                    #Create embeddings instance

#Invoking main function
if __name__ == '__main__':
    main()
