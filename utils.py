import boto3, json
import os
from langchain.schema import Document
from pypdf import PdfReader
import boto3
from langchain_aws import ChatBedrock
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import PromptTemplate
import re
import ast
import time

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime',region_name='us-east-1')
#Extract Information from PDF file
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

def clasificador_texto(texto):
    session=boto3.Session()
    bedrock=session.client(service_name='bedrock-runtime',region_name='us-east-1')
    llm=ChatBedrock(client=bedrock,model_id='anthropic.claude-3-sonnet-20240229-v1:0',region_name='us-east-1')

    response_schema=[
        ResponseSchema(name='Nombre',description='Nombre del Candidato. Por ejemplo: Raul, Juan, Oscar'),
        ResponseSchema(name='Apellido',description='Apellido del Candidato. Por ejemplo: Gomez, Salazar, Baez.'),
        ResponseSchema(name='Nombre y Apellido',description='Nombre y Apellido del Candidato. Por ejemplo: Raul Gomez'),
        ResponseSchema(name='Sexo',description='Sexo del Candidato. Las opciones son: Masculino, Femenino'),
        ResponseSchema(name='Correo',description='Correo electronico del Candidato. Responde No se menciona en caso de no encontrar la respuesta'),
        ResponseSchema(name='Telefono',description='Telefono del Candidato. Responde No se menciona en caso de no encontrar la respuesta'),
        ResponseSchema(name='Fecha de Nacimiento',description='Fecha de Nacimiento del Candidato. Responde No se menciona en caso de no encontrar la respuesta'),
        ResponseSchema(name='Formacion Academica',description='Cual es la formacion academica del candidato? Sea breve y conciso en la respuesta, citando los datos mas relevantes.'),
        ResponseSchema(name='Experiencia Laboral',description='Cual es la experiencia laboral?Sea breve y conciso en la respuesta, citando los datos mas relevantes.')
    ]

    output_parser=StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions=output_parser.get_format_instructions()
    template=""" A continuacion, se te proporcionara texto sobre un curriculum vitae. En base al texto proporcionado deberas extraer la siguiente informacion con el siguiente formato:
    {format_instructions} 
    {texto}


    """

    prompt=PromptTemplate(template=template, input_variables=[texto],partial_variables={'format_instructions':format_instructions})
    #prompt=prompt.format_prompt(texto=texto).text
    #respuesta=llm.invoke(prompt)
    chain=prompt|llm|output_parser
    respuesta=chain.invoke(texto)
    return respuesta

# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=read_pdf_data(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs

def preguntar_llm(extracted_text,input):
    doc_message = {
        "role": "user",
        "content": [
            {
                "text": extracted_text
            },
            { 
                "text": 'Actua como un experto reclutador de talentos para una empresa del ambito financiero.Dame una breve descripcion de las fortalezas y debilidades del candidato respecto a la Descripcion del Trabajo. Descripcion del trabajo:'+input 
            }
        ]
    }
    response = bedrock.converse(
        modelId="amazon.nova-pro-v1:0",
        messages=[doc_message],
        inferenceConfig={
            "maxTokens": 4000,
            "temperature": 0
        },
    )
    return response['output']['message']['content'][0]['text']




def embedding(text):
    body = json.dumps({
            "inputText": text,
        })
    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId="amazon.titan-embed-text-v1", accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get('body').read())

    return response_body['embedding']


def fortalezas_y_debilidades(texto,descripcion_trabajo):
    session=boto3.Session()
    bedrock=session.client(service_name='bedrock-runtime',region_name='us-east-1')
    llm=ChatBedrock(client=bedrock,model_id='anthropic.claude-3-sonnet-20240229-v1:0',region_name='us-east-1')

    response_schema=[
        ResponseSchema(name='Fortalezas',description='Cuales son las Fortalezas del Candidato con respecto al la descripcion del trabajo?'),
        ResponseSchema(name='Debilidades',description='Cuales son las Debilidades del Candidato con respecto al la descripcion del trabajo?'),
        ResponseSchema(name='Categoria',description='En que Area de la organizacion puede el candidato encajar basado en sus aptitudes y habilidades? No agregues informacion extra. Expresate con maximo 5 palabras'),
    ]

    output_parser=StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions=output_parser.get_format_instructions()
    template=""" 
    Actua como un reclutador de talentos con experiencia en el ambito bancario. 
    En base a la descripcion del puesto de trabajo que se te proporcionara deberas realizar un analisis 
    con las siguientes instrucciones tomando en cuenta los datos del curriculum vitae. 
    Tu respuesta debe ser bien detallada y precisa.
   A continuacion se presentan las instrucciones:
    {format_instructions} \n
    La descripcion de la busqueda del trabajo son las siguientes:
   {descripcion_trabajo}
    A continuacion se presentan los datos del curriculum vitae de la persona:
    {texto}

    """

    prompt = PromptTemplate(
    template=template,
    input_variables=["descripcion_trabajo", "texto"],
    partial_variables={"format_instructions": format_instructions})
    #prompt=prompt.format_prompt(texto=texto).text
    #respuesta=llm.invoke(prompt)
    prompt_text = prompt.format(descripcion_trabajo=descripcion_trabajo, texto=texto)
    respuesta=llm.invoke(prompt_text).content
    pattern=r'({[^}]+})'
    matches=re.findall(pattern,respuesta)
    diccionario=ast.literal_eval(matches[0])
    fortalezas=diccionario['Fortalezas']
    debilidades=diccionario['Debilidades']
    categoria=diccionario['Categoria']
    return fortalezas,debilidades,categoria

def analizar_requerimientos_excluyentes(texto,requerimientos):
    session=boto3.Session()
    bedrock=session.client(service_name='bedrock-runtime',region_name='us-east-1')
    llm=ChatBedrock(client=bedrock,model_id='anthropic.claude-3-sonnet-20240229-v1:0',region_name='us-east-1')
    template='''
    Actua como un reclutador de talentos con experiencia en el ambito bancario.
    A continuacion, se te proporcionara informacion sobre el curriculum vitae de un candidato 
    y tambien los requerimientos excluyentes de la busqueda de la persona que se desea contratar. 
    Deberas responder solamente con:
    Excluyente: en caso de que no cumpla con los requerimientos.
    No Excluyente: en caso de que cumpla con los requerimientos.
    Las opciones de respuesta son: Excluyente, No Excluyente
    Tambien debes agregar la razon de porque es No Excluyente y porque es Excluyente sea cual fuera el caso.
    No debes agregar texto extra a la respuesta, debes ser directo y conciso con la respuesta
    Los Requerimientos son los siguientes: \n 
    {requerimientos}
    La informacion del Curriculum Vitae del candidato es la siguiente: \n
    {texto}
    '''

    prompt = PromptTemplate(
    template=template,
    input_variables=["texto", "requerimientos"])
    prompt_text = prompt.format(requerimientos=requerimientos, texto=texto)
    respuesta=llm.invoke(prompt_text).content
    return respuesta