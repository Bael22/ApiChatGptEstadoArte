from tempfile import NamedTemporaryFile
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pdfplumber
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from fastapi.responses import FileResponse
from pathlib import Path
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analizar/")
async def analizar(tema: str = Form(...), pdfs: list[UploadFile] = File(...)):
    resultados = []
    client = OpenAI()
    df = pd.DataFrame(columns=[
        "Nombre del Artículo", "Tipo de Brecha", 
        "Vacío académico y oportunidad de innovación", "DOI", "Fuente/Revista"
    ])

    brechas_definicion = """
    a) Brecha de conocimiento: equivale a un vacío en el conocimiento que puede surgir por no haber encontrado una respuesta a lo desconocido o por la obtención de resultados no esperados.
    b) Brecha teórica: se origina cuando una teoría o modelo teórico no explica suficientemente un fenómeno, o cuando el fenómeno puede ser explicado a partir de varias teorías, lo que obliga a determinar cuál es la teoría superior.
    c) Brecha metodológica: surge cuando se emplea un método poco pertinente que produce resultados sesgados. También se observa en el uso reiterativo de un método que conduce a resultados limitados.
    d) Brecha en la selección de la población y muestra: ocurre por escoger grupos no representativos o por falta de estudios sobre poblaciones excluidas o marginadas.
    e) Brecha de evidencia contradictoria: sucede cuando estudios sobre un mismo tema presentan hallazgos opuestos.
    """
    
    for pdf in pdfs:
        if not pdf.filename.lower().endswith(".pdf"):
            resultados.append(f"El archivo **{pdf.filename}** fue rechazado porque no es un archivo PDF.")
            continue
        pdf.filename = pdf.filename.replace(" ", "_").replace("(", "").replace(")", "")
        with pdfplumber.open(pdf.file) as p:
            texto_total = ''
            texto_posibletitulo = (p.pages[0].extract_text() or '')[:650]
            texto_hoja1=(p.pages[0].extract_text() or '')
            for i, page in enumerate(p.pages):
                texto = page.extract_text() or ''
                texto_total += texto
        try:
            # ---------- EXTRACCIÓN DEL TÍTULO CON PROMPT ----------
            prompt_titulo = f"""
            

            {texto_posibletitulo}

            ¿Cuál es el título del artículo? Devuélvelo exacto, sin comillas. Solo el título.
            """
            print("Primera pagina acortada: "+texto_posibletitulo)
            response_titulo = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_titulo}]
            )
            titulo_articulo = response_titulo.choices[0].message.content.strip()
            print("Titulo: "+titulo_articulo)
            # ---------- EXTRACCIÓN DEL DOI ----------
            doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", texto_hoja1)
            if not doi_match:
                resultados.append(f"El archivo **{pdf.filename}** fue descartado porque no contiene un DOI válido, lo que sugiere que no es un artículo científico.")
                continue
            doi = doi_match.group(1)
            print("DOI: "+doi)
            # ---------- EXTRACCIÓN DE LA FUENTE/BASE DE DATOS ----------
            prompt_fuente = f"""
            Este es el título de un artículo científico:

            {titulo_articulo}

            ¿En qué base de datos o revista científica está más probablemente indexado este artículo? 
            Devuelve solo un nombre: Scopus, Springer, ScienceDirect, IEEE, MDPI, Taylor & Francis, Wiley, arXiv u otro.
            """

            response_fuente = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_fuente}]
            )
            fuente = response_fuente.choices[0].message.content.strip()
            print("Fuente: "+fuente)
            # ---------- EXTRACCIÓN DEL ABSTRACT ----------
            def extraer_resumen(texto_total: str) -> str:
                match = re.search(r"(a b s t r a c t|abstract|resumen)[\s\n:]*([\s\S]{200,2000})", texto_total, re.IGNORECASE)
                if match:
                    return match.group(2).strip()
                bloques = re.split(r'\n{2,}', texto_total.strip())
                posibles = [b for b in bloques if 300 < len(b) < 2000]
                return posibles[0] if posibles else texto_total[:2000]
            resumen = extraer_resumen(texto_hoja1)
            print("Abstract: "+resumen)
            # ---------- PRIMER PROMPT: Relevancia ----------
            primer_prompt = f"""
            Según el siguiente resumen del artículo:

            {resumen}

            Investigo sobre el tema: "{tema}".

            Evalúa qué tan relacionado está este artículo con ese tema. Clasifica la relevancia como:
            - Muy relevante
            - Relevante
            - Levemente relevante
            - No relevante

            Si es "no relevante", explica brevemente el porqué en 2 oraciones. Si sí, indica el nivel.
            """
            response1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": primer_prompt}]
            )
            relevancia = response1.choices[0].message.content.strip().lower()
            
            if "no relevante" in relevancia:
                resultados.append(f"No se pudo seguir analizando el archivo **{pdf.filename}** porque no es relevante para su investigación.\n{relevancia}")
                continue
            
            
            print("Relevacia: "+relevancia)  
        except Exception as e:
            print(f"⚠️ Error al procesar {pdf.filename}: {e}")
            continue
        # ---------- EXTRACCIÓN DE CONCLUSIÓN ----------
        texto_ultimas_paginas = ''
        ultima_pagina = min(5, len(p.pages))  # Leer hasta las últimas 5 páginas
        for page in p.pages[-ultima_pagina:]:
            texto_ultimas_paginas += page.extract_text() or ''

        conclusion_match = re.search(r"(Discussion|Summary|Conclusion|Conclusion and future research directions|Conclusión|Discusión)[\s\n:]*([\s\S]{300,4000})", texto_ultimas_paginas, re.IGNORECASE)
        conclusion = conclusion_match.group(2).strip() if conclusion_match else texto_ultimas_paginas[-2000:]

        # ---------- 2º PROMPT: Análisis preliminar ----------
        segundo_prompt = f"""
        Según esta conclusión del artículo:

        {conclusion}

        Indica:
        - Tipo de brecha (elige solo una: conocimiento, teórica, metodológica, muestra, contradictoria).
        - Vacío académico y oportunidad de innovación (breve y claro).
        """
        response2 = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": segundo_prompt}]
        )
        gpt_preliminar = response2.choices[0].message.content.strip()
        print("conclusiones: "+conclusion)
        # ---------- 3º PROMPT: Revisión precisa ----------
        tercer_prompt = f"""
        Analiza de nuevo la conclusión anterior y considerando las siguientes definiciones para elegir el tipo de brecha más adecuado:

        {brechas_definicion}

        Entrega:
        - Tipo de brecha (una sola).
        - Vacío académico y oportunidad de innovación (más detallado que antes).
        """
        response3 = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": tercer_prompt}]
        )
        gpt_refinado = response3.choices[0].message.content.strip()

        # ---------- FUSIÓN FINAL ----------
        fusion_prompt = f"""
        Estas son dos versiones del análisis para el mismo artículo:

        Versión preliminar:
        {gpt_preliminar}

        Versión refinada:
        {gpt_refinado}

        Compara ambas y entrega una sola versión final en este formato (no sobrepasar de los 400 caracteres en la respuesta ):

        Tipo de brecha: ...
        Vacío académico y oportunidad de innovación: ...

        Usa las definiciones dadas antes y escoge la brecha más adecuada.
        """
        fusion_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fusion_prompt}]
        )
        # Procesamiento del análisis fusionado
        fusion_result = fusion_response.choices[0].message.content.strip()

        # Extraemos el tipo de brecha
        tipo_match = re.search(r"(?i)tipo de brecha\s*:\s*(.+)", fusion_result)
        tipo_brecha = tipo_match.group(1).strip() if tipo_match else "No identificado"

        # Extraemos todo el contenido de 'Vacío académico y oportunidad de innovación'
        vacio_match = re.search(r"(?i)vac[ií]o acad[eé]mico.*?:\s*(.+)", fusion_result, re.DOTALL)
        oportunidad_match = re.search(r"(?i)oportunidad.*?:\s*(.+)", fusion_result, re.DOTALL)

        vacio_texto = vacio_match.group(1).strip() if vacio_match else ""
        oportunidad_texto = oportunidad_match.group(1).strip() if oportunidad_match else ""

        # Concatenamos en el formato deseado para el Excel
        vacio_innovacion = f"Vacío académico: {vacio_texto}\nOportunidad de innovación: {oportunidad_texto}"

        df.loc[len(df)] = [titulo_articulo, tipo_brecha, vacio_innovacion, doi, fuente]
        print(df.columns)
        print("tipo de brecha, vacio y innovacion: "+fusion_result)
    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False)
        tmp_path = tmp.name

    return FileResponse(tmp_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="resultado_analisis.xlsx")