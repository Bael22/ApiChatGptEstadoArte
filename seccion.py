from tempfile import NamedTemporaryFile
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
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
import requests
import urllib.parse
import requests
from bs4 import BeautifulSoup
from typing import List

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

#Se da el cuartil
def get_journal_quartile(journal_name):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }

    # 1️⃣ Buscar SID
    search_url = (
        "https://www.scimagojr.com/journalsearch.php?q="
        + '+'.join(journal_name.split())
    )
    resp = requests.get(search_url, headers=headers)
    if resp.status_code != 200:
        return f"Error al buscar la revista: {resp.status_code}"

    soup = BeautifulSoup(resp.text, 'html.parser')
    a = soup.select_one("div.search_results a")
    if not a or 'href' not in a.attrs:
        return f"No se encontró la revista '{journal_name}'."

    sid = a['href'].split('=')[1].split('&')[0]

    # 2️⃣ Ir a la página de detalles
    detail_url = (
        f"https://www.scimagojr.com/journalsearch.php?"
        f"clean=0&q={sid}&tip=sid"
    )
    resp2 = requests.get(detail_url, headers=headers)
    if resp2.status_code != 200:
        return f"Error al acceder a detalles: {resp2.status_code}"

    soup2 = BeautifulSoup(resp2.text, 'html.parser')

    # 3️⃣ Buscar la línea "SJR 2024" con cuartil
    text = soup2.get_text(separator=' ')
    m = re.search(r"SJR\s+2024\s+[\d\.]+\s+(Q[1-4])", text)
    if m:
        return  m.group(1)
    if "SJR 2024" not in text:
            print(f"'{journal_name}' no tiene cuartil asignado. Puede que no sea una revista o no esté indexada.")
            return ""
    print(f"No se pudo extraer el cuartil para '{journal_name}'.")  
    return ""

#Se confirma si es de la base de datos de scopus o scienceDirect
def buscar_articulo_elsevier(titulo_busqueda, api_key):
    url = "https://api.elsevier.com/content/search/scopus"  # Buscamos solo en Scopus
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key
    }
    params = {
        "query": f'TITLE("{titulo_busqueda}")',
        "count": 5
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"⚠️ Error en la solicitud: {response.status_code}")
        return None  # None indica que no se pudo verificar

    resultados = response.json().get('search-results', {}).get('entry', [])

    titulo_normalizado = titulo_busqueda.lower().strip()

    for entrada in resultados:
        titulo_resultado = entrada.get('dc:title', '').lower().strip()

        if titulo_resultado == titulo_normalizado:
            base = "Scopus"
            tambien_sciencedirect = False

            # Verificamos si tiene enlace a ScienceDirect
            enlaces = entrada.get('link', [])
            for enlace in enlaces:
                if "sciencedirect.com" in enlace.get('@href', ''):
                    tambien_sciencedirect = True
                    break

            print(f"✅ Artículo encontrado: '{titulo_resultado}'")
            print(f"📚 Base de datos: {base}")
            if tambien_sciencedirect:
                print("🔗 También disponible en: ScienceDirect")

            #return base if not tambien_sciencedirect else f"{base} + ScienceDirect"
            return base
        
    print("❌ Título no encontrado con coincidencia exacta.")
    return None
    
@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analizar/", response_class=HTMLResponse)
async def analizar(request: Request,tema: str = Form(...), pdfs: list[UploadFile] = File(...)):
    resultados = []
    client = OpenAI()
    df = pd.DataFrame(columns=[
        "Nombre del Artículo", "Tipo de Brecha", 
        "Vacío académico y oportunidad de innovación", "DOI","Cuartil (Q)", "Base de datos"
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
            texto_hoja1=(p.pages[0].extract_text() or '')
            texto_posibletitulo = texto_hoja1[:650]
            for i, page in enumerate(p.pages):
                texto = page.extract_text() or ''
                texto_total += texto
        try:
            
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
            # ---------- PRIMER PROMPT:  EXTRACCIÓN DEL TITULO, REVISTA Y GRADO DE RELEVANCIA ----------
            primer_prompt = f"""
            Según el siguiente resumen del artículo:

            {resumen}

            Investigo sobre el tema: "{tema}".

            Evalúa qué tan relacionado está este artículo con ese tema, tomando en cuenta su aporte teórico y/o práctico (se flexible). Clasifica la relevancia como:
            - Muy relevante
            - Relevante
            - Levemente relevante
            - No relevante

            Si es "no relevante", explica brevemente el porqué en 2 oraciones.
            Adicionalmente extrae el titulo y la revista del siguiente recorte del articulo :
            {texto_posibletitulo}
            Entrega una sola versión final en este formato:
            Titulo: ...
            Revista: ...
            Grado de relevancia: ... , porque ...
            """
            response1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": primer_prompt}]
            )
            relevancia = response1.choices[0].message.content.strip().lower()
            print("Relevacia: "+relevancia) 
            titulo_articulo1=re.search(r"(?i)t[ií]tulo:\s*(.+)", relevancia)
            titulo_articulo=titulo_articulo1.group(1).strip() if titulo_articulo1 else "no titulo"
            
            relevancia1=re.search(r"(?i)grado de relevancia:\s*(.+)", relevancia)
            relevancia2=relevancia1.group(1).strip() if relevancia1 else "no relevancia"

            revista1=re.search(r"(?i)revista:\s*(.+)", relevancia)
            revista=revista1.group(1).strip() if revista1 else "no revista"

            #key elsevier
            api_key="dd1491360fe003d3c784213b86dbed30"
            #titulo_pdf = "Using chat GPT to evaluate police threats, risk and harm"
            #print(get_journal_quartile("Theoretical and Applied Mechanics Letters"))
            resultadoBuscaArticulo = buscar_articulo_elsevier(titulo_articulo, api_key)
            cuartil=get_journal_quartile(revista)
            print("Cuartil: "+cuartil)
            if resultadoBuscaArticulo:
                print(f"➡️ Continúa el análisis ({resultadoBuscaArticulo})")
            else:
                print("⛔ Artículo no verificado, deteniendo proceso.")

            if "no relevante" in relevancia2 or not(cuartil=='Q1'or cuartil=='Q2'or cuartil=='Q3'or cuartil=='Q4') or not resultadoBuscaArticulo=='Scopus':
                resultados.append(f"No se pudo seguir analizando el archivo **{pdf.filename}** porque no es relevante para su investigación y el cuartil es {cuartil} y resultadoBuscaArticulo: {resultadoBuscaArticulo} .")
                print("resultador de relevancia: "+ f"No se pudo seguir analizando el archivo **{pdf.filename}** porque no es relevante para su investigación y el cuartil es {cuartil} y resultadoBuscaArticulo: {resultadoBuscaArticulo} .\n{relevancia}")
                continue
            #******************************************************************
            # ---------- EXTRACCIÓN DEL DOI ----------
            doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", texto_hoja1)
            if not doi_match:
                resultados.append(f"El archivo **{pdf.filename}** fue descartado porque no contiene un DOI válido, lo que sugiere que no es un artículo científico.")
                continue
            doi = doi_match.group(1)
            print("DOI: "+doi)
                
        except Exception as e:
            print(f"⚠️ Error al procesar {pdf.filename}: {e}")
            continue
        
        # ---------- EXTRACCIÓN DE CONCLUSIÓN (Vacío académico Y Oportunidad de innovación) Y BRECHA ----------
        texto_ultimas_paginas = ''
        ultima_pagina = min(6, len(p.pages))  # Leer hasta las últimas 6 páginas
        for page in p.pages[-ultima_pagina:]:
            texto_ultimas_paginas += page.extract_text() or ''

        conclusion_match = re.search(r"(Discussion|Summary|Conclusion|Conclusion and future research directions|Conclusión|Discusión)[\s\n:]*([\s\S]{300,4000})", texto_ultimas_paginas, re.IGNORECASE)
        conclusion = conclusion_match.group(2).strip() if conclusion_match else texto_ultimas_paginas[-2000:]
        print("Conclusion: "+conclusion)
        # ---------- 3º PROMPT: Revisión precisa ----------
        tercer_prompt = f"""
        Analiza la conclusión:
         {conclusion}
        
        Considera las siguientes definiciones para elegir el tipo de brecha más adecuado segun las conclusiones (solo 1):

        {brechas_definicion}

        Entrega una sola versión final en este formato (no debe sobrepasar de los 520 caracteres en la respuesta siendo detallado y claro ):
        Tipo de brecha: ...
        Vacío académico: ...
        Oportunidad de innovación: ...
        """
        response3 = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": tercer_prompt}]
        )
        gpt_refinado = response3.choices[0].message.content.strip()

        # Extraemos el tipo de brecha
        tipo_match = re.search(r"(?i)tipo de brecha\s*:\s*(.+)", gpt_refinado)
        tipo_brecha = tipo_match.group(1).strip() if tipo_match else "No identificado"

        # Extraemos todo el contenido de 'Vacío académico y oportunidad de innovación'
        vacio_match = re.search(r"(?i)vac[ií]o acad[eé]mico:\s*(.+)", gpt_refinado)
        oportunidad_match = re.search(r"(?i)oportunidad de innovaci[oó]n:\s*(.+)", gpt_refinado)

        vacio_texto = vacio_match.group(1).strip() if vacio_match else ""
        oportunidad_texto = oportunidad_match.group(1).strip() if oportunidad_match else ""

        # Concatenamos en el formato deseado para el Excel
        vacio_innovacion = f"Vacío académico: {vacio_texto}\nOportunidad de innovación: {oportunidad_texto}"
        
        df.loc[len(df)] = [titulo_articulo, tipo_brecha, vacio_innovacion, doi,cuartil,resultadoBuscaArticulo]
        print("Columnas: "+df.columns)
        print("titulo, tipo de brecha, vacio y innovacion: "+gpt_refinado)
        resultados.append("El articulo "+titulo_articulo+" es de "+revista+" cuartil "+cuartil)
    if df.empty:
        resultados.append("⚠️ No se generó Excel porque no hay datos que mostrar.")
        return JSONResponse(content={"mensajes": resultados})
    
    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False)
        tmp_path = tmp.name
        
    return FileResponse(tmp_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="resultado_analisis.xlsx")
    