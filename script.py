import os
from typing import TypedDict
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Generador de Problemas Matemáticos", layout="wide")

st.markdown(
    """
    <style>
    /* Ocultar barra superior completa (header, share, star, editar, github, menu) */
    header, [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Ocultar pie de página de Streamlit */
    footer {
        display: none !important;
    }
    
    /* Ocultar botón flotante de Manage App de Streamlit Cloud */
    #manage-app-button, 
    [data-testid="manage-app-button"],
    .viewerBadge_container__1QSob,
    div[class*="viewerBadge"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Configuración de Credenciales")
    groq_api_key = st.text_input(
        "Groq API Key:",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        help="Introduce tu clave de API de Groq para ejecutar las cadenas."
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_retries=3,
    groq_api_key=groq_api_key if groq_api_key else "dummy_key_until_input"
)

class MathState(TypedDict):
    nivel_educativo: str
    tema: str
    problema_a: str
    problema_b: str
    problema_c: str

def generar_nivel_a(state: MathState) -> MathState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", r"""Eres un experto en diseño didáctico matemático para {nivel_educativo}.
        Tu tarea es crear un problema de 'Nivel A' sobre el tema proporcionado.
        Reglas estrictas:
        - Enfócate ÚNICAMENTE en la reproducción de conocimientos y aplicación de procedimientos.
        - El estudiante debe poder resolverlo memorizando definiciones, algoritmos, reglas o fórmulas.
        - Debe ser un ejercicio rutinario resuelto mediante la aplicación mecánica de fórmulas.
        - NOTACIÓN MATEMÁTICA: SÍ debes incluir fórmulas, operaciones y ecuaciones matemáticas cuando el ejercicio lo requiera, pero escríbelas en texto estándar legible de teclado (por ejemplo: 7x - 4 = 3(2x + 5) - 9, x^2, a/b). Está estrictamente PROHIBIDO usar sintaxis o comandos de LaTeX (nada de \[, \], \( \), \bigl, \bigr, \frac, ni símbolos $).
        Devuelve únicamente el problema redactado."""),
        ("human", "Tema: {tema}")
    ])
    chain = prompt | llm | StrOutputParser()
    return {"problema_a": chain.invoke({"nivel_educativo": state["nivel_educativo"], "tema": state["tema"]})}

def generar_nivel_b(state: MathState) -> MathState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", r"""Eres un experto en diseño didáctico matemático para {nivel_educativo}.
        Tu tarea es tomar un problema de Nivel A y modificar su redacción para que encaje en el 'Nivel B'.
        Reglas estrictas:
        - Debe implicar una mayor demanda cognitiva.
        - El estudiante ahora debe interpretar información, comparar métodos de resolución, o seleccionar estrategias adecuadas.
        - Relaciona distintas representaciones y establece conexiones entre el lenguaje matemático y el cotidiano.
        - Obliga al alumno a justificar sus decisiones.
        - NOTACIÓN MATEMÁTICA: SÍ debes incluir fórmulas, operaciones y ecuaciones matemáticas cuando el ejercicio lo requiera, pero escríbelas en texto estándar legible de teclado (por ejemplo: 7x - 4 = 3(2x + 5) - 9, x^2, a/b). Está estrictamente PROHIBIDO usar sintaxis o comandos de LaTeX (nada de \[, \], \( \), \bigl, \bigr, \frac, ni símbolos $).
        Devuelve únicamente el problema redactado."""),
        ("human", "Problema Nivel A:\n{problema_a}\n\nModifícalo al Nivel B:")
    ])
    chain = prompt | llm | StrOutputParser()
    return {"problema_b": chain.invoke({"nivel_educativo": state["nivel_educativo"], "problema_a": state["problema_a"]})}

def generar_nivel_c(state: MathState) -> MathState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", r"""Eres un experto en diseño didáctico matemático para {nivel_educativo}.
        Tu tarea es tomar un problema de Nivel B y transformarlo para alcanzar el 'Nivel C' (nivel profundo de comprensión).
        Reglas estrictas:
        - El estudiante debe identificar relaciones entre conceptos matemáticos y formular generalizaciones.
        - Debe requerir que el alumno construya estrategias propias y argumente sus procedimientos.
        - Transfiere los conocimientos a una nueva situación o contexto inusual/complejo.
        - NOTACIÓN MATEMÁTICA: SÍ debes incluir fórmulas, operaciones y ecuaciones matemáticas cuando el ejercicio lo requiera, pero escríbelas en texto estándar legible de teclado (por ejemplo: 7x - 4 = 3(2x + 5) - 9, x^2, a/b). Está estrictamente PROHIBIDO usar sintaxis o comandos de LaTeX (nada de \[, \], \( \), \bigl, \bigr, \frac, ni símbolos $).
        Devuelve únicamente el problema redactado."""),
        ("human", "Problema Nivel B:\n{problema_b}\n\nTransfórmalo al Nivel C:")
    ])
    chain = prompt | llm | StrOutputParser()
    return {"problema_c": chain.invoke({"nivel_educativo": state["nivel_educativo"], "problema_b": state["problema_b"]})}

workflow = StateGraph(MathState)

workflow.add_node("nivel_a", generar_nivel_a)
workflow.add_node("nivel_b", generar_nivel_b)
workflow.add_node("nivel_c", generar_nivel_c)

workflow.add_edge(START, "nivel_a")
workflow.add_edge("nivel_a", "nivel_b")
workflow.add_edge("nivel_b", "nivel_c")
workflow.add_edge("nivel_c", END)

app = workflow.compile()

st.title("Generador Didáctico de Problemas Matemáticos")

opciones_nivel = [
    "alumnos de 1ro de primaria",
    "alumnos de 2do de primaria",
    "alumnos de 3ro de primaria",
    "alumnos de 4to de primaria",
    "alumnos de 5to de primaria",
    "alumnos de 6to de primaria",
    "alumnos de 1ro de secundaria",
    "alumnos de 2do de secundaria",
    "alumnos de 3ro de secundaria",
    "alumnos de 1ro de bachillerato",
    "alumnos de 2do de bachillerato",
    "alumnos de 3ro de bachillerato",
]

col1, col2 = st.columns([1, 2])

with col1:
    nivel_seleccionado = st.selectbox(
        "Nivel educativo:",
        options=opciones_nivel,
        format_func=lambda x: x.replace("alumnos de ", "").title()
    )

with col2:
    tema_input = st.text_input(
        "Tema:",
        placeholder="Ej. problemas sobre ecuaciones lineales"
    )

if st.button("Generar problemas", type="primary"):
    if not groq_api_key.strip():
        st.error("Por favor, ingresa tu API Key de Groq en la barra lateral para continuar.")
    elif not tema_input.strip():
        st.warning("Por favor, ingresa un tema antes de continuar.")
    else:
        inputs = {
            "nivel_educativo": nivel_seleccionado,
            "tema": tema_input.strip()
        }

        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.subheader("Nivel A (Reproducción)")
            area_a = st.empty()
        with col_b:
            st.subheader("Nivel B (Conexiones)")
            area_b = st.empty()
        with col_c:
            st.subheader("Nivel C (Profundización)")
            area_c = st.empty()

        with st.spinner("Ejecutando andamiaje didáctico en LangGraph..."):
            for output in app.stream(inputs):
                for key, value in output.items():
                    if "problema_a" in value:
                        area_a.markdown(value["problema_a"])
                    elif "problema_b" in value:
                        area_b.markdown(value["problema_b"])
                    elif "problema_c" in value:
                        area_c.markdown(value["problema_c"])
