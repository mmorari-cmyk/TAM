import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la interfaz
st.set_page_config(
    page_title="Agente de Circuitos y Dispositivos",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ Agente Experto en Circuitos y Dispositivos")
st.write(
    "Asistente especializado exclusivamente en **Circuitos Eléctricos** y **Dispositivos Electrónicos**. "
    "Cualquier consulta fuera de este dominio será rechazada automáticamente."
)

# 2. Selector de modelo y API Key en la barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    selected_model = st.selectbox(
        "Modelo de Gemini",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"],
        index=0,
        help="Google actualizó sus modelos activos a la serie 2.5 y 2.0.",
    )

api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    with st.sidebar:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Ingresa tu clave de Google AI Studio (aistudio.google.com)",
        )
        st.caption("Obtén tu clave gratuita en [Google AI Studio](https://aistudio.google.com/).")

if not api_key:
    st.info("👋 Ingresa tu **Google Gemini API Key** en la barra lateral para comenzar.")
    st.stop()

# 3. Inicialización del modelo con el modelo seleccionado
@st.cache_resource
def load_llm(key: str, model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        google_api_key=key,
    )

try:
    llm = load_llm(api_key, selected_model)
except Exception as e:
    st.error(f"Error al inicializar Gemini: {e}")
    st.stop()

# 4. Prompt estricto del dominio
SYSTEM_PROMPT = """
Sos un profesor y tutor experto especializado EXCLUSIVAMENTE en Circuitos Eléctricos y Dispositivos Electrónicos.

Tus áreas de conocimiento son únicamente:
- Leyes y teoremas de circuitos (Ley de Ohm, leyes de Kirchhoff LCK/LVK, Thévenin, Norton, Superposición, máxima transferencia de potencia).
- Componentes pasivos (resistencias, condensadores, inductores).
- Dispositivos semiconductores (diodos rectificadores, Zener, transistores BJT en corte/activa/saturación, transistores MOSFET canal N/P).
- Amplificadores operacionales (Op-Amps) y sus configuraciones típicas (inversor, no inversor, seguidor, sumador, diferencial).

REGLA ESTRICTA DE FILTRADO:
- Si el usuario pregunta sobre cualquiera de estos temas, responde con precisión técnica, rigor y de forma didáctica en español.
- Si el usuario pregunta sobre CUALQUIER tema que NO sea de circuitos o dispositivos electrónicos (por ejemplo: fútbol, cocina, viajes, historia, programación general, chistes o preguntas cotidianas), DEBES responder ÚNICAMENTE:
"No puedo responder a esta consulta. Solo estoy capacitado para responder dudas sobre circuitos y dispositivos electrónicos."
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

chain = prompt_template | llm | StrOutputParser()

# 5. Historial de chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "¡Hola! Soy tu asistente de **Circuitos Eléctricos y Dispositivos**. ¿En qué circuito, componente o teorema te puedo colaborar?",
        }
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Entrada del usuario
if user_query := st.chat_input("Escribe tu consulta de circuitos o componentes..."):
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analizando consulta con Gemini..."):
            try:
                response = chain.invoke({"question": user_query})
                st.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error al generar respuesta con Gemini: {e}")
