# Agente Experto en Circuitos y Dispositivos Electrónicos

Aplicación interactiva desarrollada en **Streamlit** y basada en **Google Gemini (LangChain)**, diseñada para desplegarse de forma directa y gratuita en **Streamlit Cloud** o ejecutarse localmente.

El agente está restringido mediante un *System Prompt* estricto para responder exclusivamente sobre circuitos eléctricos y dispositivos semiconductores.

## Estructura del Proyecto

```text
CIRCUITOSyDISPOSITIVOS/
├── app.py              # Aplicación Streamlit completa con el agente Gemini
├── requirements.txt    # Dependencias mínimas listas para Streamlit Cloud
├── .gitignore          # Exclusiones de Git
└── README.md           # Instrucciones de uso y despliegue
```

## Requisitos de Ejecución

1. Obtener una clave gratuita de API en [Google AI Studio](https://aistudio.google.com/).
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución Local

```bash
streamlit run app.py
```
*Puedes ingresar tu API Key directamente en la barra lateral de la app o configurarla como variable de entorno `export GEMINI_API_KEY="tu_clave"`.*

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a tu cuenta de GitHub.
2. Ingresa a [share.streamlit.io](https://share.streamlit.io/) y selecciona tu repositorio.
3. En **Advanced Settings > Secrets**, añade tu clave:
   ```toml
   GEMINI_API_KEY = "tu_clave_de_google_ai_studio"
   ```
4. Haz clic en **Deploy**.
