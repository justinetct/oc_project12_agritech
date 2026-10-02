"""Point d'entrée de l'application Agritech Answers.

Déclare les pages ; chaque page dessine son propre bandeau de navigation, la
navigation native de Streamlit est donc masquée.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent / "views"

st.set_page_config(page_title="Agritech Answers", page_icon="🌿", layout="wide")

page = st.navigation(
    [
        st.Page(PAGES_DIR / "predict.py", title="Predict", default=True),
        st.Page(PAGES_DIR / "recommend.py", title="Recommend"),
    ],
    position="hidden",
)
page.run()
