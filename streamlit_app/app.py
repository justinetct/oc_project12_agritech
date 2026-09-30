"""Point d'entrée Streamlit d'Agritech Answers.

Sous-étape 21.1 : configuration minimale de la page et message
d'accueil. Aucun appel à l'API et aucun style graphique custom :
ces éléments sont ajoutés aux sous-étapes suivantes.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Agritech Answers",
    page_icon="🌿",
    layout="wide",
)

st.title("Agritech Answers")
st.write(
    "Estimez le rendement d'une parcelle ou comparez les cultures possibles "
    "pour votre pays. Choisissez un parcours dans la barre latérale."
)
