"""Identité visuelle de l'interface : couleurs, logo et feuille de style.

Aucune dépendance à Streamlit : ``page_css()`` renvoie une balise ``<style>``
que la page injecte avec ``st.html``.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

LOGO_WHITE_PATH = Path(__file__).resolve().parent / "assets" / "logo_agritech_white.png"

# Palette de l'interface validée.
COLORS = {
    "forest": "#14452f",      # panneau, boutons, sélection
    "forest-top": "#113b28",  # bandeau, un ton plus sombre que le panneau
    "leaf": "#2c7a53",        # pictogrammes
    "teal": "#2a7c81",        # Recommend : bleu-vert (texte blanc lisible, contraste ≥ 4,5)
    "slate": "#4f6f8f",       # Predict : bleu ardoise désaturé (texte blanc lisible)
    "alert": "#c0532f",       # erreurs et anomalies : orange brique, réservé aux alertes
    "mint-panel": "#d9e8d7",  # cercle du panneau
    "mint-soft": "#a9cbb8",   # textes secondaires sur fond vert
    "cream": "#f7f4ec",       # fond de page
    "warn-bg": "#ecdfbb",
    "warn-line": "#a87925",
    "warn-ink": "#5f430f",
    "card": "#f0f4ec",        # cadres du formulaire
    "card-line": "#dce5d5",
    "field": "#fcfbf7",       # champs, badges, Oui / Non
    "field-line": "#bccfc1",
    "mint": "#e2ede4",        # cercles des pictogrammes du formulaire
    "mint-line": "#d1e2d5",
    "ink": "#1e2a24",
    "ink-soft": "#3a4a43",
    "muted": "#66706a",
    "faint": "#8b938c",
    "rule": "#dfe3d6",
}

_CSS = """
/* Chrome natif de Streamlit : barre d'outils masquée, page sans marges. */
header[data-testid="stHeader"]{display:none}
[data-testid="stMainBlockContainer"]{padding:0 !important;max-width:none !important}
/* Streamlit donne une marge basse négative au Markdown : annulée pour nos blocs HTML. */
[data-testid="stMarkdownContainer"]:has(> .ag-topbar),
[data-testid="stMarkdownContainer"]:has(> .ag-panel),
[data-testid="stMarkdownContainer"]:has(> .ag-rank){margin-bottom:0}
.st-key-notice [data-testid="stAlert"]{margin:32px 48px}

/* Bandeau */
.ag-topbar{display:flex;align-items:center;justify-content:space-between;gap:24px;height:88px;padding:0 48px;
  background:var(--forest-top);box-shadow:inset 0 -1px 0 rgba(255,255,255,.10),0 8px 16px -12px rgba(0,0,0,.45);
  position:relative;z-index:1}
.ag-logo{display:block;height:48px;width:auto}
.ag-nav{display:flex;align-items:center;gap:6px}
.ag-nav-item{display:inline-flex;align-items:center;gap:8px;height:42px;padding:0 20px;border-radius:999px;
  color:rgba(255,255,255,.80) !important;font-weight:600;font-size:17px;text-decoration:none !important}
.ag-nav-item svg{width:18px;height:18px}
.ag-nav-item.active{background:var(--cream);color:var(--forest) !important}

/* Ossature : panneau vert à gauche, formulaire à droite (colonnes principales seulement,
   pas les colonnes imbriquées du formulaire). */
.st-key-shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]{gap:0;align-items:stretch}
.st-key-shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child{background:var(--forest);padding:56px 48px 40px}
.st-key-shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child{padding:56px 48px 36px}

/* Panneau */
.ag-panel{display:flex;flex-direction:column;min-height:calc(100vh - 88px - 96px);color:#fff}
.ag-eyebrow{font-size:13px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--mint-soft)}
.ag-big-ic{width:108px;height:108px;border-radius:50%;background:var(--mint-panel);color:var(--forest);
  display:flex;align-items:center;justify-content:center;margin-top:32px}
.ag-big-ic svg{width:52px;height:52px}
.ag-panel-title{margin-top:40px;font-size:50px;line-height:1.03;letter-spacing:-.035em;font-weight:700}
.ag-panel-lead{margin-top:22px;font-size:18px;line-height:1.6;color:rgba(255,255,255,.82);max-width:34ch}
/* Recommend : cultures évaluées, simple ligne d'information (rien de cliquable) */
.ag-panel-crops{margin-top:28px;max-width:40ch;font-size:15px;line-height:1.6;color:rgba(255,255,255,.68)}
.ag-panel-crops b{font-weight:600;color:rgba(255,255,255,.88)}
.ag-res-label{margin-top:36px;font-size:22px;font-weight:600}
.ag-res-value{margin-top:4px;font-size:100px;font-weight:700;line-height:1;letter-spacing:-.045em;white-space:nowrap}
.ag-res-value small{font-size:28px;font-weight:600;letter-spacing:0;margin-left:10px;color:var(--mint-soft)}
.ag-warn{display:flex;gap:12px;margin-top:28px;padding:16px 18px;border-left:4px solid var(--warn-line);
  border-radius:10px;background:var(--warn-bg);color:var(--warn-ink)}
.ag-warn > svg{width:21px;height:21px;flex:none;margin-top:1px;color:var(--warn-line)}
.ag-warn-title{font-weight:700;font-size:16px}
.ag-warn ul{margin:6px 0 0;padding-left:18px;font-size:15.5px;line-height:1.55}
.ag-warn li{margin:0;padding:0}
.ag-warn-text{margin-top:8px;font-size:15.5px;line-height:1.5}
.ag-panel-foot{margin-top:auto;padding-top:22px;border-top:1px solid rgba(255,255,255,.14);font-size:15.5px;
  line-height:1.6;color:rgba(255,255,255,.86)}

/* Formulaire : cadre vert très pâle des conditions */
.st-key-cond_card{background:var(--card);border:1px solid var(--card-line);border-radius:16px;
  padding:32px 28px 36px}
/* Titre, libellés et domaines du cadre sans la marge basse négative du Markdown de Streamlit :
   le gap du cadre et celui des colonnes s'appliquent normalement. */
.st-key-cond_card [data-testid="stMarkdownContainer"]:has(> .ag-sec-head),
.st-key-cond_card [data-testid="stMarkdownContainer"]:has(> .ag-field-label),
.st-key-cond_card [data-testid="stMarkdownContainer"]:has(> .ag-help){margin-bottom:0}
.ag-sec-title{font-size:26px;font-weight:700;letter-spacing:-.02em;line-height:1.2;color:var(--forest)}
.ag-step{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;margin-right:12px;
  border-radius:50%;background:var(--forest);color:#fff;font-size:17px;font-weight:700;letter-spacing:0;
  vertical-align:3px}
.ag-sec-sub{margin-top:3px;font-size:16px;color:var(--muted)}
.ag-field-label{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:600;color:var(--ink);
  white-space:nowrap}
.ag-ic{width:36px;height:36px;border-radius:50%;background:var(--mint);box-shadow:inset 0 0 0 1px var(--mint-line);
  color:var(--leaf);display:inline-flex;align-items:center;justify-content:center;flex:none}
.ag-ic svg{width:54%;height:54%}
.ag-ic.lg{width:44px;height:44px}

/* Oui / Non */
.st-key-cond_card button[data-variant="segmented_control"]{background:var(--field);color:var(--muted);font-weight:600}
.st-key-cond_card button[data-variant="segmented_control"] p{font-size:16px}
.st-key-cond_card button[data-variant="segmented_control"][aria-checked="true"]{background:var(--forest);
  border-color:var(--forest);color:#fff}

/* Champs numériques : sans boutons +/-, unité à droite */
.st-key-cond_card [data-testid="stNumberInputStepDown"],
.st-key-cond_card [data-testid="stNumberInputStepUp"]{display:none}
.st-key-cond_card [data-testid="stNumberInputContainer"]{background:var(--field);border:1px solid var(--field-line);
  border-radius:8px}
.st-key-cond_card [data-testid="stNumberInputContainer"]::after{align-self:center;padding:0 16px;font-size:15px;
  color:var(--muted)}
.st-key-cond_card input{font-size:20px;font-weight:600}
/* Curseur synchronisé, discret sous son champ : ni valeur au-dessus de la poignée ni bornes */
.st-key-cond_card .stElementContainer[class*="_slider"]{margin-top:-6px;margin-bottom:-6px}
.st-key-cond_card [data-testid="stSliderThumbValue"],
.st-key-cond_card [data-testid="stSliderTickBar"]{display:none}
.ag-help{margin-top:-6px;font-size:14px;color:var(--faint)}
.st-key-cond_card .ag-help{margin-top:0}
/* Plage d'un domaine (« 2,6 – 30,2 °C ») jamais coupée : elle passe entière à la ligne */
.ag-range{white-space:nowrap}

/* Bouton principal, note et pied */
[data-testid="stBaseButton-primary"]{min-height:54px;border-radius:8px;font-size:17.5px;font-weight:700}
.ag-cta-note{text-align:center;font-size:14.5px;color:var(--muted)}
.ag-foot{display:flex;justify-content:space-between;gap:12px;margin-top:22px;padding-top:18px;
  border-top:1px solid var(--rule);font-size:14.5px;color:var(--muted)}

/* Recommend : cadres du pays et du classement, comme ceux du formulaire */
.st-key-country_card,.st-key-ranking_card{background:var(--card);border:1px solid var(--card-line);border-radius:16px;
  padding:24px 28px 28px}
/* Sélecteur de pays (Streamlit 1.59 : ComboBox react-aria, sans attribut data-baseweb) */
.st-key-rec_country [role="group"]{min-height:52px;background:var(--field);border:1px solid var(--field-line);
  border-radius:8px}
.st-key-rec_country [role="group"]:focus-within{border-color:var(--forest)}
/* Tant qu'aucun pays n'est choisi (placeholder affiché), le sélecteur est mis en avant. */
.st-key-rec_country:has(input:placeholder-shown) [role="group"]{border:2px solid var(--forest)}
.st-key-rec_country input{font-size:18px;font-weight:600;color:var(--ink)}
/* Saisie libre du sélecteur de pays : l'entrée « Add: … » de Streamlit, en anglais, devient
   « Rechercher ce pays » (seul sélecteur de l'application qui accepte un texte libre). */
[role="option"][data-key="__creatable__"] [data-item-hl]{position:relative;visibility:hidden}
[role="option"][data-key="__creatable__"] [data-item-hl]::before{content:"Rechercher ce pays";position:absolute;
  visibility:visible;white-space:nowrap}

/* Recommend : classement des cultures, dans l'ordre de l'API */
.ag-rank{margin-top:10px}
.ag-rank-row{display:grid;grid-template-columns:34px minmax(0,2fr) minmax(0,1fr) 116px 250px;align-items:center;
  gap:12px;padding:10px 0;border-top:1px solid var(--card-line)}
.ag-rank-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:var(--mint);box-shadow:inset 0 0 0 1px var(--mint-line);color:var(--forest);font-size:15.5px;font-weight:700}
.ag-rank-row:first-child .ag-rank-num{background:var(--forest);box-shadow:none;color:#fff}
.ag-rank-crop{font-size:17px;font-weight:600;color:var(--ink)}
.ag-rank-bar{height:10px;border-radius:999px;background:var(--mint)}
.ag-rank-bar span{display:block;height:100%;border-radius:999px;background:var(--leaf)}
.ag-rank-value{text-align:right;font-size:17px;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;
  white-space:nowrap}
.ag-badge{justify-self:start;display:inline-block;padding:3px 10px;border-radius:999px;font-size:13.5px;font-weight:600;
  white-space:nowrap}
.ag-badge.observed{background:var(--mint);box-shadow:inset 0 0 0 1px var(--mint-line);color:var(--forest)}
.ag-badge.unobserved{box-shadow:inset 0 0 0 1px var(--field-line);color:var(--muted)}

/* Recommend : culture classée première et conditions utilisées, dans le panneau */
.ag-top-crop{margin-top:6px;font-size:46px;font-weight:700;line-height:1.05;letter-spacing:-.03em;text-wrap:balance}
.ag-top-yield{margin-top:12px;font-size:34px;font-weight:700;letter-spacing:-.02em;white-space:nowrap}
.ag-top-yield small{margin-left:8px;font-size:20px;font-weight:600;color:var(--mint-soft)}
.ag-panel .ag-badge{margin-top:14px;align-self:flex-start}
.ag-panel .ag-badge.observed{background:rgba(255,255,255,.14);box-shadow:none;color:#fff}
.ag-panel .ag-badge.unobserved{box-shadow:inset 0 0 0 1px rgba(255,255,255,.35);color:var(--mint-soft)}
.ag-cond{margin-top:26px;padding-top:16px;border-top:1px solid rgba(255,255,255,.14)}
.ag-cond-title{font-size:13px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--mint-soft)}
.ag-cond-row{display:flex;justify-content:space-between;gap:12px;margin-top:8px;font-size:16.5px;
  color:rgba(255,255,255,.82)}
.ag-cond-row b{color:#fff;font-variant-numeric:tabular-nums;white-space:nowrap}

/* Formulaire trop étroit pour les 5 colonnes du classement : culture et rendement, puis barre, puis badge */
@media (max-width:1200px){
  .ag-rank-row{grid-template-columns:30px minmax(0,1fr) auto;grid-template-areas:"num crop value" "num bar bar" "num badge badge";
    row-gap:6px}
  .ag-rank-num{grid-area:num;align-self:start}
  .ag-rank-crop{grid-area:crop}
  .ag-rank-value{grid-area:value}
  .ag-rank-bar{grid-area:bar}
  .ag-rank-row .ag-badge{grid-area:badge}
}

/* Panneau étroit : titre réduit */
@media (max-width:1100px){
  .ag-panel-title{font-size:38px}
}

/* Écrans étroits (Streamlit empile ses colonnes sous 640 px) : panneau au-dessus du formulaire */
@media (max-width:640px){
  .ag-topbar{height:auto;padding:14px 18px;gap:12px}
  .ag-logo{height:36px}
  .ag-nav-item{height:36px;padding:0 14px;font-size:15px}
  .st-key-shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child{padding:30px 20px 32px}
  .st-key-shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child{padding:30px 20px 28px}
  .ag-panel{min-height:0}
  .ag-big-ic{width:76px;height:76px;margin-top:20px}
  .ag-big-ic svg{width:38px;height:38px}
  .ag-panel-title{font-size:36px;margin-top:26px}
  .ag-res-value{font-size:76px}
  .ag-panel-foot{display:none}
  .st-key-cond_card{padding:24px 16px 28px}
  .st-key-cond_card [data-testid="stHorizontalBlock"]{row-gap:48px}
  .ag-foot{flex-direction:column;gap:6px}
  .st-key-country_card,.st-key-ranking_card{padding:18px 16px}
  .ag-top-crop{font-size:36px}
}
"""


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """Logo blanc (fond transparent) encodé pour une balise ``<img>``."""
    encoded = base64.b64encode(LOGO_WHITE_PATH.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def page_css() -> str:
    """Feuille de style complète : variables de couleur puis règles."""
    variables = "".join(f"--{name}:{value};" for name, value in COLORS.items())
    return f"<style>:root{{{variables}}}{_CSS}</style>"


# Spécificités du cadre des conditions de Predict, injectées par cette page seulement.
_PREDICT_FORM_CSS = """
/* Libellés des 4 conditions un peu plus présents ; pictogrammes inchangés */
.st-key-cond_card .ag-field-label{font-size:18px}
/* Filet discret entre conditions climatiques (1re ligne) et pratiques agricoles (2e ligne) */
.st-key-cond_card > [data-testid="stLayoutWrapper"] + [data-testid="stLayoutWrapper"]{padding-top:32px;
  border-top:1px solid var(--card-line)}
"""


def predict_form_css() -> str:
    """Spécificités du formulaire Predict, à injecter après ``page_css()``."""
    return f"<style>{_PREDICT_FORM_CSS}</style>"
