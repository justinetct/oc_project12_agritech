"""Pictogrammes Agritech Answers, recopiés des symboles ``ico-*`` des slides.

Chaque entrée est le contenu d'un SVG 24 × 24 dessiné au trait avec
``currentColor`` : la couleur vient du CSS. ``alerte`` est la seule icône
absente des slides ; elle est dessinée dans le même style.
"""

from __future__ import annotations

ICONS: dict[str, str] = {
    "ble": (
        '<g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21V7.5"/><path d="M12 8.2c0-2.3 1.1-4.1 3-5.2.7 2.7-.2 4.8-3 5.2z"/><path d="M12 8.2c0-2.3-1.1-4.1-3-5.2-.7 2.7.2 4.8 3 5.2z"/><path d="M12 12.7c0-2.3 1.1-4.1 3-5.2.7 2.7-.2 4.8-3 5.2z"/><path d="M12 12.7c0-2.3-1.1-4.1-3-5.2-.7 2.7.2 4.8 3 5.2z"/><path d="M12 17.2c0-2.3 1.1-4.1 3-5.2.7 2.7-.2 4.8-3 5.2z"/><path d="M12 17.2c0-2.3-1.1-4.1-3-5.2-.7 2.7.2 4.8 3 5.2z"/></g>'
    ),
    "rendement": (
        '<g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18"/><rect x="4.6" y="13.2" width="3.9" height="7.8" rx="1"/><rect x="10.1" y="9.2" width="3.9" height="11.8" rx="1"/><rect x="15.6" y="4.6" width="3.9" height="16.4" rx="1"/></g>'
    ),
    "pousse": (
        '<g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21v-8"/><path d="M12 13c0-3.6-2.6-6.2-6.5-6.5C5.2 10.5 8 13.4 12 13z"/><path d="M12 13.5c0-4 2.9-7 7-7.3.3 4-2.6 7.3-7 7.3z"/><path d="M6.5 21h11"/></g>'
    ),
    "goutte": (
        '<g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3s6 6.6 6 11a6 6 0 0 1-12 0c0-4.4 6-11 6-11z"/></g>'
    ),
    "thermo": (
        '<g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.8V5a2 2 0 1 0-4 0v9.8a4 4 0 1 0 4 0z"/><path d="M12 9v7"/></g>'
    ),
    "engrais": (
        '<g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M7.6 7h8.8a1 1 0 0 1 .99 1.16l-1 6.1A2.2 2.2 0 0 1 14.2 16H9.8a2.2 2.2 0 0 1-2.17-1.74l-1-6.1A1 1 0 0 1 7.6 7z"/><path d="M9.3 7 10.2 3.6h3.6L14.7 7"/><path d="M8.8 19.6h.01"/><path d="M12 20.8h.01"/><path d="M15.2 19.2h.01"/></g>'
    ),
    "irrigation": (
        '<g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21v-7"/><path d="M7.6 21h8.8"/><path d="M12 14c0-2.7 2.2-4.9 4.9-4.9"/><path d="M12 14c0-2.7-2.2-4.9-4.9-4.9"/><path d="M12 14V8.2"/><circle cx="12" cy="5.6" r="1.6"/></g>'
    ),
    "podium": (
        '<g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21V9.5h6V21"/><path d="M3 21v-6h6v6"/><path d="M15 21v-8.5h6V21"/><path d="M3 21h18"/></g>'
    ),
    "pesticide": (
        '<g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9.8 2.8h4.4"/><path d="M10.6 2.8v5.6L6.2 17.6a2.1 2.1 0 0 0 1.9 3h7.8a2.1 2.1 0 0 0 1.9-3L13.4 8.4V2.8"/><path d="M8.4 14.2h7.2"/></g>'
    ),
    "alerte": (
        '<g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.8"/><path d="M12 7.6v5.6"/><path d="M12 16.5h.01"/></g>'
    ),
}

def icon_svg(name: str) -> str:
    """Balise ``<svg>`` en ligne pour l'icône ``name`` (``KeyError`` si inconnue)."""
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{ICONS[name]}</svg>'
