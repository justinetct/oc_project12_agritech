"""Observabilité externe optionnelle : configuration de Logfire.

Contrairement à `agritech.monitoring` qui est la source de vérité locale
(SQLite, toujours actif), ce package configure une passerelle vers un
service de traces externe. Il est **optionnel** : sans token, l'API tourne
normalement et n'émet aucune donnée réseau.
"""
