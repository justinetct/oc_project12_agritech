# Rapport technique — Agritech Answers

*Système de prédiction de rendement et de recommandation de cultures.*

> [!NOTE]
> Ce rapport couvre la préparation des données : contexte, exploration, ACP, consolidation
> des sources et jeux d'entraînement. La modélisation, le suivi des expériences, l'API et le
> déploiement seront ajoutés dans les prochaines versions du rapport. Aucun résultat de modèle
> n'est donné ici.

## Sommaire

1. [Objectifs du projet](#1-objectifs-du-projet)
2. [Données disponibles](#2-données-disponibles)
3. [Analyse du dataset Agriculture CropYield](#3-analyse-du-dataset-agriculture-cropyield)
4. [Analyse du dataset CropYield Prediction](#4-analyse-du-dataset-cropyield-prediction)
5. [Comparaison des deux datasets](#5-comparaison-des-deux-datasets)
6. [Préparation des données d'entraînement](#6-préparation-des-données-dentraînement)
7. [Limites identifiées](#7-limites-identifiées)
8. [Suite du projet](#8-suite-du-projet)

## 1. Objectifs du projet

Agritech Answers veut proposer aux agriculteurs une application web qui répond à deux
questions. Dans la première, la culture est déjà choisie. Dans la seconde, c'est justement le
choix de la culture qui est en jeu. La différence a l'air minime, mais c'est elle qui explique
toute l'organisation des données décrite ici.

### `/predict` — estimer un rendement

L'utilisateur choisit une culture et décrit les conditions de sa parcelle : pluie de la
saison, température moyenne, type de sol, engrais et irrigation. Le service renvoie une
estimation du rendement, en tonnes par hectare.

On raisonne à l'échelle d'**une parcelle, sur une saison**.

### `/recommend` — classer les cultures

L'utilisateur indique son pays. L'application récupère le contexte historique de ce pays,
estime le rendement de chaque culture possible, puis renvoie un classement par rendement
décroissant. La culture recommandée est donc celle dont le rendement prédit est le plus élevé.

Ici, il faut **comparer les cultures entre elles**.

### Pourquoi deux jeux de données et deux modèles

Pour classer des cultures, il faut que la culture change le rendement prédit. Sinon le
classement ne veut rien dire. Or dans le jeu de données parcellaire, les six cultures donnent
le même rendement à conditions comparables : 0,03 t/ha d'écart au maximum, à température
comparable (section 3). Ce jeu convient donc à `/predict`, mais pas à un classement.

Le second jeu, historique et à l'échelle du pays, sépare bien mieux les cultures : jusqu'à
3,3 t/ha d'écart entre elles à climat comparable (section 5). En revanche, il ne décrit aucune
parcelle.

Chaque service utilise donc le jeu qui répond à sa question, et les lignes des deux jeux ne
sont jamais mises bout à bout.

## 2. Données disponibles

| | **Agriculture CropYield** | **CropYield Prediction** |
|---|---|---|
| Une ligne | une parcelle sur une saison | un pays × une année × une culture |
| Volume brut | 1 000 000 lignes × 10 colonnes | 4 fichiers sources + une fusion fournie |
| Période | pas de dimension temporelle | 1990-2013 après consolidation |
| Géographie | 4 zones abstraites (North/East/South/West) | 168 pays après harmonisation géographique |
| Cultures | 6 | 10 |
| Variables clés | pluie, température, sol, engrais, irrigation | rendement, température, pluie, pesticides |
| Cible | `Yield_tons_per_hectare` (t/ha) | `yield_t_ha` (t/ha, converti depuis hg/ha) |
| Usage retenu | **`/predict`** | **`/recommend`** |

Le second jeu arrive en cinq fichiers : quatre sources — `yield.csv`, `temp.csv`,
`rainfall.csv`, `pesticides.csv` — et `yield_df.csv`, qui est une fusion déjà faite des quatre
autres. Cette fusion n'est pas utilisée ; la section 4 explique pourquoi elle a été refaite.

Aucune clé ne relie les deux jeux. Le premier n'a ni pays ni année, et sa colonne `Region` ne
contient que North, East, South et West, qui ne correspondent à aucun territoire réel. Seule
la culture est commune aux deux jeux, et encore : 4 valeurs sur 6. Mettre les lignes bout à
bout supposerait que les deux jeux mesurent la même chose à la même échelle. Ce n'est le cas
ni pour la pluie, ni pour la température, ni pour le rendement.

## 3. Analyse du dataset Agriculture CropYield

**1 000 000 de lignes et 10 colonnes.** Les types sont corrects et aucune valeur manquante ni
doublon n'est détecté.

Côté distributions, `Rainfall_mm`, `Temperature_Celsius` et `Days_to_Harvest` sont uniformes,
et la cible suit une courbe en cloche centrée sur 4,65 t/ha. Ces distributions ne présentent
pas d'asymétrie marquée : aucune transformation particulière n'est prévue à ce stade.

### Variables les plus liées au rendement

| Variable | Corrélation | Relation observée |
|---|---|---|
| `Rainfall_mm` | **0,765** | la plus liée au rendement ; relation droite, dispersion constante |
| `Fertilizer_Used` | **0,442** | **+1,50 t/ha** en moyenne |
| `Irrigation_Used` | **0,354** | **+1,20 t/ha** en moyenne |
| `Temperature_Celsius` | 0,086 | +0,445 t/ha entre les tranches extrêmes, sans reculer |
| `Days_to_Harvest` | −0,003 | aucune relation visible avec le rendement (0,029 t/ha d'amplitude) |

![Relation pluie / rendement et effet des pratiques agricoles](assets/figures/01_pluie_rendement.png)

Les écarts entre les groupes restent similaires : sur le graphique de droite, les courbes sont
presque parallèles. Les variables d'entrée ne sont pas non plus corrélées entre elles
(|r| ≤ 0,003), donc chaque effet se lit séparément.

La température demande un mot de plus. Avec une corrélation de 0,086, on pourrait l'écarter.
Mais en découpant en tranches, le rendement moyen monte de 4,423 à 4,867 t/ha du plus froid au
plus chaud, sans jamais reculer. L'effet est petit comparé à la dispersion générale, mais il
est régulier. On garde donc la variable.

### Variables avec peu ou pas d'effet visible

`Region`, `Soil_Type`, `Crop` et `Weather_Condition` sont réparties de façon équilibrée et ne
montrent pas de différence notable sur le rendement : **0,012 t/ha d'écart au maximum** entre
modalités, soit 0,2 %.
À pluie comparable, les six cultures restent à 0,028 t/ha les unes des autres, et le gain
apporté par l'engrais va de +1,494 à +1,514 selon la culture. Les cultures ne se distinguent
donc pas davantage dans un contexte particulier.

`Weather_Condition` n'a pas non plus de lien avec les mesures : la pluie moyenne vaut 549,5,
550,1 et 550,3 mm selon que la météo est déclarée ensoleillée, nuageuse ou pluvieuse.

> [!IMPORTANT]
> **`/recommend` ne peut pas s'appuyer sur ce jeu de données.** Classer six cultures qui
> donnent le même rendement reviendrait à sortir un ordre au hasard. C'est ce constat qui nous
> a poussés à explorer le second jeu.

### Qualité de la cible

Le rendement va de −1,15 à 9,96 t/ha, avec une moyenne et une médiane à 4,65. **231 lignes
(0,023 %) ont un rendement négatif**, ce qui n'a pas de sens physique : toutes sont sans
engrais ni irrigation, et 223 d'entre elles ont reçu moins de 200 mm de pluie. Ce sont des
valeurs très basses et très rares. Ce qu'on en fait est décidé à la section 6.

### Analyse en composantes principales

L'ACP permet de voir si plusieurs variables apportent une information similaire et peuvent être
regroupées. L'objectif est de vérifier si on peut simplifier les données en utilisant moins de
variables, sans perdre trop d'information.

L'ACP porte sur les six variables numériques, centrées-réduites, sur l'ensemble des lignes.
Les quatre variables texte sont laissées de côté : leur donner des numéros créerait des écarts
qui n'existent pas.

| Axe | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| Inertie (%) | **32,58** | 16,71 | 16,68 | 16,67 | 16,62 | 0,74 |
| Cumul (%) | 32,58 | 49,29 | 65,97 | 82,64 | 99,26 | 100,00 |

Un seul axe se détache. F2 à F5 sont pratiquement au même niveau, autour de 16,7 % chacun :
aucun ne ressort. Il faut 5 axes sur 6 pour couvrir 99 % de l'information.

![Cercle des corrélations et plan factoriel coloré par culture](assets/figures/02_acp.png)

F1 regroupe le rendement (0,99), la pluie (0,79), l'engrais (0,46) et l'irrigation (0,37),
tous dans le même sens : c'est l'axe du rendement, et il redit ce qu'on a vu plus haut.
`Days_to_Harvest` reste à l'écart (−0,01 sur F1) et se retrouve seule sur F4 (0,91).

Deux conclusions comptent pour la suite.

**L'ACP ne permet pas de réduire le nombre de variables.** Refaite sans la cible — c'est le
cas réel, garder le rendement ferait entrer dans les composantes ce qu'on cherche à prédire —
chaque composante explique environ 20 % de la variance et aucune ne domine. Les cinq variables
explicatives sont peu corrélées entre elles : les regrouper ferait perdre de l'information. On
ne l'utilise donc pas comme prétraitement.

**Les cultures ne se séparent pas sur les plans factoriels.** Leurs coordonnées moyennes
diffèrent de 0,010 au maximum sur F1, alors que les individus ont un écart-type de 1,398 sur
ce même axe. Le nuage de droite le montre : les six couleurs se recouvrent entièrement.

## 4. Analyse du dataset CropYield Prediction

### Pourquoi refaire la fusion fournie

`yield_df.csv` assemble déjà les quatre sources. Deux défauts nous ont fait renoncer à
l'utiliser.

Le premier est une **multiplication des lignes lors de la jointure avec la température** (un
fan-out). La clé `(country, year)` de `temp.csv` n'est pas unique : un même couple pays / année
peut compter jusqu'à **52 lignes** (41 valeurs différentes une fois les doublons exacts
retirés). Si on joint sans moyenner d'abord, chaque rendement est recopié autant de fois. Les
pays concernés pèsent alors plus lourd que les autres à l'entraînement, et un découpage
aléatoire mettrait des copies d'une même ligne dans le train et dans le test — donc une fuite
de données.

Le second est une **perte de pays à la jointure**. `yield_df.csv` ne retient que 99 pays
cartographiables, alors que les quatre sources en ont **117** en commun. Une partie de cette
perte de 18 pays vient de différences de libellés, y compris pour de gros producteurs —
`United States of America` contre `United States`, `Russian Federation` contre `Russia`,
`China, mainland` contre `China`.

La fusion a donc été refaite à partir des quatre sources. Les noms de pays ont d'abord été
harmonisés vers des **codes ISO3**, ajoutés pendant la préparation, puis les jointures ont été
faites sur ces codes plutôt que sur les libellés d'origine.

### Les quatre sources et leur préparation

| Source | Granularité | Période | Préparation |
|---|---|---|---|
| `yield.csv` | pays × culture × année | 1961-2016 | conversion hg/ha → t/ha ; `China, mainland` retenu |
| `temp.csv` | pays × année × relevé | 1743-2013 | doublons exacts retirés, puis moyenne par pays et année |
| `rainfall.csv` | pays × année | 1985-2017 | en-tête nettoyé, conversion numérique forcée |
| `pesticides.csv` | pays × année | 1990-2016 | rien à faire, la clé est déjà unique |

La période retenue, **1990-2013**, est ce que les quatre sources ont en commun :
`pesticides.csv` fixe la borne basse, `temp.csv` la borne haute.

Il a fallu trancher pour la Chine. `yield.csv` propose deux entités qui reçoivent le même
code ISO3 : `China` (avec Taïwan, Hong Kong et Macao) et `China, mainland` (Chine
continentale). Les trois autres sources traitent ces territoires à part, et `pesticides.csv`
n'a même pas de libellé `China`. On garde donc `China, mainland`, pour joindre des territoires
comparables. Sur les 448 couples (culture, année) où les deux entités coexistent, l'écart
relatif moyen n'est que de 0,26 %.

### Le dataset consolidé

La clé `iso3 + year + crop` est unique, et le nombre de lignes ne bouge pas au fil des trois
jointures : **aucun fan-out**.

| Caractéristique | Valeur |
|---|---|
| Lignes | **22 679** |
| Pays | **168** |
| Cultures | **10** |
| Période | **1990-2013** (24 années) |
| Colonnes | `iso3`, `area`, `year`, `crop`, `yield_t_ha`, `avg_temp`, `rain_mm`, `pesticides_t` |

Les valeurs manquantes sont gardées telles quelles — température 19,5 %, pesticides 13,1 %,
pluie 5,0 %. Les manquants de température et de pesticides correspondent surtout à des pays non
couverts par ces sources. **69,1 % des lignes sont complètes, pour 117 pays** : n'exiger que des
lignes complètes ferait passer de 168 à 117 pays.

### Évolution du rendement par culture

Le dataset couvre 24 années. On regarde donc comment le rendement de chaque culture évolue sur
la période, pour savoir si les écarts entre cultures sont stables ou s'ils changent d'une année
à l'autre.

Les cultures sont affichées sur deux graphiques parce que leurs niveaux de rendement sont très
différents. Sur une seule échelle, les céréales et les légumineuses seraient tassées en bas du
graphique et leurs évolutions deviendraient illisibles. Ce découpage sert uniquement à la
lecture.

![Évolution du rendement moyen par culture, 1990-2013](assets/figures/03_evolution_rendements.png)

- Les niveaux restent très différents d'une culture à l'autre : les moyennes annuelles vont de
  7,8 à 19,7 t/ha pour les tubercules et le plantain, contre 1,4 à 4,8 t/ha pour les céréales
  et les légumineuses.
- Les dix cultures progressent entre 1990 et 2013.
- Ces écarts se retrouvent sur toute la période : ce n'est pas l'effet d'une seule année.

Ce sont ces différences entre cultures qui manquent au jeu parcellaire, et c'est ce qui conduit
à utiliser l'historique pour comparer les cultures dans `/recommend`. La figure montre que
l'information est présente dans les données ; elle ne dit rien de ce qu'un modèle saura en
faire.

### Trois pièges rencontrés

**La pluie n'est pas une mesure annuelle.** `rain_mm` est identique sur les 24 années pour
chaque pays qui en a une : c'est une normale climatique répétée. Elle décrit le climat habituel
d'un pays, pas la météo d'une année, et ne peut donc montrer aucune évolution dans le temps.

**Le nombre de pays change d'une année à l'autre.** Il passe de 138 pays en 1990 à 168 en 2013,
avec un saut net en 1992. Du coup, la température moyenne brute *baisse* sur la période
(20,9 → 20,2 °C) alors qu'à panel constant elle augmente de +0,44 °C : c'est le changement de
pays observés qui explique la baisse, pas le climat. Les hausses de rendement, elles, restent
visibles à panel constant — maïs +47,2 % brut contre +41,6 %, blé +22,9 % contre +19,1 %.

**Les valeurs manquantes concernent des pays entiers.** `avg_temp` manque pour 36 pays et
`pesticides_t` pour 24, jamais sur quelques années seulement. On ne peut donc rien compléter à
partir des données ; la conséquence est traitée à la section 6.

## 5. Comparaison des deux datasets

Quatre cultures sont communes aux deux jeux — maïs, riz, soja, blé — après avoir uniformisé
deux libellés. C'est sur elles qu'on peut comparer : 11 044 lignes côté historique, 666 638
côté parcellaire.

### Trois variables comparables, trois lectures différentes

| Variable | Agriculture CropYield | Historique consolidé |
|---|---|---|
| Rendement | −1,15 à 9,96 t/ha, médiane 4,65 | 0 à 36,8 t/ha, médiane 2,22 |
| Température | 15 à 40 °C, médiane 27,5 | −3,4 à 30,4 °C, médiane 19,5 |
| Pluie | 100 à 1 000 mm sur la saison | 51 à 3 240 mm/an, normale climatique |
| Corrélation pluie / rendement | **+0,764** | **−0,161** |
| Corrélation température / rendement | +0,085 | −0,267 |

*Valeurs calculées sur les quatre cultures communes.*

Les corrélations changent de signe. Ce n'est pas contradictoire : elles ne portent pas sur la
même chose. Côté parcellaire, une ligne est une parcelle, et plus il pleut, plus le rendement
monte. Côté historique, une ligne est un pays et une année : le signe négatif vient d'écarts
*entre pays*, où les plus arrosés n'ont pas les meilleurs rendements. Ce n'est pas un effet de
la pluie sur une parcelle. L'historique ne peut donc ni confirmer ni contredire ce qu'on
observe à l'échelle de la parcelle.

### La différenciation des cultures

![Distribution du rendement par culture dans les deux jeux](assets/figures/04_comparaison_datasets.png)

Cette figure résume le contraste principal entre les deux jeux. À gauche de chaque paire,
l'historique : quatre niveaux nettement distincts, du soja (médiane 1,45 t/ha) au riz
(3,06 t/ha). À droite, le jeu parcellaire : quatre boîtes superposées, avec des médianes de
4,64 à 4,66 t/ha.

Le même contraste apparaît en contrôlant la température, sur la plage commune aux deux jeux :

| Bande de température | Écart max entre cultures — parcellaire | Écart max entre cultures — historique |
|---|---|---|
| 15-20 °C | 0,01 t/ha | **3,29 t/ha** |
| 20-25 °C | 0,03 t/ha | **1,39 t/ha** |
| 25-30 °C | 0,02 t/ha | **1,68 t/ha** |

Dans l'historique, les cultures se répartissent aussi selon le climat des pays : les lignes
« blé » ont une température médiane de 16,2 °C et 661 mm de pluie, celles du riz 22,0 °C et
1 181 mm. Le jeu parcellaire donne 27,5 °C et 550 mm pour les quatre cultures, sans
distinction.

### Stratégie retenue

| Option | Verdict |
|---|---|
| **A.** Mettre les lignes des deux jeux bout à bout | écartée — une ligne ne représente pas la même chose, et les plages de valeurs diffèrent |
| **B.** Agriculture CropYield seul | `/predict` possible, mais pas de `/recommend` |
| **C.** Historique seul | cultures bien séparées, mais aucune variable de parcelle : pas de `/predict` |
| **D.** Les deux, chacun son rôle | **retenue** — un modèle par service, sans mélanger les lignes |

![Les deux pipelines de données du projet](assets/figures/06_pipelines.svg)

Dans `/recommend`, le pays sert de **clé pour retrouver le contexte** : température
historique, normale de pluie et niveau de pesticides. Ce n'est pas une variable du modèle. Si
on le mettait comme variable, le modèle apprendrait surtout un rendement par pays.

## 6. Préparation des données d'entraînement

À ce stade, aucun encodage, aucune mise à l'échelle et aucune imputation statistique. Tout
traitement qui calcule une statistique sera ajusté plus tard sur les seules données
d'entraînement, dans un pipeline.

### `/predict` — 999 769 lignes, 6 variables

**Cible :** `Yield_tons_per_hectare`.

| Variable retenue | Motif |
|---|---|
| `Rainfall_mm` | la plus liée au rendement |
| `Temperature_Celsius` | effet faible mais régulier |
| `Fertilizer_Used` | +1,50 t/ha |
| `Irrigation_Used` | +1,20 t/ha |
| `Crop` | l'utilisateur la choisit : elle fait partie de la saisie, même sans effet mesuré |
| `Soil_Type` | l'utilisateur la connaît, 6 modalités, sans effet mesuré |

Trois variables sont écartées, avec comme critère principal leur **disponibilité au moment de
la requête**. `Region` désigne une zone où personne ne peut se situer ; `Weather_Condition` et
`Days_to_Harvest` ne sont connues qu'après la saison.

Les **231 rendements négatifs sont supprimés**, plutôt que ramenés à zéro : les ramener à zéro
supposerait de connaître le vrai rendement, qu'on ne connaît pas. Ce retrait déplace la moyenne
de la cible de 0,001 t/ha seulement. Aucune autre ligne n'est retirée : les valeurs extrêmes au
sens de l'écart interquartile sont conservées.

### `/recommend` — 15 664 lignes, 4 variables

**Cible :** `yield_t_ha`. **Périmètre :** 117 pays, 10 cultures, 1991-2013.

| Variable | Rôle |
|---|---|
| `crop` | la culture à classer |
| `temp_hist` | température moyenne des années précédentes |
| `rain_mm` | normale climatique du pays |
| `log_pest_hist` | pesticides des années précédentes, en logarithme |
| `iso3`, `area` | **gardés hors du modèle** — pour retrouver le contexte et grouper la validation |
| `year` | **gardée hors du modèle** pour le découpage temporel ; reste à tester comme variable pendant la modélisation |

**On se place au moment d'une requête en 2014, avec des données disponibles jusqu'en 2013.**
Pour prédire l'année *t*, on n'utilise donc que des informations connues au plus tard en
*t−1*, aussi bien à l'entraînement qu'à l'inférence. `temp_hist` et `pest_hist` sont la moyenne
des trois années précédentes au plus, calculée pays par pays. Pour 2013, la fenêtre est
2010-2012 ; pour une requête sur 2014, ce sera 2011-2013. L'année 1990 n'a aucune année avant
elle et sort du périmètre : le jeu commence donc en 1991.

Trois contrôles vérifient qu'aucune donnée du futur ne passe : les années se suivent bien pour
chaque pays, la première année de chaque pays a des variables historiques vides, et le recalcul
manuel de la moyenne *t−3* à *t−1* sur 300 lignes tirées au hasard donne un écart maximum de 0.

![Pesticides : tonnage brut et logarithme](assets/figures/05_pesticides_log.png)

Le logarithme change vraiment la lecture. Les tonnages nationaux vont de 0 à 1 806 000 tonnes,
si bien qu'en valeur brute presque tous les points sont collés à gauche du graphique. Après
`log1p`, la corrélation avec le rendement passe de 0,05-0,25 à **0,26-0,57** selon la culture.

Deux ajustements complètent la préparation.

`rainfall.csv` ne contient **aucune ligne pour 2003** : 954 lignes de la fusion se retrouvaient
sans pluie, et toute l'année serait sortie du périmètre. Comme la pluie est constante par pays,
on réutilise la valeur déjà connue pour ce pays. 953 lignes récupérées ; les 177 restantes sont
celles de la Nouvelle-Calédonie, le seul pays sans aucune valeur de pluie.

Demander les pesticides **coûte 15 pays et 1 907 lignes** : le périmètre passe de 132 à
117 pays. On a testé une estimation à partir des voisins sur trois de ces pays, en regardant à
la fois les voisins par le climat et les pays frontaliers. On l'a écartée : les niveaux de
pesticides varient fortement entre pays voisins ou climatiquement proches, jusqu'à un facteur
10 000.

## 7. Limites identifiées

| Limite | Conséquence concrète |
|---|---|
| Le jeu de `/predict` n'a ni pays ni année, et sa colonne `Region` ne renvoie à aucun territoire | les estimations ne peuvent pas être rattachées à un lieu ni à une saison précise |
| Les 6 cultures y sont indiscernables | la culture apporte très peu d'information au rendement dans ce jeu |
| `/recommend` travaille à l'échelle du pays | un classement vaut pour le contexte d'un pays, jamais pour une parcelle précise |
| `rain_mm` est une normale climatique | le service ne réagit pas à la météo de l'année : deux requêtes d'années différentes sur un même pays donneront la même pluie |
| `pesticides_t` est un tonnage national | difficile à comparer directement entre pays de tailles différentes ; ce n'est pas une pratique individuelle |
| Couverture limitée à 117 pays | les 51 autres pays de l'historique ne peuvent pas être servis ; mieux vaut un refus clair qu'une prédiction dégradée |
| 39 % des couples pays × culture sont absents de l'historique | une culture peut être classée dans un pays où on ne l'a jamais observée ; igname, plantain et manioc sont rarement observés sous 16-17 °C dans les données |
| Cultures inégalement représentées | igname 547 lignes et 25 pays, plantain 602 lignes et 27 pays, contre 2 407 lignes et 109 pays pour le maïs |
| Les niveaux de rendement diffèrent beaucoup selon la culture | en t/ha, les tubercules se placent devant les céréales (pomme de terre, médiane 16 t/ha, contre 1,3 pour le sorgho) : le classement reflète d'abord cet écart entre familles de cultures |

## 8. Suite du projet

Les deux jeux d'entraînement sont prêts et contrôlés. Ils permettent d'attaquer :

- l'entraînement et la comparaison de modèles de régression pour les deux services ;
- la validation temporelle de `/recommend`, avec 2013 gardée pour le test final et deux
  baselines simples prévues : la moyenne par culture et la persistance pays × culture ;
- le test de `year` comme variable explicative ;
- le suivi des expériences ;
- un garde-fou pour signaler les recommandations hors du domaine climatique observé.

Ces travaux et leurs résultats seront ajoutés dans une version ultérieure de ce rapport.

---

*Analyses détaillées et code : notebooks `01` à `06` du dépôt. Figures régénérables avec
`scripts/make_report_figures.py`, rapport HTML avec `scripts/build_report.py`.*
