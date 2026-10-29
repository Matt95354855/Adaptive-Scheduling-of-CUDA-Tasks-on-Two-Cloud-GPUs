# Contrôle préenregistré : workers GPU conservés entre lots

Ce test est **secondaire** et sera lancé seulement après la grille factorielle,
ses exports et les profils Nsight, si le budget reste suffisant. Aucun résultat
GPU de ce plan n'existe au moment de sa publication.

## Question et comparaison

Les campagnes précédentes créaient deux workers PyTorch, puis les calibraient,
à chaque lot. Dans le contrôle dense du 24 octobre, 16 lots chronométrés
totalisaient 76,57 s sur 153 secondes de télémétrie. Le nouveau test mesure si
conserver les workers pendant **quatre lots** réduit le temps complet de ces
quatre lots et accroît l'occupation du pod. Il compare :

- `cold` : `study.run()` recrée et calibre les deux workers à chaque lot ;
- `warm` : deux workers et leurs tampons épinglés restent actifs pendant
  quatre lots ; une seule calibration, puis arrêt des workers.

Les deux bras utilisent les mêmes tâches, des entrées épinglées réutilisées
de valeur 1, deux GPU, H0, 12 tâches par lot avec quatre matrices de 4096,
quatre de 5120 et quatre de 6144, **64 GEMM par tâche**. Les politiques
`round_robin` et `feedback` sont toutes deux exécutées deux fois dans un ordre
identique entre bras et randomisé selon la seed.

## Grille et mesure

- Seeds **1700–1707**, huit paires `cold/warm`. L'ordre des bras alterne :
  `cold→warm` pour les indices pairs, `warm→cold` pour les indices impairs.
- **16 blocs**, **64 lots/runs**, **768 tâches**. Le seed 1699 sert à quatre
  runs de vérification et est exclu des 64 mesures.
- Le temps primaire est le **temps mural complet d'un bloc de quatre lots**,
  depuis avant la création des workers jusqu'après leur arrêt. Le résultat
  par seed est `100 × (1 − temps_warm / temps_cold)`. Rapporter médiane, IC
  bootstrap 95 % sur les huit seeds, et nombre de seeds gagnées ; ce bloc
  reste un diagnostic d'ingénierie avec un faible effectif, sans revendication
  de supériorité générale.
- Mesurer aussi les makespans des lots, les sommes d'événements CUDA par GPU
  et la télémétrie SM à une seconde **séparément pour chaque bloc**. Rapporter
  moyenne SM et part de relevés à zéro et à ≥70 % pour chaque bras/GPU. Le
  seuil de 70 % d'occupation moyenne de campagne reste un objectif pratique,
  pas le critère statistique principal.
- Valider les 768 checksums, les politiques/répétitions uniques et les 16
  blocs complets. Aucune suppression de seed après mesure.

La comparaison inclut délibérément les différences de calibration et de durée
de vie des tampons ; elle mesure un **choix d'architecture complet**, pas le
coût isolé d'une décision d'ordonnancement. Les bras sont mesurés l'un après
l'autre ; l'alternance limite mais n'annule pas une dérive de l'instance.
Ce contrôle n'est pas fusionné avec la grille H0/H40 × entrées.

Le code, les 64 runs, 768 tâches, 16 journaux GPU, analyses, figures et logs
seront copiés et publiés sur GitHub avant arrêt du pod. Les scripts peuvent
reprendre un checkpoint entier uniquement sur le même pod et le même commit.
Le [générateur de figures](../experiments/plot_pool_control.py) est préparé avant
location ; aucune figure de résultat n'est créée avant les mesures réelles.
