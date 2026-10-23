# Avancement par rapport au plan de recherche, 23 octobre 2026

Ce suivi reprend les phases du plan de travail v0.1 fourni séparément. Le plan
était rédigé avant les premiers résultats ; les observations actuelles se
trouvent dans ce dépôt et priment sur son état initial.

| Phase | État vérifiable | Reste à faire |
| --- | --- | --- |
| 0. Diagnostic | Deux RTX PRO 4000 visibles ; pilote, PyTorch, quota CPU, topologie et correction relevés sur plusieurs pods. | Conserver un manifeste machine automatisé pour chaque nouvelle instance. |
| 1. Harnais | Seeds, politiques, tâches, chronologies et checksums dans `experiments/study.py`. | Transformer le prototype en API de tâches et tests de rejeu indépendants du benchmark. |
| 2. Baselines | Mono GPU, round-robin, feedback et politiques statiques fondées sur taille dans les campagnes précédentes. | Ajouter un oracle hors ligne et une vraie file avec work stealing sur charges identiques. |
| 3. Observabilité | Événements CUDA, H2D, attente, traces Nsight et compteurs cgroup ; détails par tâche publiés en CSV. | Isoler le coût de décision et l'occupation GPU dans chaque fenêtre de run, puis instrumenter une charge applicative. |
| 4. Adaptation | Politique réactive au premier GPU libre ; bénéfice conditionnel prospectivement validé. | Concevoir une décision en ligne à partir de signaux disponibles avant ou pendant le lot, sans regarder les durées cachées a posteriori. |
| 5. Pilote à coût réduit | Études des 20–23 octobre, pods éphémères et budget de session plafonné. | Tenir un journal automatisé de coût par expérience. |
| 6. Évaluation | Appariement par seed, ordre des politiques randomisé, IC bootstrap, H0/H40, réplication indépendante et validation prospective de 24 seeds. | Comparer à un modèle de coût transfert/CPU, à des DAG et à des arrivées en ligne. |
| 7. Réplication | Une autre instance RTX PRO 4000 a été utilisée ; sa topologie `NODE` diffère du `PHB` observé précédemment. Les résultats ne sont pas fusionnés entre instances. | Répéter le protocole prospectif et alterner H0/H40 sur une instance comparable, avec les mêmes versions et un budget préalloué. |

## Décision scientifique actuelle

La relation entre gain de `feedback` et déséquilibre statique des quatre tâches
longues est [validée prospectivement sur la charge synthétique de ce
pod](contention-results-2026-10-23.md). Cela répond à une partie de H1 et H4,
mais le critère de groupe utilise la durée vraie connue a posteriori. H2
(affinité des données et politique de transfert), H3 (signaux prédictifs en
ligne) et H5 (coût total d'un ordonnanceur complet) restent ouverts.

## Prochaines expériences logiques

1. Implémenter une politique en ligne qui estime la durée à partir de quelques
   terminaisons observées ; figer son seuil sur des seeds de calibration,
   puis comparer sur des seeds nouvelles à round-robin, feedback et baselines
   statiques. Mesurer aussi le coût CPU de cette décision.
2. Alterner H0/H40 et allocation fraîche/réutilisation d'entrées épinglées
   sur les mêmes seeds, pour départager contention de préparation, dérive
   temporelle et coût de coordination.
3. Ajouter affinité mémoire, DAG et arrivées par rafale avec un manifeste de
   charge ; exiger correction et même travail utile pour toutes les politiques.
4. Répliquer les conclusions sur un second pod comparable avant toute
   revendication de généralisation au matériel RTX PRO 4000 Blackwell.
