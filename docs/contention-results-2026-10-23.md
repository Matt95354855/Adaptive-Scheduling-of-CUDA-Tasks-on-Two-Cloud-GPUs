# Étude de contention hôte, 23 octobre 2026

## État de la campagne

Les blocs H0 et H40 du [protocole préenregistré](next-campaign-protocol-2026-10-21.md)
sont terminés. La réplication sur de nouvelles seeds reste à mesurer.

## Instance et contrôle

Pod RunPod `twqrwkf0s7u2nu`, deux NVIDIA RTX PRO 4000 Blackwell de 24 467 MiB,
pilote 595.91.07, PyTorch 2.8.0+cu128, runtime CUDA 12.8. `nproc` signale 48
processeurs logiques ; `cpu.max` vaut `2040000 100000`, soit un quota de 20,4
cœurs. La topologie `nvidia-smi topo -m` affiche `NODE` entre les GPU. Elle
diffère du `PHB` relevé lors de la campagne précédente ; les mesures entre
instances ne sont donc pas fusionnées. Le conteneur est éphémère, sans volume
persistant.

Code exécuté : commit `0adf59b9e4c17eabc0e90d7b6a0e0e3122af215d`.
Chaque tâche a passé la vérification de somme du banc d'essai.

## H0 : sans charge CPU supplémentaire

Douze seeds indépendantes (800–811), deux répétitions chacune, deux politiques
dans un ordre randomisé : 48 mesures. Les répétitions sont résumées par seed
avant la comparaison appariée. Le pourcentage ci-dessous est la médiane des
changements par seed par rapport au tourniquet.

| Politique | Temps médian par seed | Changement apparié | Victoires | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | ---: |
| `round_robin` | 679,396 ms | référence | — | — |
| `feedback` | 573,896 ms | −16,18 % | 9/12 | [−17,99 ; +2,23] % |

Le seuil préenregistré de 9 victoires est atteint, mais l'intervalle inclut
zéro. Le résultat est prometteur et encore incertain ; H40 et la réplication
doivent départager cette hypothèse.

Les métriques médianes par run montrent des sommes de GEMM comparables
(789,915 ms pour round-robin, 787,864 ms pour feedback) et des copies H2D
également proches (84,900 et 85,407 ms). Les temps de préparation médian par
tâche sont 2,774 et 2,753 ms. La médiane d'attente dans la file vaut 253,079
ms avec les tâches préaffectées de round-robin et 0,200 ms avec l'envoi en ligne
de feedback : cette métrique reflète surtout la façon de remplir les files,
et ne représente pas un coût GPU additionnel. Les intervalles entre tâches de
worker sont plus grands avec feedback, cohérents avec ses échanges de contrôle.

- [48 mesures individuelles H0](../results/transfer-hidden-h0-2026-10-23.csv)
- [Diagnostics par run H0](../results/transfer-hidden-h0-task-metrics-2026-10-23.csv)
- [Chronologies par tâche H0](../results/transfer-hidden-h0-tasks-2026-10-23.csv)

L'échantillonnage GPU à une seconde comprend les calibrations hors fenêtre ; il
ne sera pas utilisé pour comparer les politiques séparément.

## H40 : 40 processus CPU concurrents

Même grille de douze seeds et deux répétitions. Le temps médian par seed de
`round_robin` passe à 1 328,371 ms ; celui de `feedback` à 1 361,405 ms.
Le changement apparié médian de `feedback` est +0,56 % (plus lent), avec 6/12
seeds gagnées et un IC bootstrap 95 % de [−7,10 ; +4,86] %. Le critère de
robustesse préenregistré n'est pas atteint.

La comparaison des gains par seed entre H40 et H0 donne une interaction médiane
de −11,09 points de pourcentage, IC bootstrap 95 % [−16,64 ; −1,80] points.
Cela soutient une réduction de l'avantage de `feedback` sous contention sur cette
instance. Les conditions ont été exécutées successivement ; un effet de dérive
temporelle de l'instance reste possible.

Le quota CPU est 20,4 cœurs pour 48 processeurs logiques visibles. Pendant H40,
le compteur cgroup `nr_throttled` est passé de 103 à 932 ; `throttled_usec`
de 185 072 778 à 2 053 610 429. Ces compteurs englobent le conteneur entier,
pas uniquement les fenêtres chronométrées. Les médianes par run montrent un
temps de préparation des entrées par tâche proche de 100 ms dans les deux
politiques, contre environ 2,8 ms en H0. La somme des temps d'événements GPU
reste proche de 777 ms, contre 788–790 ms en H0 : le ralentissement est surtout
côté hôte. La répartition de quatre tâches longues devient plus équilibrée sous
`feedback` (déséquilibre médian 0 contre 2 pour round-robin), mais les intervalles
entre tâches sur les workers passent à environ 10 ms pour `feedback`, contre
moins de 1 ms pour round-robin. Il s'agit d'une explication compatible avec les
mesures, pas d'une décomposition causale complète du makespan.

- [48 mesures individuelles H40](../results/transfer-hidden-h40-2026-10-23.csv)
- [Diagnostics par run H40](../results/transfer-hidden-h40-task-metrics-2026-10-23.csv)
- [Chronologies par tâche H40](../results/transfer-hidden-h40-tasks-2026-10-23.csv)
- [Interaction et valeurs par seed](../results/transfer-hidden-contention-2026-10-23.json)

## Choix préalablement à la réplication

H0 présente le signal le plus crédible : 9/12 seeds gagnées, contre 6/12 en H40,
et une médiane de gain plus élevée, malgré un IC qui inclut zéro. Selon le
protocole, la réplication portera donc sur H0, seeds 900–911, deux répétitions
et les mêmes politiques. Ce choix est fixé avant l'exécution de ces nouvelles
mesures. La réplication déterminera si l'avantage descriptif de H0 se maintient
sur de nouveaux ordres de tâches ; elle ne réparera pas à elle seule la limite
de généralisation à d'autres instances et charges.
