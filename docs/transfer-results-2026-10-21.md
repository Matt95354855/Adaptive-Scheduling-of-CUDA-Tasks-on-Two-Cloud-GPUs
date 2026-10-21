# Transferts hôte→GPU : résultats et analyse du 21 octobre 2026

## Objectif

Cette campagne prolonge l'étude de calcul résident par une charge où chaque tâche crée deux matrices en mémoire hôte épinglée, les copie vers le GPU choisi, puis exécute un GEMM et vérifie sa somme. La question est de savoir si un ordonnanceur à deux GPU reste utile lorsque la préparation des entrées et les copies H2D font partie du lot mesuré.

L'instance est un pod RunPod à deux NVIDIA RTX PRO 4000 Blackwell (24 467 MiB par GPU), pilote 595.91.07, PyTorch 2.8.0+cu128 et CUDA 12.8. Toutes les sommes vérifiées ont passé. Huit ordres indépendants (seeds 410–417) et deux répétitions ont été réalisés par condition ; l'ordre des politiques est randomisé dans chaque bloc.

## Résultat principal : entrées épinglées allouées par tâche

La charge comporte 24 tâches, huit par taille 1024, 1536 et 2048. Les politiques `single`, `round_robin`, `size_lpt`, `calibrated` et `feedback` sont comparées. Les valeurs ci-dessous sont la médiane des médianes par seed contre `round_robin`; l'intervalle bootstrap décrit les huit ordres sur cette instance.

| Politique | Médiane par seed (ms) | Écart à round-robin | Victoires par seed | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | --- |
| single | 73,561 | -23,07 % | 8/8 | [-32,84 ; -13,88] % |
| round_robin | 104,604 | 0,00 % | — | — |
| calibrated | 102,389 | +1,63 % | 4/8 | [-10,69 ; +32,64] % |
| size_lpt | 110,342 | +12,05 % | 1/8 | [+0,70 ; +16,80] % |
| feedback | 125,982 | +18,80 % | 0/8 | [+6,69 ; +37,64] % |

Le résultat brut suggère à tort que n'utiliser qu'un GPU est préférable. Les métriques internes montrent pourtant des sommes de GEMM très proches : 9,550 ms (`single`) contre 9,693 ms (`round_robin`) par run médian, et des sommes H2D comparables : 26,610 ms contre 27,938 ms. Le temps de préparation hôte médian est faible mais non nul (0,309 ms contre 0,428 ms par tâche). Les valeurs complètes sont dans `results/transfer-paired-2026-10-21.csv` et les diagnostics dans `results/transfer-diagnostic-summary-2026-10-21.csv`.

## Contrôle causal : mêmes copies, entrées hôte réutilisées

Le contrôle `--reuse-pinned-inputs` conserve une copie H2D de deux matrices à chaque tâche, mais les tampons hôte épinglés sont créés une fois par taille et par worker avant le lot. Les entrées réutilisées valent un ; le coût numérique du GEMM est inchangé, mais l'allocation hôte par tâche est retirée. Seules `single` et `round_robin` sont nécessaires pour ce contrôle.

| Politique | Médiane par seed (ms) | Écart à round-robin | Victoires par seed | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | --- |
| round_robin | 41,001 | 0,00 % | — | — |
| single | 53,887 | +35,86 % | 0/8 | [+27,99 ; +39,21] % |

Le renversement est net sur les huit seeds. Avec les tampons réutilisés, les sommes médianes de GEMM restent comparables (9,415 ms pour `single`, 9,425 ms pour `round_robin`) et les copies H2D ne favorisent pas `single` (20,916 ms contre 23,849 ms). Le gain de round-robin vient donc de l'usage concurrent des deux GPU lorsque l'allocation de mémoire hôte épinglée par tâche a disparu. L'hypothèse initiale d'une simple saturation du chemin H2D est insuffisante.

## Trace Nsight Systems

Nsight Systems 2025.1.1 a profilé une répétition du seed 410 avec `single` et `round_robin`. La capture contient aussi l'initialisation et la calibration : elle décrit le mélange de l'exécution, et ne permet pas une attribution propre par politique. Dans cette capture, les copies H2D représentent 99,9 % du temps des opérations mémoire GPU : 216 copies, 118,355 ms cumulés et 0,501 ms de médiane par copie. Les noyaux GEMM principaux totalisent 41,457 ms dans la même capture.

Cette trace confirme que le transfert est un coût réel. Elle ne prouve ni une saturation PCIe ni une topologie particulière. Le contrôle avec tampons réutilisés apporte la preuve plus utile ici : il isole l'allocation hôte par tâche comme variable qui change le sens de la comparaison.

## Charge GPU soutenue

Pour éviter de conclure à partir de lots trop courts, une deuxième charge transfère 12 paires de matrices de tailles 4096, 5120 et 6144 (quatre de chaque), puis répète chaque GEMM 16 fois. Les entrées épinglées sont allouées dans chaque tâche. Quatre ordres de tâches (seeds 600–603) et deux répétitions par politique ont été exécutés.

| Politique | Médiane par seed (ms) | Écart à round-robin | Victoires par seed | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | --- |
| round_robin | 1 528,394 | 0,00 % | — | — |
| single | 2 489,462 | +64,86 % | 0/4 | [+50,13 ; +88,97] % |

Les sommes d'événements GEMM sont presque identiques entre politiques (2 037,097 ms pour round-robin, 2 050,718 ms pour single), mais la somme des événements n'est pas un temps mural : elle dépasse le makespan lorsqu'ils se chevauchent sur les deux GPU. Les copies H2D cumulées sont aussi proches (125,828 ms et 125,434 ms). Ici, le calcul soutenu amortit l'allocation hôte et le parallélisme de round-robin devient déterminant.

Un échantillonnage `nvidia-smi` à environ une seconde pendant cette série, qui mélange volontairement les deux politiques, a relevé un pic de 100 % sur chacun des GPU. Il contient 25 échantillons à au moins 90 % pour le GPU 0, huit pour le GPU 1 et cinq instants où les deux lignes associées étaient simultanément à au moins 90 %. Il ne mesure pas une moyenne d'occupation par politique, mais confirme que la charge atteint effectivement les deux appareils. Les résultats détaillés sont dans `results/transfer-heavy-paired-2026-10-21.csv`, `results/transfer-heavy-diagnostic-summary-2026-10-21.csv` et `results/transfer-heavy-utilization-summary-2026-10-21.csv`.

## Durées cachées sous charge de transfert

Pour tester l'adaptation en régime calculé, douze tâches de taille visible identique 4096 ont été copiées depuis l'hôte. Quatre exécutent 32 GEMM et huit en exécutent 4 ; cette multiplicité est mélangée par seed et n'est jamais fournie à l'ordonnanceur. Huit ordres indépendants sont exécutés une fois avec `single`, `round_robin` et `feedback`.

| Politique | Médiane par seed (ms) | Écart à round-robin | Victoires par seed | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | --- |
| feedback | 550,130 | -8,11 % | 6/8 | [-19,18 ; +4,40] % |
| round_robin | 604,551 | 0,00 % | — | — |
| single | 975,915 | +63,05 % | 0/8 | [+43,99 ; +82,02] % |

Les sommes de GEMM et de copies H2D sont pratiquement égales entre `feedback` et round-robin (769,102 contre 773,680 ms de GEMM ; 84,301 contre 84,133 ms H2D). La différence concerne la fin de lot : round-robin attend davantage de tâches déjà affectées à un worker (médiane de file 219,329 ms), tandis que feedback réaffecte une tâche dès une fin observée. Le signe est cohérent avec cette explication, mais l'intervalle inclut zéro : huit ordres et une répétition ne suffisent pas pour affirmer un gain général. Les données sont dans `results/transfer-hidden-paired-2026-10-21.csv` et `results/transfer-hidden-diagnostic-summary-2026-10-21.csv`.

## Sonde finale de saturation hôte et GPU

Cette sonde est séparée de l'évaluation des politiques. Pendant 60 secondes demandées, deux processus ont répété des GEMM 6144×6144, un par GPU, tandis que 40 processus CPU exécutaient une préparation hôte synthétique indépendante. Les deux checksums GPU ont passé ; l'exécution a duré 69,744 s avec 3 910 et 3 915 GEMM complétés.

`nvidia-smi`, échantillonné chaque seconde, mesure 84,5 % et 84,3 % de moyenne sur les GPU 0 et 1, une médiane de 100 % et 59 échantillons sur 70 à au moins 70 % pour chacun. La cible GPU est donc atteinte. `top` rapporte seulement 37,3 % de moyenne CPU sur les 48 processeurs logiques visibles. Le cgroup du pod limite cependant le conteneur à 2 040 000 µs par période de 100 000 µs, soit 20,4 cœurs : ce plafond vaut 42,5 % de ces 48 processeurs. Les 40 producteurs atteignent donc le plafond CPU du pod, même si l'échelle hôte ne peut pas afficher 70 %. La table reproductible est `results/saturation-probe-summary-2026-10-21.csv` et le code est `experiments/saturation_probe.py`.

## Contrôle de longueur de lot

Sur le sous-lot de 12 tâches, l'allocation épinglée par tâche conserve le résultat court : `single` est 23,84 % plus rapide que round-robin (4/4 seeds, IC [-28,29 ; -18,94] %). Avec les mêmes copies mais les entrées hôte réutilisées, round-robin reprend l'avantage : `single` est 22,14 % plus lent (0/4, IC [+10,58 ; +26,48] %). Ces contrôles plus courts, publiés dans `results/transfer-scale-12-fresh-paired-2026-10-21.csv` et `results/transfer-scale-12-reused-pinned-paired-2026-10-21.csv`, confirment que le changement ne dépend pas uniquement du nombre de tâches.

## Conclusion opérationnelle

Pour cette implémentation, la règle n'est pas « deux GPU accélèrent toujours » ni « les copies H2D imposent un seul GPU ». Le choix dépend de l'état des entrées :

- avec allocation épinglée dans chaque tâche, `single` a été le meilleur sur les huit ordres testés ;
- avec tampons épinglés réutilisés, `round_robin` a été le meilleur sur les huit mêmes ordres ;
- sous calcul soutenu, `round_robin` a été le meilleur sur les quatre ordres lourds et a réellement chargé les deux GPU ;
- avec durées cachées sous transfert, feedback a un avantage descriptif de 8,11 %, encore trop incertain pour être généralisé ;
- le CPU du pod est plafonné à 20,4 cœurs, ce qui doit être pris en compte lors de toute analyse de pression hôte.

Un ordonnanceur qui ignore le cycle de vie des entrées peut sélectionner une mauvaise politique même si ses estimations de GEMM sont exactes. La prochaine expérience utile est de varier séparément la taille des copies et le nombre de tâches, puis de mesurer l'allocation épinglée avec un profileur CPU. Elle devra aussi répéter l'étude sur une seconde instance avant toute généralisation.

## Limites et conservation

Cette étude porte sur un seul pod et un seul logiciel. Les JSON détaillés et le fichier binaire Nsight étaient sur le disque éphémère du pod, car aucun stockage persistant n'a été demandé ; ils ne seront pas conservés après l'arrêt. Le dépôt conserve le code, les 80 mesures de l'expérience allouée, les 32 mesures du contrôle, les 16 mesures lourdes, les 24 mesures à durée cachée, les contrôles de longueur de lot, la sonde de saturation, les statistiques Nsight agrégées et la méthode d'analyse. Les intervalles internes de worker publiés dans le CSV ne sont pas une mesure directe d'occupation GPU.
