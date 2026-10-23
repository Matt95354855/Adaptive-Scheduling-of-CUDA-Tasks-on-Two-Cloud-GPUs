# Étude de contention hôte, 23 octobre 2026

## État de la campagne

Les trois blocs du [protocole de contention](next-campaign-protocol-2026-10-21.md)
sont terminés : H0, H40 et la réplication indépendante de H0. Une
[validation prospective](strata-validation-protocol-2026-10-23.md) de la
relation découverte entre gain et déséquilibre statique a ensuite été menée.

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
ne sera pas utilisé pour comparer les politiques séparément. Sur les blocs
entiers, les médianes d'utilisation sont nulles et les moyennes vont de 9,8 à
14,9 % selon GPU et condition, malgré des pics à 100 %. Cela reflète notamment
la création répétée de processus, la calibration et les intervalles sans lot ;
ce n'est pas l'occupation pendant un run chronométré. La campagne de charge
soutenue du 21 octobre a été conçue séparément pour mesurer la saturation.
[Agrégats de l'échantillonnage GPU](../results/transfer-hidden-gpu-sampling-2026-10-23.csv).

## H40 : 40 processus CPU concurrents

Même grille de douze seeds et deux répétitions. Le temps médian par seed de
`round_robin` passe à 1 328,371 ms ; celui de `feedback` à 1 361,405 ms.
Le changement apparié médian de `feedback` est +0,56 % (plus lent), avec 6/12
seeds gagnées et un IC bootstrap 95 % de [−7,10 ; +4,86] %. Le critère de
robustesse préenregistré n'est pas atteint.

La comparaison des gains par seed entre H40 et H0 donne une interaction médiane
de −11,09 points de pourcentage, IC bootstrap 95 % [−16,64 ; −1,80] points ;
10 des 12 seeds perdent du gain sous H40.
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

## Réplication H0 : nouvelles seeds 900–911

Les 48 nouvelles mesures ont toutes passé le contrôle de somme. Le temps médian
par seed vaut 551,578 ms pour `round_robin` et 565,918 ms pour `feedback`.
Le changement apparié médian de `feedback` est **+1,16 %** (plus lent), avec
4/12 seeds gagnées et un IC bootstrap 95 % de [−6,43 ; +6,88] %.
L'avantage descriptif initial de 16,18 % ne se reproduit donc pas sur ce jeu
indépendant. Les 24 seeds réunies donnent un gain médian de seulement 0,21 %,
13/24 victoires et un IC bootstrap descriptif de [−2,94 ; +16,70] %.
La décision de répliquer H0 avait été prise après lecture de H0/H40 ; cette
sélection interdit de présenter le résultat regroupé comme une confirmation
préenregistrée.

- [48 mesures individuelles de réplication](../results/transfer-hidden-replication-h0-2026-10-23.csv)
- [Diagnostics par run de réplication](../results/transfer-hidden-replication-h0-task-metrics-2026-10-23.csv)
- [Chronologies par tâche de réplication](../results/transfer-hidden-replication-h0-tasks-2026-10-23.csv)

## Analyse exploratoire de l'hétérogénéité entre seeds

Les quatre tâches longues sont invisibles au planificateur. Leur affectation
effective en round-robin est mesurable a posteriori : elle est identique dans
les deux répétitions d'une seed, et ses deux GPU peuvent recevoir 2–2, 3–1
ou 4–0 tâches longues. Dans le premier bloc H0, 8 des 24 runs round-robin
étaient équilibrés 2–2 ; dans la réplication, 18 des 24 l'étaient. Ce changement
de composition explique pourquoi une moyenne ou une médiane globale varie
fortement entre blocs.

Sur les 24 seeds réunies, la médiane du gain `feedback` est **−3,66 %** parmi
les 13 seeds équilibrées (2 victoires), et **+17,21 %** parmi les 11 seeds
déséquilibrées (11 victoires). Les IC bootstrap descriptifs sont respectivement
[−10,09 ; −0,16] % et [+15,67 ; +18,98] %. `feedback` équilibre souvent les
tâches longues, mais paye ses échanges de contrôle lorsque round-robin est déjà
équilibré. Cette stratification a été définie après lecture des résultats : ses
intervalles ne constituent pas une validation confirmatoire ni une politique
applicable en ligne, puisque la durée longue est cachée avant exécution.

- [Gain et déséquilibre pour chaque seed](../results/transfer-hidden-seed-strata-2026-10-23.csv)
- [Agrégats stratifiés](../results/transfer-hidden-strata-summary-2026-10-23.json)
- [Code de la stratification](../experiments/seed_stratification.py)

## Contrôle ciblé avec Nsight Systems

Nsight Systems 2025.1.1.0 a capturé séparément H0 et H40 sur la seed 801,
une répétition de `round_robin` et `feedback` par condition :

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o results/nsys-contention-h40-seed801 \
  python experiments/study.py --out results/nsys-contention-h40-seed801.json \
  --workload transfer_hidden --host-load-workers 40 \
  --seed-start 801 --seeds 1 --repetitions 1 \
  --policies round_robin feedback
```

La référence H0 utilise la même commande sans `--host-load-workers 40`.
Les rapports `cuda_gpu_kern_sum` et `cuda_gpu_mem_time_sum` contiennent dans
chaque trace 344 GEMM principaux et 88 copies H2D, **calibration comprise**.
Le temps total de ces GEMM est 1 667,551 ms en H0 et 1 648,157 ms en H40 ;
le temps total des copies H2D est 288,061 puis 249,083 ms. Le travail GPU
capturé est donc de même ordre dans les deux conditions, alors que le makespan
profilé de `round_robin` passe de 691,605 à 1 429,115 ms. Cette trace est
compatible avec le diagnostic d'un goulot hôte. Comme elle agrège les deux
politiques et leur calibration, elle ne quantifie pas leurs temps GPU séparés.
Le profilage modifie l'exécution ; ces deux runs sont exclus des 144 mesures
statistiques H0/H40/réplication. Les fichiers `.nsys-rep` restent éphémères,
conformément au protocole sans stockage permanent.

- [Agrégats Nsight et temps profilés](../results/nsys-contention-seed801-summary-2026-10-23.csv)

## Validation prospective : seeds 1000–1023

Le [protocole](strata-validation-protocol-2026-10-23.md) et son critère ont été
publiés avant le lancement des nouvelles mesures. Les 96 runs attendus et les
1 152 tâches sont présents ; chaque tâche a passé le contrôle de somme. Les
24 seeds se répartissent en 10 ordres où les quatre tâches longues sont
équilibrées 2–2 sous round-robin, et 14 ordres déséquilibrés 3–1 ou 4–0.

| Groupe | Seeds | Victoires `feedback` | Gain médian `feedback` | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | ---: |
| Équilibré | 10 | 5/10 | −1,60 % | [−5,89 ; +1,18] % |
| Déséquilibré | 14 | 14/14 | +19,34 % | [+13,20 ; +25,02] % |

La différence préenregistrée des médianes de gain est **+20,94 points de
pourcentage** en faveur des ordres déséquilibrés, IC bootstrap 95 %
[+13,02 ; +27,66]. Les deux groupes dépassent la taille minimale de six
seeds et la borne inférieure est positive : **le critère de validation est
atteint sur cette instance et cette charge**. Sur les 24 seeds prises ensemble,
`feedback` gagne 19/24 fois et son gain médian est +10,72 %, mais ce chiffre
dépend de la proportion de seeds déséquilibrées ; il ne constitue pas un gain
général de la politique.

- [96 runs prospectifs](../results/strata-validation-2026-10-23.csv)
- [Diagnostics par run](../results/strata-validation-task-metrics-2026-10-23.csv)
- [1 152 chronologies par tâche](../results/strata-validation-tasks-2026-10-23.csv)
- [Classement et gain par seed](../results/strata-validation-seeds-2026-10-23.csv)
- [Verdict préenregistré](../results/strata-validation-summary-2026-10-23.json)
- [Code d'analyse](../experiments/validate_strata.py)

## Conclusion à ce stade

La validation prospective soutient un résultat **conditionnel** : le feedback
de fin de tâche aide quand le round-robin déséquilibre les durées cachées, et
son coût peut annuler le gain quand la répartition statique est déjà bonne.
Une forte contention CPU a supprimé le gain médian dans le bloc H40, malgré
l'équilibrage des tâches longues. Il reste à tester un signal observable en
ligne permettant de choisir la politique, puis à isoler l'effet de la
préparation des entrées sous contention. Une réplication alternant les
conditions sur une autre instance est nécessaire pour qualifier la portée du
résultat H40 et la variabilité cloud. Le critère de groupe actuel utilise les
durées connues seulement après exécution ; il ne constitue pas lui-même un
ordonnanceur utilisable en ligne.
