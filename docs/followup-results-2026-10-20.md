# Résultats de l'étude de suivi — 20 octobre 2026

## Conclusion courte

Sur cette seule paire de GPU cloud et pour des lots de tâches GEMM indépendantes déjà résidentes, le feedback de fin de tâche n'est pas une politique universellement meilleure. Il devient utile lorsque le coût réel des tâches ne peut pas être déduit des métadonnées visibles et que l'ordre round-robin est déséquilibré. Quand le coût est prévisible, une répartition statique informée par la taille est plus rapide. Quand les tâches sont courtes, le tour de messages du feedback domine.

Toutes les données publiées ici sont des temps muraux appariés. Chaque seed correspond à un ordre de tâches indépendant ; les répétitions d'une même seed sont résumées par leur médiane avant toute comparaison. Les intervalles bootstrap à 95 % décrivent la variation observée entre les douze ordres sur cette machine. Ils ne généralisent pas à toutes les machines cloud ou applications CUDA.

## Matériel et protocole

- RunPod, deux NVIDIA RTX PRO 4000 Blackwell identiques ; les deux GPU communiquent en PHB, sans NVLink affiché.
- PyTorch 2.8.0+cu128, pilote 595.91.07, runtime CUDA 12.8, spawn de deux workers.
- Le temps commence avant l'envoi des tâches et s'arrête après toutes les confirmations des workers. Les créations de processus, warm-ups et calibrations sont hors fenêtre.
- Les événements CUDA encadrent torch.mm seulement. Une somme analytique valide chaque tâche avec une tolérance relative de 1e-4.
- Les plans de charges, les seeds et les politiques sont décrits avant mesure dans [le protocole de suivi](followup-protocol-2026-10-20.md).

Les politiques statiques envoient un lot entier au début. feedback garde une tâche en vol par GPU et envoie la suivante au premier worker qui confirme une fin. Son coût de coordination est donc volontairement inclus dans le temps mesuré.

## Résultats appariés

| Charge et politiques | Round-robin, médiane seed (ms) | Résultat principal face à round-robin | Interprétation |
|---|---:|---:|---|
| Calcul, 24 GEMM de tailles 1024–4096, 12 seeds × 3 répétitions | 51,678 | size_lpt 48,015 ms, -7,18 %, 12/12 seeds, IC [-10,01 ; -3,97] % | Connaître les tailles suffit ici à équilibrer le travail. |
| Même charge | 51,678 | calibrated 48,160 ms, -6,38 %, 12/12, IC [-8,59 ; -3,21] % | Le modèle calibré confirme le gain statique ; il n'améliore pas nettement LPT. |
| Même charge | 51,678 | feedback 51,922 ms, -0,08 %, 6/12, IC [-2,43 ; +1,96] % | Résultat indécis à cette échelle : aucune amélioration robuste. |
| Petites tâches, allocations par tâche, 12 seeds × 2 répétitions | 25,746 | feedback 29,088 ms, +12,53 %, 0/12, IC [+9,59 ; +15,69] % | Le coût d'orchestration masque tout bénéfice de rééquilibrage. |
| Petites tâches, opérandes préalloués | 24,615 | feedback 28,048 ms, +14,32 %, 0/12, IC [+12,33 ; +15,65] % | La pénalité persiste lorsque les allocations par tâche sont retirées. |
| Durée cachée, 24 tâches 3072 dont 8 répétées 12 fois, 12 seeds × 2 répétitions | 172,854 | feedback 154,200 ms, -10,91 %, 8/12, IC [-16,14 ; +2,52] % | Le feedback aide sur les ordres déséquilibrés, mais perd sur les ordres déjà équilibrés. |

Les fichiers de mesures sont publiés sous forme de tableaux larges, une ligne par (seed, répétition) :

- [calcul](../results/compute-paired-2026-10-20.csv)
- [petites tâches](../results/micro-paired-2026-10-20.csv)
- [petites tâches préallouées](../results/resident-paired-2026-10-20.csv)
- [durée cachée](../results/hidden-paired-2026-10-20.csv)

## Pourquoi les résultats changent selon la charge

### 1. Coût prévisible : la partition statique gagne

Sur la charge de calcul, la somme médiane des événements CUDA est presque identique entre politiques : 46,410 à 46,714 ms. La différence de temps mural ne provient donc pas d'un travail GPU global différent. En revanche, l'écart des événements entre GPU vaut 6,977 ms pour round-robin et seulement 0,089 ms pour LPT. LPT distribue donc le travail visible beaucoup plus régulièrement et termine 7,18 % plus tôt.

feedback réduit aussi l'écart mais laisse 8,515 ms d'intervalles cumulés entre les tâches d'un même worker, contre 0,768 ms pour round-robin. Dans ce code, les confirmations séquentielles ajoutent donc des creux entre tâches. Elles annulent son bénéfice de rééquilibrage lorsque les tailles sont déjà connues.

### 2. Tâches courtes : le contrôle coûte plus que le calcul

Pour les petites tâches, la somme médiane des événements CUDA est de 2,851 ms avec feedback, alors que le lot prend 29,371 ms côté hôte. Les intervalles cumulés entre tâches atteignent 10,053 ms avec feedback, contre 1,254 ms pour round-robin. La file préremplie d'une politique statique est adaptée à ce lot très court ; attendre une confirmation entre décisions réduit le débit.

Le contrôle préalloué réduit le temps round-robin à 24,615 ms, mais feedback reste 14,32 % plus lent. Cela isole une conclusion prudente : les allocations par tâche ne suffisent pas à expliquer sa pénalité. Ce test conserve néanmoins validation, messages interprocessus et création de flux ; il ne sépare pas chacun de ces coûts.

### 3. Coût caché : le feedback traite le déséquilibre réel

Dans la charge cachée, toutes les tâches ont la même taille 3072. Huit exécutent douze GEMM et seize un seul, mais les politiques statiques ne reçoivent que la taille. L'ordre round-robin reçoit parfois une quantité très différente de répétitions par GPU :

| Groupe d'ordres | Seeds | Déséquilibre round-robin | Variation feedback |
|---|---|---:|---:|
| Déjà équilibré | 301, 302, 307, 308 | 56 contre 56 répétitions | de +1,23 % à +10,87 % |
| Déséquilibré | 300, 303–306, 309–311 | ratio de 1,489 à 3,870 | 8/8 gains, de -6,05 % à -32,53 % |

La seed 309 est la plus parlante : round-robin répartit 23 contre 89 répétitions et prend 224,800 ms ; feedback prend 151,669 ms, soit -32,53 %. À l'opposé, la seed 308 est équilibrée à 56 contre 56 et feedback est 10,87 % plus lent. Le résultat global, dont l'intervalle contient zéro, exprime précisément ce mélange de deux régimes. Il ne justifie pas d'activer feedback sans connaître les caractéristiques du lot.

## Vérification Nsight Systems

Une trace ciblée Nsight Systems 2025.1.1.0 a été enregistrée pour la seed 309 avec round_robin puis feedback, via :

    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      -o nsys-hidden python experiments/study.py \
      --workload hidden --seed-start 309 --seeds 1 --repetitions 1 \
      --policies round_robin feedback

Le rapport cuda_gpu_kern_sum affiche 248 instances de GEMM CUTLASS, qui représentent 98,8 % des temps de noyaux de cette trace, avec 543,763 ms cumulés. Il montre aussi des noyaux de remplissage et de réduction PyTorch, attendus pour la préparation et la validation. Cette trace confirme que le profilage suit bien les noyaux GPU de la charge. Le temps sous profilage (230,404 ms pour round-robin, 158,273 ms pour feedback) n'est pas mélangé aux résultats statistiques : l'instrumentation modifie le coût observé.

Les statistiques de la trace sont conservées ici, mais le fichier binaire .nsys-rep de 1,9 Mo et les JSON détaillés sont restés sur un conteneur sans volume persistant. Le pod est maintenant arrêté. Pour une prochaine campagne, il faut attacher un volume avant lancement afin de publier la trace et les métriques par tâche sans perte.

La lecture des rapports s'appuie sur la [documentation Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) et son [guide d'analyse](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).

## Limites et décision pratique

1. Une seule référence matérielle a été testée. Les résultats n'établissent ni généralité sur les GPU cloud, ni supériorité scientifique d'une politique.
2. Les tâches sont indépendantes, déjà résidentes et sans transferts. Aucun résultat ne concerne une charge avec copies, pression mémoire, multi-tenant, dépendances ou NVLink.
3. Les intervalles entre tâches des workers ne sont pas une mesure directe d'occupation GPU. Les temps d'attente de file élevés des politiques statiques reflètent surtout des tâches préenfilées, pas un surcoût de décision.
4. La recommandation pour cette implémentation est conditionnelle : utiliser une partition statique informée lorsqu'un coût prédictif est disponible ; réserver feedback aux lots dont les temps sont réellement imprédictibles et suffisamment longs pour amortir les messages.

## Reproduction

Les scripts publiés sont experiments/study.py, experiments/analyze.py, experiments/task_metrics.py, experiments/balance_metrics.py et experiments/hidden_analysis.py. Les commandes complètes figurent dans [le protocole](followup-protocol-2026-10-20.md). Pour régénérer les statistiques à partir d'un CSV publié :

    python experiments/analyze.py results/compute-paired-2026-10-20.csv
    python experiments/analyze.py results/hidden-paired-2026-10-20.csv
    python experiments/hidden_analysis.py results/hidden-paired-2026-10-20.csv
