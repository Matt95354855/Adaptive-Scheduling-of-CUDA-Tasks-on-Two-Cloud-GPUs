# Contrôle de charge soutenue sur deux GPU, 24 octobre 2026

## Protocole et résultat

Le [protocole](sustained-utilization-protocol-2026-10-24.md) a été publié avant ce bloc. La charge `transfer_heavy` utilise 12 tâches par run, des matrices 4096–6144, une copie H2D par tâche et 64 GEMM par tâche. Les entrées épinglées sont réutilisées. Les seeds 1600–1603 ont chacune deux répétitions pour `round_robin` et `feedback` : **16/16 runs et 192/192 tâches** enregistrés ; tous les contrôles de somme passent. Un checkpoint de 11 runs a été poussé pendant le calcul pour protéger les données du pod éphémère ; le fichier final est complet.

| Mesure | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Occupation SM moyenne sur 153 relevés d'une seconde | **48,37 %** | **49,47 %** |
| Occupation SM médiane | 40 % | 54 % |
| Relevés SM ≥ 70 % | 45,75 % | 48,37 % |
| Relevés SM = 0 % | 41,18 % | 43,79 % |
| Médiane du temps d'événements CUDA / makespan, par run | **96,27 %** | **97,41 %** |

Les deux GPU sont donc occupés presque constamment **pendant les fenêtres de lots** ; des relevés individuels atteignent 100 %. L'objectif de **70 % d'occupation moyenne sur toute la campagne** n'est **pas atteint**. La mesure d'ensemble inclut le lancement des processus PyTorch, leur calibration et les intervalles entre runs. Le rapport d'événements CUDA est une mesure de temps dans la fenêtre chronométrée, pas un compteur d'occupation SM ; les deux colonnes ne sont pas interchangeables. La charge dense a néanmoins relevé l'occupation moyenne de campagne d'environ 14 % pour `transfer_hidden` à environ 49 % ici, sans que ces charges aient la même question de scheduling.

Le makespan médian descriptif est **4 913,3 ms** pour round-robin et **4 571,1 ms** pour feedback. Par seed appariée, le gain médian de feedback est **+4,86 %**, avec 3 victoires sur 4 seeds. Quatre seeds ne permettent pas un verdict de supériorité. Cette charge change les tailles, le nombre de GEMM et le cycle de vie des entrées : elle ne réplique donc pas le test de durées cachées.

## Portée et suite

Le résultat met en évidence deux goulots distincts. Pendant un lot suffisamment long, les deux cartes calculent presque sans interruption ; entre lots, la création et la calibration répétées des workers dominent le temps facturé. La prochaine modification utile est de conserver un pool de workers et des entrées entre lots, puis de mesurer simultanément makespan, latence complète d'une requête et occupation de campagne. Une nouvelle trace Nsight de ce bloc reste à faire après la pause ; les profils des campagnes antérieures sont conservés sous forme d'agrégats dans le dépôt.

## Données intégrales

- [JSON brut](../results/sustained-2026-10-24.json), [CSV des 16 runs](../results/sustained-2026-10-24.csv), [CSV des 192 tâches](../results/sustained-tasks-2026-10-24.csv)
- [Métriques par run](../results/sustained-task-metrics-2026-10-24.csv), [télémétrie GPU brute](../results/sustained-gpu-util-2026-10-24.log), [résumé d'occupation](../results/sustained-utilization-summary-2026-10-24.json)
- [Journal d'exécution](../results/sustained-2026-10-24.log), [checkpoint de secours](../results/sustained-checkpoint-2026-10-24.json)
- [Script de campagne](../experiments/run_sustained_control.sh), [analyse d'occupation](../experiments/analyze_utilization.py)
