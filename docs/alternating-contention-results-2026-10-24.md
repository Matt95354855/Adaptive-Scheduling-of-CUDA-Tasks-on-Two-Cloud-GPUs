# Contention CPU avec ordre alterné, 24 octobre 2026

## Protocole et intégrité

Le [protocole du bloc B](replication-protocol-2026-10-24.md) était publié avant la mesure. Sur le pod `zgi4mo43z7dkw9` à deux RTX PRO 4000 Blackwell, les seeds 1200–1211 ont été exécutées deux fois par politique (`round_robin`, `feedback`) et par niveau de contention CPU (H0, H40). L'ordre H0→H40 et H40→H0 alterne selon la seed. Les **96 runs et 1 152 tâches** attendus sont présents ; les contrôles de somme sont satisfaits. H40 lance 40 processus de hachage CPU, sur un pod dont le quota est de 20,4 cœurs.

## Comparaison appariée

Le gain positif signifie que `feedback` réduit le makespan par rapport à `round_robin` pour la même seed et la même condition.

| Condition | Gain médian feedback | Victoires par seed | IC bootstrap 95 % |
| --- | ---: | ---: | ---: |
| H0 | +5,57 % | 8/12 | [−7,08 ; +18,09] % |
| H40 | −4,63 % | 4/12 | [−9,53 ; +5,42] % |

Le contraste préenregistré, `gain_H40 − gain_H0`, vaut **−13,08 points de pourcentage**, IC bootstrap 95 % **[−21,81 ; +5,19]**. La direction est compatible avec une dégradation du feedback sous forte contention CPU, mais l'intervalle inclut zéro : cette expérience ne démontre pas le signe de l'interaction. Par ordre d'exécution, le contraste médian descriptif est −17,39 points pour les six seeds H0→H40 et −5,49 points pour les six seeds H40→H0. L'écart appelle un contrôle plus fin ; les sous-groupes sont trop petits pour un verdict.

## Diagnostic des tâches et limites causales

Les médianes descriptives de préparation d'une entrée passent d'environ **2,7–2,8 ms** sous H0 à **95,5–98,1 ms** sous H40. La somme des durées d'événements GPU par run reste proche : **786–789 ms** sous H0 et **776–777 ms** sous H40. Le makespan médian passe de **616,7 à 1 319,1 ms** pour `round_robin` et de **581,9 à 1 399,1 ms** pour `feedback`. Les intervalles entre tâches sur les deux workers sont autour de **0,8–0,9 ms** pour `round_robin` ; sous H40 ils atteignent **11,0 et 7,9 ms** pour `feedback`.

Ces mesures situent le surcoût surtout autour de la préparation et de l'ordonnancement hôte, pas dans une hausse du temps des noyaux CUDA. Elles ne séparent pas encore l'effet de la contention CPU de celui de l'allocation répétée des entrées épinglées. Le [plan factoriel H0/H40 × entrées fraîches/réutilisées](factorial-input-protocol-2026-10-24.md) a été fixé avant la mesure suivante pour isoler cette interaction.

La télémétrie instantanée peut afficher 0 % entre de brèves pointes de calcul. Un échantillonnage `nvidia-smi dmon` de 15 secondes pendant le bloc suivant a observé des pointes de 100 % et 97 % sur les deux cartes, ainsi que beaucoup de secondes à 0 %. Il faut donc distinguer durée des noyaux, occupation moyenne et makespan ; ces données ne prouvent pas une saturation continue du pod.

## Données intégrales

- [JSON brut par run et par tâche](../results/alternating-2026-10-24.json)
- [CSV des 96 runs](../results/alternating-2026-10-24.csv)
- [CSV des 1 152 tâches](../results/alternating-tasks-2026-10-24.csv)
- [Diagnostics par run](../results/alternating-2026-10-24-task-metrics.csv)
- [Comparaison H0 par seed](../results/alternating-2026-10-24-h0.csv)
- [Comparaison H40 par seed](../results/alternating-2026-10-24-h40.csv)
- [Résumé statistique](../results/alternating-2026-10-24-summary.json)
- [Médianes diagnostiques](../results/alternating-2026-10-24-diagnostics.json)
- [Journal d'exécution](../results/alternating-2026-10-24.log)
