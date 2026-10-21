# Prochaine campagne préenregistrée : contention hôte et adaptation

## Question

La campagne précédente a montré que le cycle de vie des entrées hôte peut inverser
le meilleur ordonnanceur. Le test de saturation a aussi établi que ce pod est
plafonné à 20,4 cœurs CPU alors que la machine expose 48 processeurs logiques.
Cette campagne mesure si cette contention hôte change l'avantage potentiel de la
politique `feedback` lorsque les durées GPU sont volontairement cachées au
planificateur.

La charge principale est `transfer_hidden` : douze tâches 4096x4096 avec copies
H2D. Quatre tâches effectuent 32 GEMM et huit tâches quatre GEMM. La taille
visible est identique pour toutes les tâches, donc seul le retour de fin de tâche
permet à `feedback` de s'adapter.

## Conditions préparées

| Bloc | Contention hôte | Politiques | Seeds | Répétitions | Mesures appariées |
| --- | ---: | --- | ---: | ---: | ---: |
| H0 | 0 processus | `round_robin`, `feedback` | 800–811 | 2 | 24 paires |
| H40 | 40 processus CPU | `round_robin`, `feedback` | 800–811 | 2 | 24 paires |
| Réplication | condition gagnante de H0/H40 | mêmes politiques | 900–911 | 2 | 24 paires |

Chaque seed est une disposition cachée distincte. L'ordre des politiques est
aléatoire dans chaque répétition. Les processus de charge hôte commencent après
la calibration des GPU et avant le chronomètre de chaque lot : ils ne gonflent
donc pas directement le temps de démarrage. Ils représentent une contention CPU
contrôlée, pas une application hôte réelle.

## Mesures et vérifications

Avant chaque bloc, enregistrer le nom des GPU, la version de PyTorch, le pilote,
`nproc` et `cpu.max`. Pendant chaque bloc, relever l'utilisation GPU par
`nvidia-smi` et l'utilisation CPU du conteneur. Ne conserver sur GitHub que les
CSV, les agrégats et les conclusions ; le JSON brut et les traces restent
éphémères sur le pod.

Les critères sont déclarés avant l'exécution : `feedback` sera décrit comme
robuste dans une condition seulement si son intervalle bootstrap apparié à 95 %
sur le pourcentage de gain exclut zéro et s'il gagne au moins 9 des 12 seeds.
La comparaison H0–H40 portera sur la variation seed par seed du gain de
`feedback` contre `round_robin`, pas sur des moyennes non appariées.

## Commandes prêtes à exécuter

```bash
python experiments/study.py --out results/hidden-h0.json \
  --workload transfer_hidden --seed-start 800 --seeds 12 --repetitions 2 \
  --policies round_robin feedback

python experiments/study.py --out results/hidden-h40.json \
  --workload transfer_hidden --host-load-workers 40 --seed-start 800 \
  --seeds 12 --repetitions 2 --policies round_robin feedback
```

Après chaque bloc, pousser immédiatement le CSV et une note d'observation.
Lancer le bloc de réplication uniquement après l'analyse appariée de H0 et H40,
sur la condition qui donne le signal le plus crédible.

## Raisonnement

Le test sépare deux mécanismes que les résultats actuels confondent : la capacité
de `feedback` à corriger une durée GPU inconnue, et le coût CPU de préparer et
envoyer les tâches. Il complète les contrôles déjà effectués sur l'allocation
des entrées, la réutilisation de mémoire épinglée et la charge GPU soutenue.
