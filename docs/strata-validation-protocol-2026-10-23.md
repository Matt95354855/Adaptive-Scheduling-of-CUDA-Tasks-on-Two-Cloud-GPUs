# Validation prospective de l'effet du déséquilibre statique

## Motif et statut

L'étude H0/H40 et sa réplication sur 24 seeds a révélé après coup une relation
entre le gain de `feedback` et la répartition, par `round_robin`, des quatre
tâches longues à 32 GEMM. Cette relation est exploratoire. Le présent protocole
est fixé avant la mesure de nouvelles seeds sur le même pod.

## Expérience

- Charge `transfer_hidden`, entrées hôte épinglées allouées par tâche, sans
  processus de contention CPU supplémentaire (`host_load_workers=0`).
- Seeds contiguës **1000–1023**, deux répétitions par seed ; ordre des politiques
  randomisé par le banc d'essai. `round_robin` et `feedback` seulement.
- Même matériel, logiciel, code de charge et contrôle de somme que H0.
- 96 runs attendus, 24 seeds indépendantes. Aucun retrait de seed après mesure ;
  les échecs sont consignés et l'analyse ne sera pas déclarée complète.

## Variable et analyse fixées avant mesure

Pour chaque seed, compter sur les résultats par tâche de `round_robin` le nombre
de tâches à 32 GEMM affectées à chaque GPU. Le groupe **équilibré** est 2–2
(déséquilibre 0). Le groupe **déséquilibré** est 3–1 ou 4–0 (déséquilibre 2 ou 4).
Ce classement dépend de l'ordre statique, pas du makespan observé.

Résumer d'abord les deux répétitions par la médiane du makespan de chaque
politique pour chaque seed, puis calculer le gain en pourcentage
`100 × (1 − feedback / round_robin)`. Les seeds, et non les répétitions, sont
les unités statistiques. Le critère principal est la **différence des médianes
de gain** entre groupe déséquilibré et groupe équilibré, avec IC bootstrap à
95 % obtenu par rééchantillonnage des seeds à l'intérieur de chaque groupe,
10 000 tirages et seed aléatoire 7. Le signal sera validé sur cette instance
si la borne inférieure de cet IC est strictement positive et si chaque groupe
contient au moins six seeds. Les gains médians, IC et nombres de victoires de
chaque groupe sont secondaires. La taille des groupes n'est pas connue avant
l'exécution et ne sera pas ajustée en choisissant d'autres seeds.

Cette validation porte sur un prédicteur connu seulement a posteriori, car les
durées réelles sont volontairement cachées au planificateur. Elle ne démontre
donc pas qu'une politique peut sélectionner automatiquement le bon régime.

```bash
python experiments/study.py --out results/strata-validation.json \
  --workload transfer_hidden --seed-start 1000 --seeds 24 --repetitions 2 \
  --policies round_robin feedback
```

Après analyse, publier les 96 runs, la chronologie par tâche, le calcul par
seed, les agrégats et la conclusion, puis arrêter le pod sans dépasser le
budget de 2 dollars prévu pour cette session.
