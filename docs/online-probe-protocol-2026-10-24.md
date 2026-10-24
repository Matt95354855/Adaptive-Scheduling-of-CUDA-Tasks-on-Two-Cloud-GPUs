# Protocole fixé avant l'évaluation de la décision en ligne

## Question

Les quatre tâches longues sont indiscernables avant exécution sur la charge
`transfer_hidden`. La relation publiée entre gain de feedback et déséquilibre
round-robin utilise leurs durées vraies après coup. Ce bloc teste si quatre
terminaisons observées suffisent à choisir utilement entre une continuation
statique et une continuation au premier GPU disponible.

## Règle et contrôles

Les trois politiques `probe_rr`, `probe_feedback` et `probe_switch` exécutent
**les mêmes quatre premières tâches**, deux par GPU, avant d'envoyer les huit
suivantes. Ce préfixe commun contrôle le coût de l'attente nécessaire au
diagnostic. Après les quatre réponses, chaque tâche dont l'événement GEMM GPU
dépasse **60 ms** est classée comme longue, seuil fixé d'après les campagnes
précédentes où quatre et trente-deux GEMM étaient nettement séparés. Le code
ne lit ni le champ `repeats` ni la vraie classe de durée pour décider.

- `probe_rr` : envoyer les huit autres tâches selon leur index pair/impair ;
- `probe_feedback` : envoyer une tâche par GPU puis suivre les fins de tâche ;
- `probe_switch` : si les nombres de tâches longues observées diffèrent entre
  GPU, suivre les fins de tâche ; sinon, continuer en round-robin.

Les politiques historiques `round_robin` et `feedback` sans pause de sonde
sont conservées comme références secondaires.

## Grille et décision

- Seeds **1400–1423**, deux répétitions, ordre des cinq politiques randomisé,
  soit **240 runs** et 2 880 tâches attendus ; H0, entrées épinglées fraîches.
- Les deux répétitions sont résumées par médiane par seed et politique.
- Critère principal : gain médian de `probe_switch` par rapport à
  `probe_feedback`, avec IC bootstrap à 95 % sur les 24 seeds, 10 000 tirages
  et seed aléatoire 7. Une amélioration sera soutenue seulement si la borne
  inférieure de cet IC est positive. Les comparaisons aux autres contrôles,
  le taux de choix de chaque route et le coût de décision CPU sont secondaires.
- Les seeds ne sont pas filtrées selon le choix de route ou le résultat. Toute
  grille incomplète est signalée, sans verdict principal.

Une absence d'amélioration sera informative : un signal disponible en ligne
peut être trop faible, même si la stratification a posteriori est nette.
Le coût de la pause après la sonde fait partie du temps total ; il n'est pas
soustrait. Cette politique est une preuve de faisabilité, pas un ordonnanceur
général ni une optimisation de l'allocation mémoire.
