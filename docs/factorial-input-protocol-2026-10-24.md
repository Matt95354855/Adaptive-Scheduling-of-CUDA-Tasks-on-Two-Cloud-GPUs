# Protocole factoriel hôte × durée de vie des entrées

Ce bloc est fixé avant les mesures. Il cherche à départager la contention CPU
et l'allocation de mémoire hôte épinglée par tâche, tout en conservant les
copies H2D et les mêmes dimensions/GEMM. Il sera exécuté après les blocs de
réplication et de décision en ligne, seulement si le plafond de 2 $ le permet.

## Grille

- Charge `transfer_hidden`, seeds **1500–1511**, deux répétitions, politiques
  `round_robin` et `feedback` : **192 runs** attendus et 2 304 tâches.
- Facteurs : H0 ou H40 ; entrées épinglées allouées par tâche (`fresh`) ou
  réutilisées (`reused`). Les copies H2D restent présentes dans les deux cas.
- Pour chaque seed, exécuter les quatre combinaisons dans une séquence de type
  carré latin. La séquence de base est `fresh/H0`, `reused/H40`, `fresh/H40`,
  `reused/H0`, tournée d'une position pour chaque seed suivante. Chaque
  combinaison apparaît ainsi trois fois à chaque rang d'exécution.
- L'ordre des politiques dans une combinaison reste randomisé par
  `experiments/study.py`. Aucun retrait de seed après mesure. Contrôle de
  somme sur les 2 304 tâches.

## Analyse fixée

Chaque cellule est résumée par la médiane des deux répétitions par seed et
politique. Calculer le gain de feedback par rapport à round-robin, puis
`interaction = gain_H40 − gain_H0` pour chaque mode d'entrées. Le contraste
principal est, par seed, `interaction_reused − interaction_fresh` ; rapporter
sa médiane et un IC bootstrap à 95 % de 10 000 tirages, seed 7. Un contraste
positif sera soutenu seulement si la borne inférieure de l'IC est positive.
Les temps de préparation, H2D, événements GPU, intervalles de worker et
makespans sont des diagnostics secondaires.

La réutilisation emploie des entrées constantes de valeur 1, tandis que le
mode frais varie l'échelle numérique selon la seed ; la taille et le nombre
d'opérations GEMM sont identiques, mais cette différence de contenu doit être
mentionnée. L'étude reste limitée à une charge synthétique et une instance.

Avant l'arrêt du pod, publier les JSON bruts, CSV par run et par tâche,
diagnostics, code et conclusions. Les runs profilés avec Nsight sont exclus
des 192 runs statistiques.
