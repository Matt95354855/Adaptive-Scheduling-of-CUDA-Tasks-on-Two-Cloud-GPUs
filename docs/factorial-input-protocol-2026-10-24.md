# Protocole factoriel hôte × durée de vie des entrées

Ce bloc est fixé avant les mesures. Il cherche à départager la contention CPU
et l'allocation de mémoire hôte épinglée par tâche, tout en conservant les
copies H2D et les mêmes dimensions/GEMM. Il sera exécuté après les blocs de
réplication et de décision en ligne, seulement si le plafond de 2 $ le permet.

## Amendement du 29 octobre 2026, avant la première mesure factorielle

Aucun run de cette grille n'avait été exécuté au moment de cet amendement.
Dans la version du 24 octobre, les entrées fraîches variaient numériquement
selon la seed et les entrées réutilisées valaient un. Ce détail brouillait
la comparaison de l'allocation. Les deux modes utiliseront maintenant des
matrices de valeur **1** au moyen de `--fixed-input-scale 1` ; ils conservent
exactement les mêmes dimensions, répétitions GEMM et copies H2D. La seule
différence prévue du facteur « entrée » est l'allocation épinglée par tâche
ou la réutilisation de tampons déjà alloués. Les seeds, l'ordre des cellules,
les politiques, le contraste principal et son critère restent identiques.
Le code et ce texte seront publiés avant la location du GPU.

## Grille

- Charge `transfer_hidden`, seeds **1500–1511**, deux répétitions, politiques
  `round_robin` et `feedback` : **192 runs** attendus et 2 304 tâches.
- Facteurs : H0 ou H40 ; entrées épinglées allouées par tâche (`fresh`) ou
  réutilisées (`reused`). Les copies H2D restent présentes dans les deux cas.
- Les deux modes emploient `--fixed-input-scale 1` ; chaque tâche conserve
  le même calcul et les **12 sorties sont vérifiées** dans chaque run. Quatre runs de vérification H0, seed 1499,
  sont exclus des 192 runs statistiques.
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

Le choix d'une entrée constante retire le changement de contenu numérique
entre les modes. Il ne reproduit pas exactement les campagnes historiques
à échelle variable ; celles-ci ne sont pas fusionnées à cette grille.
L'étude reste limitée à une charge synthétique et une instance.

Avant l'arrêt du pod, publier les JSON bruts, CSV par run et par tâche,
diagnostics, code et conclusions. Les runs profilés avec Nsight sont exclus
des 192 runs statistiques.

Les quatre profils Nsight secondaires utilisent la seed 1512, une politique
`feedback` et une répétition pour chaque combinaison entrée × H0/H40. Les
traces `.nsys-rep` et leurs statistiques CUDA/NVTX seront copiées hors du pod
avant son arrêt. Les profils ne servent pas à la décision statistique.
