# Étude de suivi : protocole du 20 octobre 2026

## Question et portée

Pour des tâches GEMM indépendantes déjà résidentes sur deux GPU identiques, mesurer quand le retour de fin de tâche compense son coût d'orchestration, et quand une partition statique informée par la taille suffit. L'étude ne mesure ni transfert hôte–GPU, ni transfert inter-GPU, ni graphe de dépendances. Les conclusions concernent cette implémentation PyTorch à deux processus et cette seule instance RunPod.

## Plans de tâches et blocs

- **Calcul** : 24 tâches, six exemplaires de chacune des tailles 1024, 2048, 3072, 4096. Douze permutations produites par `random.Random(seed).shuffle`, seeds 100 à 111. Deux répétitions initiales par seed et politique. Le script randomise l'ordre des six politiques dans chaque bloc `(seed, répétition)` avec `random.Random(seed * 1009 + répétition)`.
- **Petites tâches avec allocations** : 32 tâches, huit exemplaires de 256, 512, 768, 1024. Seeds 200 à 211, deux répétitions et six politiques, avec la même randomisation de l'ordre.
- **Petites tâches préallouées** : mêmes seeds, tailles et répétitions, mais seuls `single`, `round_robin`, `feedback`. Chaque worker réutilise les opérandes et le tampon de sortie par taille. Ce contrôle retire les allocations par tâche ; la validation et les échanges avec les workers restent dans la fenêtre mesurée. Les valeurs des matrices deviennent constantes à 1, ce qui garde le même GEMM mais change la donnée numérique : comparer les temps de façon descriptive, sans attribuer tout écart aux allocations seules.
- Un bloc complémentaire de répétition du calcul peut être lancé si les deux premières répétitions exposent des valeurs extrêmes. Il sera identifié séparément dans les résultats : il est motivé par les données observées et ne faisait pas partie du plan initial.

## Politiques

- `single` : toutes les tâches sur le GPU 0 ; le deuxième processus est initialisé mais reste inactif pendant le lot.
- `round_robin` : répartition alternée, sans connaissance du coût.
- `size_fifo` : parcours dans l'ordre d'entrée ; prochaine tâche au GPU dont la somme assignée de `n³` est la plus petite.
- `size_lpt` : affectation des plus grandes tâches d'abord au GPU le moins chargé en `n³`, puis exécution dans l'ordre d'entrée de chaque file. Cette politique connaît tout le lot et constitue un contrôle statique avec davantage d'information.
- `calibrated` : même parcours FIFO, avec estimation par GPU et par taille obtenue par médiane de cinq événements CUDA hors de la fenêtre chronométrée. La décision minimise la fin prédite après ajout de la tâche.
- `feedback` : une tâche en vol par GPU ; le GPU qui confirme une fin reçoit la suivante. Cette politique est réactive, sans modèle de durée, de transfert ou de mémoire. L'appellation « adaptive » du premier pilote désignait cette même idée.

Les politiques statiques envoient toutes leurs tâches au début du lot. Le retour de fin de tâche impose au contraire une boucle de messages hôte–worker pour `feedback`. Les coûts de cette boucle font partie de la comparaison de temps de lot ; la calibration initiale et la création des processus n'en font pas partie. Cette asymétrie est intentionnelle pour mesurer le débit de lots indépendants disponibles à l'avance, et limite la portée à ce scénario.

## Mesures et validation

Chaque politique démarre deux processus `spawn`, initialise CUDA, chauffe chaque taille et calibre cinq GEMM par taille/GPU. Le chronomètre monotone du parent commence juste avant les envois et s'arrête après la réception de toutes les confirmations. Chaque worker crée les matrices pendant la fenêtre, sauf en mode préalloué. Un événement CUDA encadre seulement `torch.mm`, puis une somme de sortie est comparée à une somme analytique avec tolérance relative `1e-4`. Toute erreur de somme invalide le lot. Le contrôle n'est pas une vérification élément par élément.

Le JSON brut enregistre, pour chaque tâche, l'heure de dépôt, de début et de fin dans le worker, de réception par le parent et le temps de l'événement CUDA. `task_metrics.py` en déduit notamment les intervalles entre tâches d'un même worker. Ce sont des **intervalles du worker**, pas une mesure directe d'occupation GPU ; ils excluent les temps avant la première et après la dernière tâche. Les sommes d'événements CUDA des deux GPU ne sont pas un temps mural et ne doivent pas être soustraites directement du temps de lot.

`analyze.py` prend la médiane des répétitions pour chaque seed, puis la médiane des douze écarts en pourcentage contre `round_robin`. Il rééchantillonne les douze seeds 10 000 fois pour un intervalle bootstrap descriptif à 95 %. Les répétitions ne sont pas traitées comme autant de seeds indépendantes. Les intervalles décrivent la variation des ordres de tâches sur cette machine, pas une population de GPU cloud.

## Commandes

```bash
python experiments/study.py --out results/compute.json --seeds 12 --repetitions 2
python experiments/study.py --out results/micro.json --workload micro --seed-start 200 --seeds 12 --repetitions 2
python experiments/study.py --out results/resident.json --workload micro --resident --seed-start 200 --seeds 12 --repetitions 2 --policies single round_robin feedback
python experiments/analyze.py results/compute.csv
python experiments/task_metrics.py results/compute.json results/compute-derived.csv
python experiments/summarize_metrics.py results/compute-derived.csv
```

Les CSV publiés en format large ont une ligne par `(seed, répétition)` et un temps mural en millisecondes par politique, arrondi à 0,001 ms. `analyze.py` les lit directement. Le fichier `diagnostic-summary` conserve des médianes descriptives des mesures internes. Les versions détaillées peuvent être plus grandes que les extraits publiés ; toute perte de données due à l'absence de volume persistant est signalée dans les observations.
