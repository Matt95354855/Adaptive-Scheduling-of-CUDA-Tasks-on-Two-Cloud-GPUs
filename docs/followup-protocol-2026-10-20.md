# Étude de suivi : protocole du 20 octobre 2026

## Question et portée

Pour des tâches GEMM indépendantes déjà résidentes sur deux GPU identiques, mesurer quand le retour de fin de tâche compense son coût d'orchestration, et quand une partition statique informée par la taille suffit. L'étude ne mesure ni transfert hôte–GPU, ni transfert inter-GPU, ni graphe de dépendances. Les conclusions concernent cette implémentation PyTorch à deux processus et cette seule instance RunPod.

## Plans de tâches et blocs

- **Calcul** : 24 tâches, six exemplaires de chacune des tailles 1024, 2048, 3072, 4096. Douze permutations produites par `random.Random(seed).shuffle`, seeds 100 à 111. Deux répétitions initiales par seed et politique. Le script randomise l'ordre des six politiques dans chaque bloc `(seed, répétition)` avec `random.Random(seed * 1009 + répétition)`.
- **Petites tâches avec allocations** : 32 tâches, huit exemplaires de 256, 512, 768, 1024. Seeds 200 à 211, deux répétitions et six politiques, avec la même randomisation de l'ordre.
- **Petites tâches préallouées** : mêmes seeds, tailles et répétitions, mais seuls `single`, `round_robin`, `feedback`. Chaque worker réutilise les opérandes et le tampon de sortie par taille. Ce contrôle retire les allocations par tâche ; la validation et les échanges avec les workers restent dans la fenêtre mesurée. Les valeurs des matrices deviennent constantes à 1, ce qui garde le même GEMM mais change la donnée numérique : comparer les temps de façon descriptive, sans attribuer tout écart aux allocations seules.
- **Coût de tâche caché** : 24 tâches de même taille 3072 ; huit répètent le GEMM 12 fois et seize l'exécutent une fois. La répartition des répétitions est mélangée par `random.Random(seed ^ 0x5A17)` pour les seeds 300 à 311, avant tout chronométrage de cette série. Les ordonnanceurs statiques ne voient que `n`, tandis que le worker reçoit aussi le nombre de répétitions. Deux répétitions par seed, quatre politiques (`single`, `round_robin`, `size_lpt`, `feedback`). Cette charge synthétique teste le cas où la durée n'est pas prédictible à partir des métadonnées visibles. Elle n'implique pas que les applications réelles aient cette distribution.
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
python experiments/study.py --out results/hidden.json --workload hidden --seed-start 300 --seeds 12 --repetitions 2 --policies single round_robin size_lpt feedback
python experiments/analyze.py results/compute.csv
python experiments/task_metrics.py results/compute.json results/compute-derived.csv
python experiments/summarize_metrics.py results/compute-derived.csv
```

Les CSV publiés en format large ont une ligne par `(seed, répétition)` et un temps mural en millisecondes par politique, arrondi à 0,001 ms. `analyze.py` les lit directement. Le fichier `diagnostic-summary` conserve des médianes descriptives des mesures internes. Les versions détaillées peuvent être plus grandes que les extraits publiés ; toute perte de données due à l'absence de volume persistant est signalée dans les observations.

## Phase suivante : entrées hôte→GPU, exécutée le 21 octobre 2026

La campagne a été exécutée sans volume persistant, conformément à la demande de ne pas utiliser de stockage durable. Elle ajoute la charge `transfer` : 24 tâches indépendantes, huit exemplaires de 1024, 1536 et 2048, avec un ordre mélangé par seed. Pour chaque tâche, le worker crée deux matrices en mémoire hôte épinglée, les copie vers le GPU sélectionné, puis exécute le GEMM et valide la somme. Les JSON par tâche et le fichier binaire Nsight restent donc éphémères ; les mesures appariées et les agrégats de trace sont publiés dans le dépôt.

Cette première expérience de transfert mesure le coût de copies hôte→GPU dans la fenêtre du lot ; elle ne prétend pas encore modéliser la localité de données entre GPU. Les événements CUDA séparent le temps H2D du GEMM, et le script conserve aussi le temps de préparation de l'entrée hôte. La calibration de la politique `calibrated` inclut les copies depuis des tampons épinglés déjà alloués, mais le temps de création des entrées reste volontairement dans les tâches chronométrées.

Les politiques `single`, `round_robin`, `size_lpt`, `calibrated` et `feedback` ont été testées sur huit seeds et deux répétitions avec ordre de politiques randomisé. Le contrôle postérieur `--reuse-pinned-inputs`, motivé par le premier résultat, teste `single` et `round_robin` en conservant les copies H2D tout en préallouant les entrées hôte épinglées. La comparaison porte sur les temps appariés par seed ; les métriques secondaires sont les sommes H2D par GPU, les événements GEMM, le temps de préparation hôte et les intervalles de worker.

Une charge soutenue a ensuite été ajoutée après constat que le lot initial ne chargeait pas durablement les GPU : 12 tâches de tailles 4096, 5120 et 6144, quatre de chaque, avec 16 GEMM par tâche après la copie H2D. Elle compare `single` et `round_robin` sur quatre seeds et deux répétitions. Une surveillance `nvidia-smi` à environ une seconde relève les pics et les instants de charge simultanée ; elle est descriptive et ne remplace pas une mesure d'occupation par politique.
