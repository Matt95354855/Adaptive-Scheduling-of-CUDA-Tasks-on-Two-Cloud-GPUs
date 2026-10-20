# Observations du pilote RunPod — 20 octobre 2026

## Statut

Il s'agit d'un pilote de diagnostic sur une seule instance, avec trois répétitions par cellule. Les chiffres ci-dessous ne démontrent ni supériorité générale d'un ordonnanceur, ni nouveauté scientifique. Le fichier [`results/pilot-summary-2026-10-20.csv`](../results/pilot-summary-2026-10-20.csv) contient les 36 mesures individuelles relevées pendant la session. Les JSON par tâche et la trace binaire Nsight n'ont pas été transférés hors du pod ; ils ne sont donc pas disponibles dans ce dépôt.

## Instance et reproductibilité

- RunPod, pod `wtd0haxfn39yuy`, deux NVIDIA RTX PRO 4000 Blackwell identiques, environ 24 Go de VRAM par GPU, 24 vCPU AMD EPYC 7352 et 62 Go de RAM selon l'interface.
- Tarif affiché : 1,14 $/h pour les deux GPU. Le solde affiché est passé de 8,19 $ avant le démarrage à 7,94 $ après l'arrêt, soit 0,25 $ de différence visible, sous le plafond de 2 $. L'interface indique désormais 0,00 $/h pour le pod arrêté. Ce relevé de solde n'est pas une facture détaillée.
- Pilote NVIDIA 595.91.07 ; `nvidia-smi` annonce CUDA 13.2 ; toolkit `nvcc` 12.8.93 ; Python 3.12.3 ; PyTorch 2.8.0+cu128 avec runtime CUDA 12.8.
- Les deux GPU sont visibles dans PyTorch. `nvidia-smi topo -m` indique `PHB` entre eux, sans lien NVLink affiché. Les deux lignes annoncent la même affinité CPU `0-47` et NUMA `0`.
- Le test PyTorch déterministe a produit des résultats conformes sur toutes les cellules. La vérification porte sur la somme analytique d'une multiplication de matrices constantes, tolérance relative `1e-4` ; ce n'est pas une validation élément par élément.
- Les tâches opèrent sur des matrices créées directement sur chaque GPU. Aucun transfert H2D ou D2D n'est mesuré. Chaque politique démarre de nouveaux processus et effectue un warm-up hors de la fenêtre du lot. La fenêtre inclut l'envoi des tâches, les allocations et vérifications dans les workers, ainsi que les attentes côté hôte. Les événements CUDA encadrent seulement `torch.mm`.

## Résultats

Médiane du temps de complétion du lot, en millisecondes (`n=3` par politique) :

| Charge | Un GPU | Deux GPU round-robin | Deux GPU réactif | Réactif face au round-robin |
|---|---:|---:|---:|---:|
| 32 petites tâches, seed 42 | 29,240 | 25,512 | 29,522 | 15,7 % plus lent |
| 24 tâches 1024–4096, ordre périodique, seed 42 | 75,304 | 57,934 | 53,559 | 7,5 % plus rapide |
| Même tailles mélangées, seed 43 | 74,728 | 57,290 | 52,590 | 8,2 % plus rapide |
| Même tailles mélangées, seed 58 choisie pour équilibrer le round-robin | 74,286 | 48,587 | 52,185 | 7,4 % plus lent |

Le contrôle seed 58 a été choisi **après** avoir calculé l'équilibre théorique du round-robin ; il sert à tester un cas limite, pas à estimer une performance moyenne sur des ordres aléatoires. Le rapport inverse dans ce cas, conformément à l'hypothèse que le feedback coûte plus qu'il ne rapporte quand la répartition statique est déjà bonne. Trois répétitions et une seule machine ne suffisent pas à quantifier l'incertitude ou à généraliser.

Les affectations du premier run expliquent l'essentiel de l'écart sur la charge périodique : round-robin confie 12 tâches à chaque GPU, mais environ 180,4 et 463,9 milliards d'unités `n³` respectivement ; les sommes des événements CUDA sont 15,042 et 32,780 ms. La politique réactive confie 16 et 8 tâches, mais exactement 322,1 milliards d'unités `n³` à chaque GPU, avec 24,742 et 23,183 ms d'événements CUDA. Sur la seed 58, le round-robin répartit exactement 322,1 milliards d'unités `n³` par GPU, avec 23,805 et 23,658 ms d'événements CUDA, et gagne. Le nombre de tâches par GPU n'est donc pas un indicateur suffisant d'équilibrage.

Sur les petites tâches, les sommes des événements CUDA ne représentent qu'environ 3–4 ms par lot, contre 25–30 ms de temps hôte. Les coûts côté hôte, les allocations, la vérification des sorties et les communications entre processus pèsent donc fortement dans cette expérience. `dispatch_ms` mesure uniquement le temps passé à déposer les demandes dans les files ; il ne mesure pas tout le coût de décision ni tout le temps d'attente. Le coût réel de l'ordonnanceur reste à isoler.

## Nsight Systems

Nsight Systems 2025.1.1.0 était présent dans l'image à `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`, sans être dans `PATH`. Une capture `--trace=cuda,nvtx --sample=none` du smoke test a abouti et a produit `nsight-smoke.nsys-rep` sur le pod. Le rapport `cuda_gpu_kern_sum` affichait des noyaux GEMM CUTLASS/cuBLASLt, de réduction PyTorch et de remplissage ; la ligne GEMM la plus coûteuse regroupait 9 instances et 726 547 ns, soit 63,4 % du temps total des noyaux de cette trace. Le rapport `nvtx_sum` contenait les plages `task_<id>_n<size>_gpu<gpu>` émises par le code. Ces statistiques confirment que la capture contient des travaux CUDA attribuables aux tâches ; elles ne suffisent pas à expliquer seules les écarts de temps hôte.

La trace binaire et les JSON détaillés étaient stockés dans le conteneur sans volume persistant. L'export direct via un proxy public a été refusé par la revue automatique pour risque d'exposition des résultats bruts. Le résumé CSV ci-joint est la copie conservée. Pour une réplication, attacher d'abord un stockage persistant ou configurer une méthode de transfert restreinte.

## Interprétation et suite

Cette politique dite « adaptive » est seulement un ordonnanceur réactif : elle donne la prochaine tâche au premier GPU libre. Elle n'est pas encore consciente du coût des transferts, de la mémoire ou d'un modèle de durée par GPU. Les résultats soutiennent une conclusion limitée : avec cette implémentation et ces tâches indépendantes, le feedback aide lorsque le round-robin est déséquilibré, mais ajoute un coût visible sur une charge courte ou déjà équilibrée. Il faut ensuite comparer au minimum une partition statique équilibrée, une politique `least-loaded` fondée sur un coût calibré et plusieurs ordres non sélectionnés, en randomisant l'ordre des politiques. Les mesures futures devront inclure une charge avec données réutilisées ou transférées, un protocole de sauvegarde des traces et des intervalles d'incertitude.

## Commandes de réplication

Voir [`docs/protocol.md`](protocol.md). Le code a été exécuté à différentes étapes du dépôt : smoke et petites tâches depuis `1429349`, charge périodique depuis `9e50245`, et contrôles mélangés depuis `8563138`. Les changements entre ces révisions ajoutent seulement les options de charge et de mélange ; le worker et les politiques sont restés identiques.
