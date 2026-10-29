# Campagne prête avant location des GPU, 29 octobre 2026

## Décision expérimentale

**Aucun pod n'est loué par cette préparation.** La prochaine campagne teste une
question laissée ouverte par les expériences précédentes : la pénalité du
feedback sous 40 processus CPU provient-elle surtout de la contention hôte,
de l'allocation répétée de mémoire épinglée, ou de leur interaction ?

Le [protocole factoriel amendé avant mesure](factorial-input-protocol-2026-10-24.md)
fixe l'ordre, les données numériques, les seeds et le critère statistique.
Les scripts et ce plan doivent être présents sur GitHub **avant** la location.

| Élément | Configuration fixée |
| --- | --- |
| Matériel | Un pod avec exactement **2 NVIDIA RTX PRO 4000 Blackwell**, au moins 23 Go par carte ; aucun volume persistant |
| Charge | `transfer_hidden` : 12 tâches indépendantes, matrices 4096, quatre tâches à 32 GEMM et huit à 4 GEMM, copies H2D conservées |
| Facteur CPU | H0 ou H40 (40 processus de hachage ; quota CPU constaté auparavant : 20,4 cœurs) |
| Facteur entrées | Tampons épinglés alloués par tâche ou réutilisés ; **valeur 1 dans les deux modes** via `--fixed-input-scale 1` |
| Politiques | `round_robin` et `feedback`, ordre randomisé par seed et répétition |
| Grille principale | Seeds **1500–1511**, deux répétitions : **48 cellules, 192 runs, 2 304 tâches** ; quatre combinaisons dans un ordre tournant équilibré |
| Vérification préalable | Seed 1499, H0, deux modes × deux politiques : **4 runs**, exclus de l'analyse |
| Profils Nsight secondaires | Seed 1512, une répétition de `feedback` pour chacune des quatre combinaisons : **4 runs**, exclus de l'analyse |

La statistique principale est, pour chaque seed, la différence
`(gain_H40 − gain_H0)_reused − (gain_H40 − gain_H0)_fresh`, où gain désigne le
pourcentage de réduction du makespan de feedback par rapport à round-robin.
Rapporter la médiane et l'IC bootstrap 95 % sur les **12 seeds** (10 000
tirages, graine 7). Le critère positif fixé précédemment est satisfait
uniquement si la borne inférieure de l'IC est supérieure à zéro. Les sorties
avec checksums faux, cellules manquantes ou doublons invalident le verdict ;
aucune seed ne sera retirée après mesure.

## Contrôles préparés sans GPU

- `experiments/factorial_host_input.py --plan-only` produit le plan de 48
  cellules, 192 runs et 2 304 tâches sans charger CUDA.
- Le runner refuse d'écraser un résultat existant, valide chaque cellule et
  sauvegarde un checkpoint JSON/CSV après **quatre runs**. `--resume` reprend
  uniquement un checkpoint entier, sur **le même pod et le même commit**.
- L'analyse vérifie la grille complète, la valeur numérique fixe, les
  politiques/répétitions uniques, les 12 identifiants de tâche et les 2 304
  checksums avant de calculer le contraste.
- Six tests CPU passent ; un passage complet de l'analyse sur **192 runs
  synthétiques**, non publiés comme mesures, a été vérifié localement.
- La [figure factorielle](../experiments/plot_factorial.py) est déjà codée et
  rendue sur ce jeu synthétique pour vérifier sa mise en page ; elle ne sera
  publiée avec des valeurs qu'après les mesures réelles.
- [L'audit de sortie](../experiments/audit_factorial_exports.py) vérifie les
  192 runs, 2 304 tâches, quatre traces Nsight si demandées et les fichiers
  requis, puis écrit un manifeste de tailles et SHA-256. Il a lui aussi passé
  un essai complet sur les données synthétiques.
- Les scripts shell passent `bash -n` et les sources Python `py_compile`.

## Ordre d'exécution après location

1. Vérifier dans RunPod la disponibilité, le prix affiché et l'absence de
   volume permanent. Si les deux RTX PRO 4000 ne sont pas disponibles, ne pas
   louer une autre configuration pour cette réplication.
2. Cloner le commit publié du dépôt sur le pod. Exécuter
   `bash experiments/run_factorial_campaign.sh` avec sortie dans
   `results/factorial-v2-console.log`. Le script refuse un autre type ou
   nombre de GPU, vérifie mémoire/disque/Nsight, enregistre le manifeste,
   lance les quatre smoke runs, puis la grille avec télémétrie SM à une
   seconde et analyses automatiques.
3. Copier les checkpoints JSON/CSV et le journal hors du pod régulièrement,
   idéalement après chaque groupe de quatre seeds. Si une interruption se
   produit, ne lancer `--resume` que sur le même pod et le même commit.
4. Après export de la grille complète, lancer
   `bash experiments/run_factorial_nsight.sh`. Les quatre traces incluent
   CUDA, NVTX et OS runtime ; elles sont **hors statistique**. Copier chaque
   `.nsys-rep`, JSON profilé, CSV `nsys stats` et journal immédiatement.
5. Publier sur GitHub les JSON bruts, CSV des 192 runs et 2 304 tâches,
   résultats par cellule et seed, résumé statistique, télémétrie, compteurs
   cgroup, manifestes, logs, quatre traces Nsight et leurs rapports. Vérifier
   les empreintes SHA-256 du manifeste après copie et avant d'arrêter le pod.
6. **Seulement si le budget le permet après cette publication**, lancer le
   [contrôle secondaire des workers persistants](persistent-pool-protocol-2026-10-29.md)
   avec `bash experiments/run_pool_campaign.sh`. Il prévoit 16 blocs appariés,
   64 runs et 768 tâches, plus quatre runs de vérification. Exporter les
   journaux GPU de chaque bloc, le JSON brut, les CSV et le résumé ; publier
   chaque fichier sur GitHub avant d'arrêter le pod. Générer les figures
   PNG/SVG après récupération des mesures, sur un environnement Matplotlib.
   Ne pas lancer ce contrôle
   si son coût estimé compromet la marge d'arrêt.

L'analyse est séquentielle : aucun autre calcul GPU ne tourne pendant la
grille principale. Les traces profilées ne contaminent pas les 192 mesures.
Les profils Nsight suivent la [syntaxe CLI documentée par NVIDIA](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

## Enveloppe de coût et ordre d'arrêt

Le dernier pod identique a affiché **1,14 $/h pour deux cartes**. Une
[publication RunPod](https://www.runpod.io/articles/guides/best-gpu-for-comfyui)
indiquait 0,57 $/h par RTX PRO 4000 en Secure Cloud au 10 septembre 2026 ;
**le tarif du panier RunPod au moment de la location fera foi**. À 1,14 $/h,
45 minutes coûtent environ **0,86 $** et 60 minutes **1,14 $**. La durée des
48 cellules H0/H40 et des exports reste incertaine ; ces chiffres sont une
enveloppe, pas une promesse de durée.

La limite de session reste **2 $**. Relever le solde au démarrage. À **1,60 $
dépensé**, ne plus lancer de nouveau profil ni de bloc secondaire : sauvegarder et pousser les
résultats présents. Arrêter le pod au plus tard vers **1,80 $ dépensé**
(10 % de marge), ou plus tôt si la grille et les exports sont terminés.
Si la grille est incomplète, publier son état sans verdict principal. Aucun
volume permanent n'est demandé ; les fichiers sont copiés avant l'arrêt.

## Résultats attendus et limites

Cette expérience peut soutenir ou réfuter le **signe du contraste
préenregistré**, ou rester inconclusive. Elle ne prouvera pas à elle seule
une cause microarchitecturale : le facteur H40 sollicite le CPU, les
allocations et les transferts peuvent interagir, et une seule instance ne
montre pas la généralisation. Nsight et les métriques par tâche servent à
expliquer les chronologies ; ils ne remplacent pas l'analyse appariée.
