# Étude de contention hôte, 23 octobre 2026

## État de la campagne

Le bloc H0 du [protocole préenregistré](next-campaign-protocol-2026-10-21.md)
est terminé. H40 et la réplication restent à mesurer. Il s'agit d'un état
intermédiaire, sans conclusion sur l'effet de la contention CPU.

## Instance et contrôle

Pod RunPod `twqrwkf0s7u2nu`, deux NVIDIA RTX PRO 4000 Blackwell de 24 467 MiB,
pilote 595.91.07, PyTorch 2.8.0+cu128, runtime CUDA 12.8. `nproc` signale 48
processeurs logiques ; `cpu.max` vaut `2040000 100000`, soit un quota de 20,4
cœurs. La topologie `nvidia-smi topo -m` affiche `NODE` entre les GPU. Elle
diffère du `PHB` relevé lors de la campagne précédente ; les mesures entre
instances ne sont donc pas fusionnées. Le conteneur est éphémère, sans volume
persistant.

Code exécuté : commit `0adf59b9e4c17eabc0e90d7b6a0e0e3122af215d`.
Chaque tâche a passé la vérification de somme du banc d'essai.

## H0 : sans charge CPU supplémentaire

Douze seeds indépendantes (800–811), deux répétitions chacune, deux politiques
dans un ordre randomisé : 48 mesures. Les répétitions sont résumées par seed
avant la comparaison appariée. Le pourcentage ci-dessous est la médiane des
changements par seed par rapport au tourniquet.

| Politique | Temps médian par seed | Changement apparié | Victoires | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | ---: |
| `round_robin` | 679,396 ms | référence | — | — |
| `feedback` | 573,896 ms | −16,18 % | 9/12 | [−17,99 ; +2,23] % |

Le seuil préenregistré de 9 victoires est atteint, mais l'intervalle inclut
zéro. Le résultat est prometteur et encore incertain ; H40 et la réplication
doivent départager cette hypothèse.

Les métriques médianes par run montrent des sommes de GEMM comparables
(789,915 ms pour round-robin, 787,864 ms pour feedback) et des copies H2D
également proches (84,900 et 85,407 ms). Les temps de préparation médian par
tâche sont 2,774 et 2,753 ms. La médiane d'attente dans la file vaut 253,079
ms avec les tâches préaffectées de round-robin et 0,200 ms avec l'envoi en ligne
de feedback : cette métrique reflète surtout la façon de remplir les files,
et ne représente pas un coût GPU additionnel. Les intervalles entre tâches de
worker sont plus grands avec feedback, cohérents avec ses échanges de contrôle.

- [48 mesures individuelles H0](../results/transfer-hidden-h0-2026-10-23.csv)
- [Diagnostics par run H0](../results/transfer-hidden-h0-task-metrics-2026-10-23.csv)

L'échantillonnage GPU à une seconde comprend les calibrations hors fenêtre ; il
ne sera pas utilisé pour comparer les politiques séparément.
