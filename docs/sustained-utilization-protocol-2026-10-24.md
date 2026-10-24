# Contrôle prospectif de charge GPU soutenue

## Motivation et séparation des blocs

Pendant la campagne `transfer_hidden`, un échantillonnage `nvidia-smi dmon` de 15 secondes a montré de nombreux relevés à 0 % et des pointes de 100 % et 97 % sur GPU 0 et 1. Les runs courts alternent démarrage des workers, préparation hôte, copies et calcul. Le présent contrôle teste une charge où le calcul GEMM domine, sans ajouter une tâche concurrente aux campagnes de scheduling déjà préenregistrées. Il est lancé seulement après leur fin et ses mesures ne sont pas fusionnées avec elles.

## Plan fixé avant mesure

- Pod : les mêmes deux RTX PRO 4000 Blackwell, H0.
- Charge `transfer_heavy` : 12 tâches de matrices 4096, 5120 et 6144, quatre de chaque taille, entrées épinglées réutilisées, une copie H2D par tâche, **64 GEMM par tâche**.
- Deux politiques `round_robin` et `feedback`, seeds **1600–1603**, deux répétitions, ordre randomisé selon `study.py` : 16 runs et 192 tâches attendus.
- Chaque run conserve les temps des événements CUDA par GPU, les temps H2D, préparation hôte et makespan. Pendant la campagne, enregistrer `nvidia-smi dmon -s u -d 1` en entier dans un journal horodaté. Le taux brut de relevés d'occupation `sm >= 70 %` par GPU et le taux de relevés à 0 % seront rapportés avec le nombre total d'échantillons. Le rapport `gpuN_event_ms / makespan_ms` résume la part de la fenêtre mesurée occupée par des GEMM sériels sur chaque GPU, avec les limites dues aux copies et à la granularité des événements.
- Les résultats de scheduling sont descriptifs : quatre seeds ne suffisent pas à établir une supériorité statistique. L'objectif primaire est de montrer et quantifier si cette variante maintient une occupation élevée sur les deux GPU pendant les fenêtres de calcul. Rapporter les deux cartes séparément ; aucun seuil de succès ne sera déduit d'une seule pointe.
- Les contrôles de somme doivent réussir ; sinon la cellule fautive est signalée.

Ce contrôle modifie intentionnellement le régime de calcul et le mode d'entrées. Il ne mesure donc pas le même effet causal que la charge à durées cachées. Un profil Nsight ciblé, exécuté à part, documentera la répartition du temps entre noyaux, copie et attente.
