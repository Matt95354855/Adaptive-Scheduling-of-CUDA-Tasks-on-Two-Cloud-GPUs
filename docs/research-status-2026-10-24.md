# État de la recherche après le pod du 24 octobre 2026

## Étapes achevées et traçables

| Étape | État et résultat | Données |
| --- | --- | --- |
| Réplication du gain conditionnel, nouvelle instance | 96 runs, 1 152 tâches. Contraste déséquilibré moins équilibré +26,05 points, IC [17,24 ; 30,11], critère prospectif atteint. | [Rapport](replication-results-2026-10-24.md) |
| Contention CPU à ordre alterné | 96 runs, 1 152 tâches. Interaction H40−H0 −13,08 points, IC [−21,81 ; +5,19] : signe non conclu. | [Rapport](alternating-contention-results-2026-10-24.md) |
| Décision en ligne par quatre tâches | 240 runs, 2 880 tâches. Gain de `probe_switch` vs même préfixe `probe_feedback` −0,12 %, IC [−0,59 ; +0,23] : critère non atteint. | [Rapport](online-probe-results-2026-10-24.md) |
| Charge GPU soutenue | 16 runs, 192 tâches. Temps CUDA dans les lots 96–97 % ; occupation SM moyenne de campagne 48–49 %, sous l'objectif de 70 %. | [Rapport](sustained-utilization-results-2026-10-24.md) |

Les quatre blocs ont été mesurés avec des contrôles de somme valides. Les fichiers bruts et les journaux de télémétrie sont publiés dans `results/`. Les figures et leurs scripts sont dans `figures/` et `experiments/`. Les blocs A, B et C restent distincts ; leurs intervalles ne sont pas fusionnés entre charges ou pods.

## À reprendre après la pause, par ordre logique

1. **Interaction CPU × durée de vie des entrées.** Exécuter le [plan factoriel préenregistré](factorial-input-protocol-2026-10-24.md) H0/H40 × entrées épinglées fraîches/réutilisées : 192 runs attendus. Il sépare mieux la contention CPU du coût d'allocation, sans supprimer la limite de données numériques différentes entre modes.
2. **Occupation end-to-end.** Conserver les deux workers CUDA et la calibration entre lots ; chronométrer du début de la requête à la fin, et échantillonner l'occupation SM tout au long du processus. La charge dense montre que la fenêtre du lot peut être saturée alors que le pod facturé ne l'est pas.
3. **Profil Nsight ciblé.** Capturer séparément un run `probe_feedback` et un run à charge dense, hors statistiques prospectives ; exporter les rapports CUDA/NVTX et sauvegarder la trace binaire avant l'arrêt d'un futur pod.
4. **Nouvelle politique en ligne.** Construire une règle qui exploite les coûts observés sans pause globale de quatre tâches, la préenregistrer, puis la tester sur des seeds inédites. Le choix `probe_switch` actuel n'a pas amélioré le feedback.
5. **Validité externe.** Répéter sur une autre charge applicative, puis une topologie GPU et un type d'instance différents. Les conclusions présentes concernent surtout une charge synthétique de tâches indépendantes.

L'utilisateur a demandé une pause après sauvegarde intégrale des résultats terminés. Le plan factoriel et la nouvelle trace Nsight n'ont pas été exécutés pendant cette session ; ils ne doivent pas être présentés comme des résultats.
