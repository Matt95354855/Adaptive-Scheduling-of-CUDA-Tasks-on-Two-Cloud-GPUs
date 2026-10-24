# Décision en ligne par sonde de quatre tâches, 24 octobre 2026

## Mesure et intégrité

Le [protocole](online-probe-protocol-2026-10-24.md) et le seuil de 60 ms ont été publiés avant la campagne. Sur le nouveau pod à deux RTX PRO 4000 Blackwell, les seeds 1400–1423 ont été exécutées deux fois sous cinq politiques, dans un ordre randomisé : **240/240 runs et 2 880/2 880 tâches**. Aucun contrôle de somme n'a échoué. Les trois runs de vérification de la sonde sont archivés séparément et exclus de l'analyse.

Le contrôle principal `probe_feedback` et `probe_switch` exécutent les mêmes quatre premières tâches avant de décider de l'affectation des huit restantes. Ce comparateur inclut le coût de pause commun. Le gain est positif si `probe_switch` termine le lot plus vite.

| Référence de `probe_switch` | Gain médian sur 24 seeds | IC bootstrap 95 % | Seeds gagnées |
| --- | ---: | ---: | ---: |
| `probe_feedback` (principal) | **−0,12 %** | **[−0,59 ; +0,23] %** | 12/24 |
| `probe_rr` | +8,70 % | [+2,44 ; +15,01] % | 21/24 |
| `feedback` sans sonde | −14,62 % | [−17,10 ; −11,51] % | 0/24 |
| `round_robin` sans sonde | −1,01 % | [−2,86 ; +8,61] % | 10/24 |

La borne inférieure du contrôle principal n'est pas positive : **le critère de supériorité préenregistré échoue**. La politique adaptative par sonde n'apporte aucun avantage mesurable face au même préfixe suivi systématiquement de feedback. La comparaison au feedback sans sonde indique une forte pénalité de la pause ; le gain médian descriptif de `probe_feedback` face à `feedback` est lui-même **−13,33 %**. Cette différence inclut les effets de la pause et du changement de séquence d'envoi, et ne mesure pas le seul coût CPU de la décision.

## Pourquoi la décision ne gagne pas, analyse exploratoire

`probe_switch` a choisi feedback sur **18 seeds** et round-robin sur **6** ; la route ne change pas entre les deux répétitions d'une seed. Dans les contrôles ayant exactement le même préfixe, `probe_feedback` est plus rapide que `probe_rr` sur **les 24 seeds**. Les six choix de round-robin ne rejoignent donc jamais la meilleure des deux continuations observées pour leur seed. Cette lecture utilise les résultats après exécution : elle explique l'échec, mais n'est pas une règle validée en ligne.

La sonde est également un prédicteur imparfait de la répartition finale des quatre tâches longues. Sous round-robin complet, 14 seeds aboutissent à 3–1 ou 4–0 et 10 à 2–2. La sonde choisit feedback pour 12 des 14 déséquilibrées, mais aussi pour 6 des 10 équilibrées. Elle choisit round-robin pour 2 déséquilibrées et 4 équilibrées. Même une prédiction parfaite du déséquilibre ne garantirait pas que round-robin avec pause soit le meilleur choix : sur cette grille, le contrôle `probe_feedback` gagne toujours face à `probe_rr`.

La [figure par seed](../figures/06_online_probe.svg) montre la distribution du contraste principal et les décisions prises.

## Occupation et coût du protocole

Le relevé `nvidia-smi dmon` à une seconde pendant **643 secondes** de la campagne donne une occupation SM moyenne de **13,99 % sur GPU 0** et **14,29 % sur GPU 1** ; environ 69 % des relevés sont à zéro. Les maxima atteignent 100 % sur les deux cartes. Dans la seule fenêtre chronométrée d'un run, la somme des durées d'événements CUDA par GPU représente une médiane de **69,84 % et 70,74 %** du makespan. Ce dernier rapport n'est pas une mesure SM : il décrit du temps CUDA par rapport à la fenêtre du lot. L'écart avec la télémétrie de campagne vient notamment des démarrages de deux workers PyTorch et des calibrations répétés **à chaque run**, hors makespan. La charge à calcul soutenu mesure séparément ce point dans le [contrôle dédié](sustained-utilization-results-2026-10-24.md).

## Conclusion pratique

Le déséquilibre des tâches longues explique le gain du feedback dans les validations a posteriori, mais le signal simple proposé ici ne produit pas de politique en ligne meilleure que feedback. Le prochain changement de code utile consiste à conserver les workers et la calibration entre runs, puis à mesurer le coût de la décision sur une charge applicative ou un flux continu. Il faudra concevoir et préenregistrer une nouvelle règle avant de tester d'autres seeds ; sélectionner après coup un seuil ou des seeds gagnantes invaliderait la comparaison prospective.

## Données et code

- [JSON brut intégral](../results/online-probe-2026-10-24.json), [CSV des 240 runs](../results/online-probe-2026-10-24.csv), [CSV des 2 880 tâches](../results/online-probe-tasks-2026-10-24.csv)
- [Métriques par run](../results/online-probe-task-metrics-2026-10-24.csv), [comparaisons par seed](../results/online-probe-seeds-2026-10-24.csv), [résumé statistique](../results/online-probe-summary-2026-10-24.json)
- [Diagnostic des routes par seed](../results/online-probe-route-diagnostics-2026-10-24.csv), [résumé du diagnostic](../results/online-probe-route-diagnostics-2026-10-24.json)
- [Journal d'exécution](../results/online-probe-2026-10-24.log), [télémétrie GPU brute](../results/online-probe-gpu-util-2026-10-24.log), [résumé d'occupation](../results/online-probe-utilization-summary-2026-10-24.json)
- [Runs de vérification](../results/probe-smoke-2026-10-24.json), [tâches de vérification](../results/probe-smoke-tasks-2026-10-24.csv)
- [Code des politiques](../experiments/study.py), [analyse préenregistrée](../experiments/analyze_probe.py), [diagnostic exploratoire](../experiments/analyze_probe_routes.py)
