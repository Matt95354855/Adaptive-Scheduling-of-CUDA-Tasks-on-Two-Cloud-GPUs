# Deuxième réplication prospective sur un nouveau pod, 24 octobre 2026

## Méthode

Le [protocole](replication-protocol-2026-10-24.md) a été publié avant les
mesures. Le bloc A a utilisé le pod `zgi4mo43z7dkw9` à deux RTX PRO 4000
Blackwell, PyTorch 2.8.0+cu128, pilote 595.91.07, topologie `NODE`, quota
CPU de 20,4 cœurs. Le code exécuté est le commit
`cf96cce5397d536bc37aae4e9e2bebddb24433ff`. Les seeds 1100–1123,
deux répétitions et deux politiques donnent **96 runs et 1 152 tâches** ;
aucune cellule ne manque et les contrôles de somme sont satisfaits.

## Résultat principal

| Affectation des quatre tâches longues sous round-robin | Seeds | Victoires feedback | Gain médian feedback | IC bootstrap 95 % |
| --- | ---: | ---: | ---: | ---: |
| Équilibrée 2–2 | 7 | 0/7 | −7,88 % | [−11,41 ; −4,07] % |
| Déséquilibrée 3–1 ou 4–0 | 17 | 17/17 | +18,18 % | [+9,36 ; +19,16] % |

La différence préenregistrée des médianes de gain est **+26,05 points de
pourcentage**, IC bootstrap 95 % **[+17,24 ; +30,11]**. Chaque groupe compte
au moins six seeds et la borne inférieure de l'IC est positive : le critère
fixé avant mesure est satisfait sur cette nouvelle instance. Ce bloc est une
réplication indépendante en seeds et en instance du résultat conditionnel
observé la veille. La différence était alors +20,94 points
[+13,02 ; +27,66] ; les résultats ne sont pas fusionnés entre pods.

Sur les 24 seeds prises ensemble, le feedback gagne 17 fois et le gain médian
est +11,49 %. Ce chiffre global dépend de la composition : 17/24 seeds sont
déséquilibrées dans ce bloc, contre 14/24 dans la première validation. Il ne
représente donc pas une amélioration universelle.

La variable de groupe dépend des durées cachées, révélées après exécution.
Cette confirmation ne valide pas encore une décision de politique utilisable
en ligne. L'avantage des ordres déséquilibrés est robuste dans ces deux blocs
de la même charge synthétique, mais sa généralisation à d'autres charges ou
topologies reste ouverte.

## Lecture des chronologies, exploratoire

Dans les 34 runs dont round-robin déséquilibre les tâches longues, le feedback
aboutit à une répartition 2–2 dans **28 runs**. Dans les 14 runs où
round-robin est déjà équilibré, le feedback conserve 2–2 dans **10 runs** et
crée 3–1 dans **4 runs**. Cela fournit un mécanisme plausible pour le gain
conditionnel : la politique réactive corrige souvent une mauvaise affectation
statique, mais peut aussi perturber une affectation déjà bonne.

Les médianes descriptives des runs montrent des sommes d'événements GPU
quasi identiques, autour de **786–787 ms**, et une préparation des entrées
autour de **2,7–2,9 ms** par tâche dans les deux politiques et groupes.
La somme médiane des intervalles entre tâches sur les deux workers est
environ **1,5 ms** pour round-robin et **4,6–4,8 ms** pour feedback. Ces
indicateurs sont compatibles avec un coût de coordination et des effets de
réaffectation ; ils ne décomposent pas causalement la différence de makespan.

## Données intégrales publiées

- [JSON brut de tous les runs et de toutes les tâches](../results/replication-2026-10-24.json)
- [CSV des 96 runs](../results/replication-2026-10-24.csv)
- [CSV des 1 152 tâches](../results/replication-tasks-2026-10-24.csv)
- [Diagnostics par run](../results/replication-task-metrics-2026-10-24.csv)
- [Classement et gain par seed](../results/replication-seeds-2026-10-24.csv)
- [Verdict calculé](../results/replication-summary-2026-10-24.json)
- [Journal complet d'exécution](../results/replication-2026-10-24.log)
- [Manifeste matériel et logiciel](../results/diagnostic-2026-10-24.txt)
