# Protocole figé avant les mesures du 24 octobre 2026

## Instance et budget

Pod RunPod `zgi4mo43z7dkw9`, deux RTX PRO 4000 Blackwell à 1,15 $/h,
conteneur PyTorch 2.8.0 et CUDA 12.8, stockage éphémère uniquement.
Le budget de cette session reste limité à 2 $. Arrêter le pod après publication
de toutes les données, sans attendre que le crédit du compte soit épuisé.
Relever la topologie, le pilote, `cpu.max`, le commit exécuté et les contrôles
de correction. Ne pas fusionner les mesures de pods distincts.

## Bloc A : réplication du résultat conditionnel

- Charge `transfer_hidden`, entrées épinglées fraîches, sans processus CPU
  concurrent ajouté ; politiques `round_robin` et `feedback`.
- Seeds **1100–1123**, deux répétitions par seed, ordre des politiques
  randomisé comme dans `experiments/study.py` : 96 runs attendus.
- Exporter le JSON brut, le CSV par run et le CSV par tâche avant de fermer le
  pod. Chaque tâche doit satisfaire le contrôle de somme.
- Analyse fixée : médiane des deux répétitions par politique et seed ; gain
  `100 × (1 − feedback / round_robin)`. Classement 2–2 versus 3–1/4–0
  d'après les quatre tâches à 32 GEMM sous round-robin, identique entre
  répétitions. Différence des médianes de gain entre groupes, bootstrap
  stratifié de 10 000 tirages et seed 7. Le critère est atteint si chaque
  groupe compte au moins 6 seeds et si la borne inférieure de l'IC 95 % de
  cette différence est positive. Aucun retrait de seed après mesure.
- Ce bloc réplique le phénomène sur une nouvelle instance et de nouvelles
  seeds. Le classement reste a posteriori et ne valide pas une politique en
  ligne.

## Bloc B : interaction de contention avec ordre alterné

- Charge et politiques identiques, seeds **1200–1211**, deux répétitions :
  96 runs attendus.
- Pour les seeds d'indice pair : H0 puis H40. Pour les seeds d'indice impair :
  H40 puis H0. H40 ajoute 40 processus CPU de hachage après calibration et
  avant le chronométrage de chaque run.
- Analyse : gain `feedback` apparié à `round_robin` par seed dans chaque
  condition ; médiane de `gain_H40 − gain_H0`, IC bootstrap 95 % de 10 000
  tirages par seed. Rapporter aussi le temps de préparation, le temps des
  événements GPU, les intervalles de worker et les compteurs cgroup.
- L'alternance réduit le biais d'ordre de la campagne précédente ; elle ne
  supprime ni la variabilité cloud ni l'interaction avec l'allocation épinglée.

## Règles de décision et archivage

Lancer d'abord le bloc A. Exécuter B si le budget le permet. Les profils
Nsight ciblés sont secondaires et ne doivent pas retarder l'export. Les runs
profilés sont exclus des statistiques. Copier sur le poste local et publier
sur GitHub les JSON bruts, les CSV par run et par tâche, les métadonnées
machine, les scripts et les analyses avant l'arrêt du pod. Signaler toute
cellule incomplète ; ne pas la présenter comme une confirmation.
