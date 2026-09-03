# Carnet de bord ARGOS

Démarche, sources, ce que j'ai appris et ce qui reste à explorer sur ce projet.

**En bref.** ARGOS localise et neutralise la direction de refus d'un LLM (Ministral-3-3B), mesure ce que ça coûte en capacité de raisonnement, et construit des outils pour détecter cette opération, avec ou sans connaître la direction utilisée par l'attaquant. Ce document raconte la démarche, pas seulement le résultat final : les choix pris, les sources qui les ont informés, les erreurs corrigées en cours de route, et ce que j'en retiens.

## 1. Démarche

Le point de départ était un article de blog (Labonne, 2024) qui démontre comment supprimer le réflexe de refus d'un LLM en identifiant et en retirant une direction unique de son espace d'activation. L'article s'arrête à la démonstration : le refus disparaît, la performance baisse, un fine-tuning correctif répare la casse. Mon objectif n'était pas de reproduire cette démonstration, mais de la transformer en question de recherche mesurable : quel est le vrai coût en capacité de raisonnement, et peut-on le minimiser plutôt que de le réparer après coup ?

J'ai découpé le travail en phases, chacune avec un livrable vérifiable avant de passer à la suivante.

**Phase 0, cadrage.** Structure du dépôt, choix du modèle cible (Ministral-3-3B, pour sa taille raisonnable sur mon matériel), positionnement du projet par rapport à l'article source.

**Phase 1, reproduire la baseline sur un vrai modèle.** Implémentation du pipeline de bout en bout : collecte d'activations, calcul de la direction de refus, ablation. Chaque étape a buté sur un obstacle concret que l'article original ne mentionne pas : l'architecture du modèle trop récente pour les outils standards, un dataset verrouillé, un format de poids incompatible, un environnement mal configuré. Puis un bug numérique sérieux (une direction NaN sélectionnée silencieusement) a invalidé un premier résultat entier, détecté seulement parce que j'ai pris le réflexe de tester la sortie du modèle à la main plutôt que de me fier à un seul chiffre agrégé.

**Phase 2, mesurer le compromis plutôt que le supposer.** Comparaison du refus et de la capacité (HellaSwag, MMLU, GSM8K) entre le modèle original et le modèle ablaté, avec intervalles de confiance plutôt que des pourcentages bruts. Puis un balayage sur huit couches réparties dans le réseau, pour obtenir une vraie courbe plutôt qu'un seul point de mesure.

**Phase 3, retourner le problème côté défense.** Construire un détecteur suppose d'abord de connaître la direction utilisée, un scénario peu réaliste. La version utile ne suppose rien : elle compare le profil d'un modèle suspect à celui d'un modèle de référence. Ma première tentative a cherché le mauvais signal (un pic isolé) et n'a rien détecté ; relire mon propre code d'ablation a révélé pourquoi, et corrigé le détecteur.

**Phase 4, en faire une vitrine.** Documentation honnête des résultats et des échecs (write-up technique), interface de démonstration, intégration continue, README pensé pour un lecteur qui ne connaît pas encore le projet.

> **Le moment le plus utile du projet.**
> Le bug le plus coûteux n'était pas dans la partie « intelligente » du pipeline (le calcul de la direction), mais dans l'absence de garde-fou sur une métrique d'évaluation trop naïve : une chaîne de caractères vide ne contient aucune des phrases de refus recherchées, donc elle passait pour un succès. La correction n'a pas seulement réparé un chiffre, elle a changé ma façon d'évaluer tout le reste du projet : vérifier qu'un modèle produit encore quelque chose de cohérent avant d'interpréter un score.

## 2. Sources utilisées

### Fondement théorique

| Source | Pourquoi elle a compté |
|---|---|
| Arditi, A. et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, LessWrong | L'hypothèse centrale du projet : le refus est porté par une direction quasi unique de l'espace d'activation. |
| Labonne, M. (2024), *Uncensor any LLM with abliteration*, Hugging Face | Implémentation de référence dont ARGOS s'inspire et dont il s'écarte, en substituant la mesure du compromis à la simple démonstration. |

### Modèle et données

| Source | Pourquoi elle a compté |
|---|---|
| [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) (Hugging Face) | Modèle cible ; le choix du checkpoint BF16 plutôt que le FP8 par défaut a été dicté par une contrainte technique rencontrée en cours de route. |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors), [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (Hugging Face Datasets) | Instructions nuisibles et bénignes pour calculer la direction de refus, choisies pour leur accès public sans authentification. |

### Outils et bibliothèques

| Source | Pourquoi elle a compté |
|---|---|
| PyTorch (hooks natifs), Hugging Face Transformers | Remplace TransformerLens, qui ne supporte pas l'architecture du modèle cible ; le code source de Transformers (`modeling_ministral3.py`, `modeling_mistral3.py`) a été lu directement pour comprendre la structure interne du modèle. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI) | Mesure de capacité standardisée (HellaSwag, MMLU, GSM8K), pour comparer à des résultats publiés plutôt qu'inventer une métrique maison. |
| FastAPI, uvicorn | Interface de démonstration interactive. |

## 3. Ce que j'ai appris

**Méthode.** Un chiffre agrégé peut masquer une panne totale. La métrique de refus cherchait des phrases comme « je ne peux pas » ; un texte vide n'en contient aucune, donc un modèle complètement cassé passait pour un succès parfait. Depuis, je vérifie systématiquement qu'un modèle produit une sortie cohérente avant d'interpréter un score, un réflexe qui vaut pour n'importe quelle évaluation automatisée, pas seulement pour du red teaming de LLM.

**Interprétabilité.** Le refus se concentre dans les couches médianes à tardives, pas dans les premières. Le balayage multi-couches confirme empiriquement une intuition standard en interprétabilité mécaniste : les comportements abstraits de haut niveau sont mieux représentés en profondeur, pas près de l'embedding brut.

**Ingénierie.** Une dépendance qui « marche en local » n'est pas forcément déclarée. La CI a échoué sur un module manquant (`accelerate`) alors que tout fonctionnait sur ma machine, simplement parce qu'il y était installé par autre chose. Un environnement propre (un venv jetable, pas mon environnement de travail habituel) est le seul test fiable avant de pousser du code.

**Rigueur statistique.** Un pourcentage seul ne dit rien sans intervalle de confiance. Passer d'un simple taux de refus à un intervalle de Wilson change la lecture des résultats : deux mesures qui semblent différentes peuvent se recouvrir largement une fois l'incertitude prise en compte, et inversement.

**Mécanisme.** Une opération « mesurée à une couche » n'est pas forcément « appliquée à une couche ». L'ablation édite les poids de tout le réseau avec une seule direction ; mon premier détecteur cherchait une anomalie localisée à la couche de mesure, un mauvais modèle mental corrigé seulement en relisant mon propre code d'ablation ligne par ligne.

**Contraintes réelles.** Le matériel personnel impose des choix de méthode, pas seulement de confort. Une extinction thermique en pleine mesure a forcé à repenser la stratégie d'exécution (échantillons plus légers, tâches les plus coûteuses isolées et surveillées), une contrainte qui a in fine amélioré la conception des outils de mesure plutôt que de simplement la retarder.

## 4. Perspectives d'évolution

- **Calibrer la détection à direction inconnue.** Validée sur un seul cas pour l'instant ; il faudrait la tester sur plusieurs couches d'ablation et sur un modèle réellement propre pour mesurer un taux de faux positifs, pas seulement un cas de succès.
- **Élargir la mesure de capacité.** Échantillons plus grands pour resserrer les intervalles de confiance, et davantage de tâches de raisonnement multi-étapes dans l'esprit de GSM8K.
- **Tester la généralité du résultat.** L'absence de compromis refus/capacité observée ici est spécifique à Ministral-3-3B et à cette direction précise ; reproduire le pipeline sur d'autres modèles open-weight dirait si c'est un résultat général ou un cas particulier.
- **Explorer des ablations partielles.** Le pipeline actuel ne teste que l'ablation complète (force maximale) par couche ; moduler la force de l'ablation ouvrirait un axe supplémentaire à la figure signature du projet.
- **Scénario de détection en production.** Passer d'une comparaison ponctuelle à deux modèles à un scan systématique d'un registre de modèles, pour un cas d'usage de sécurité plus réaliste.

---

Projet ARGOS · Shaïma Derouich · [github.com/shm0m/argos](https://github.com/shm0m/argos)
