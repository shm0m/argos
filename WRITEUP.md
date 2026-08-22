# ARGOS : localiser et mesurer le compromis refus/raisonnement dans un LLM open-weight

## Motivation

La technique d'abliteration (Arditi et al., 2024 ; popularisée par Labonne, 2024) montre qu'une direction unique de l'espace d'activation d'un LLM porte l'essentiel de son comportement de refus. Orthogonaliser les poids du modèle par rapport à cette direction suffit à neutraliser le refus, sans réentraînement.

Les implémentations existantes s'arrêtent à la démonstration : elles suppriment le refus, constatent une perte de performance générale, et la compensent après coup par un fine-tuning DPO correctif. ARGOS pose une question différente : peut-on caractériser ce compromis refus/raisonnement de façon mesurable, plutôt que de le réparer après coup une fois le dégât fait ? Le projet inclut aussi un volet défensif prévu pour la suite : détecter, à partir des activations, qu'un modèle a subi une ablation.

Modèle cible : Ministral-3-3B-Instruct (Mistral AI), choisi pour sa taille raisonnable sur une machine personnelle (RTX 5070 Ti Laptop, 12 Go de VRAM) et parce qu'il s'agit d'une architecture assez récente pour n'être supportée par aucun des outils d'interprétabilité standards, ce qui s'est avéré être le fil rouge technique de tout le projet.

## Méthode

1. **Collecte d'activations** : passage d'un jeu d'instructions nuisibles et bénignes dans le modèle, capture du residual stream (entrée et sortie de chaque couche decoder) à la dernière position de token.
2. **Direction de refus** : différence de moyennes (activations nuisibles moins activations bénignes) par couche, normalisée. Les directions candidates sont triées par force du signal.
3. **Sélection** : génération avec ablation par hook pour chaque direction candidate, sur un jeu d'instructions nuisibles de test. La direction retenue est celle qui minimise le taux de refus résiduel.
4. **Ablation** : orthogonalisation permanente des poids d'écriture (`embed_tokens`, `self_attn.o_proj`, `mlp.down_proj` de chaque couche) par rapport à la direction retenue.
5. **Mesure** : taux de refus résiduel et scores de capacité (HellaSwag, GSM8K, MMLU) comparés entre le modèle original et le modèle ablaté, pour quantifier le compromis plutôt que de le supposer.

## Ce qui a été réellement construit, et les obstacles rencontrés

Le déroulé « propre » ci-dessus masque le vrai travail d'ingénierie : à chaque étape, un choix technique hérité de l'article de référence s'est révélé incompatible avec le modèle cible ou avec l'environnement Windows/GPU réel. Documenter ces obstacles est en soi la partie la plus intéressante du projet.

### 1. TransformerLens ne supporte pas l'architecture de Ministral-3

L'implémentation originale de Labonne repose sur TransformerLens (`HookedTransformer`), qui ne connaît qu'une liste figée d'architectures. Ministral-3 utilise un type d'architecture (`ministral3`) trop récent pour y figurer. Plutôt que de changer de modèle cible, tout le pipeline de collecte d'activations, de calcul de direction et d'ablation a été réécrit pour opérer directement sur les modules PyTorch/Hugging Face natifs (`register_forward_hook`, `register_forward_pre_hook`, orthogonalisation directe des `nn.Linear`). Ce choix rend ARGOS réutilisable sur n'importe quel modèle causal Hugging Face récent, pas seulement ceux que TransformerLens a déjà cataloguée : un vrai gain, pas seulement un contournement.

### 2. Ministral-3 est multimodal

Le checkpoint est une classe `Mistral3ForConditionalGeneration` (backbone texte `ministral3` plus tour de vision Pixtral). Il a fallu localiser le sous-module texte réel (`model.language_model`) pour y accrocher les hooks, et charger le modèle via `AutoModelForImageTextToText` plutôt que `AutoModelForCausalLM`, qui ne reconnaît pas cette classe.

### 3. Dataset nuisible gated

Le premier choix de dataset (`walledai/AdvBench`) s'est révélé gated sur le Hub, bloquant l'exécution sans authentification. Remplacé par `mlabonne/harmful_behaviors`, la version publique reformattée par l'auteur de l'article de référence, avec ses propres splits train/test.

### 4. Checkpoint FP8 par défaut

Le checkpoint par défaut de Ministral-3-3B-Instruct est quantifié en FP8, ce qui exige un package de kernels spécifique (`kernels`, absent de l'environnement) pour l'inférence. Plutôt que d'installer une dépendance supplémentaire fragile, le pipeline utilise la variante officielle BF16 (`mistralai/Ministral-3-3B-Instruct-2512-BF16`), cohérente avec une ablation qui manipule directement des poids `nn.Linear` classiques.

### 5. Torch installé sans support CUDA

Malgré un GPU détecté par le système (driver CUDA 13.2), l'environnement Python utilisait un build torch CPU-only. Réinstallation explicite de la roue CUDA correspondante (`torch==2.13.0+cu130`) avant de pouvoir lancer quoi que ce soit sur GPU.

### 6. Tokenizer Mistral incompatible avec lm-eval-harness

Pour la mesure de capacité (Phase 2), lm-eval-harness (`HFLM`) vérifie par une assertion `isinstance` que le tokenizer est bien un `PreTrainedTokenizer` ou `PreTrainedTokenizerFast`. Or Ministral-3 utilise un nouveau backend, `MistralCommonBackend`, qui n'hérite d'aucune des deux classes attendues et fait échouer l'assertion. Contournement : passer le chemin du modèle en tant que chaîne à `HFLM`, qui charge alors son propre tokenizer via `AutoTokenizer.from_pretrained` en interne, sans jamais passer par l'assertion sur un objet déjà instancié.

### 7. Extinction thermique en pleine mesure

La tâche GSM8K de lm-eval-harness repose sur de la génération (`generate_until`), beaucoup plus coûteuse que les tâches à choix multiple évaluées par log-vraisemblance (HellaSwag, MMLU) : environ 20 secondes par échantillon. La charge soutenue sur la RTX 5070 Ti Laptop a fini par déclencher une extinction thermique de la machine en pleine mesure Phase 2, interrompant le run avant son terme.

## Résultats obtenus

### Phase 1 : ablation complète

Run complet avec la configuration par défaut (`configs/ministral-3b.yaml`) : 256 instructions d'entraînement, 32 instructions de test, 20 directions candidates évaluées.

- Durée : environ 1h05 (collecte d'activations rapide, scoring des 20 directions dominant le temps total, ~140 à 200 secondes par direction).
- Résultat : la direction #0 (la mieux classée par force de signal) a été retenue directement. Taux de refus résiduel sur les 32 instructions de test : **0,00 %**.
- Le modèle ablaté a été sauvegardé avec succès (checkpoint safetensors, ~7,7 Go).

### Phase 2 : mesure capacité vs refus (partielle)

Un smoke test (5 échantillons HellaSwag) a validé le fonctionnement mécanique du pipeline de mesure : chargement des deux modèles (original et ablaté), calcul du taux de refus sur les 32 instructions de test par défaut, évaluation de capacité via lm-eval-harness.

Chiffres obtenus sur ce petit échantillon (non représentatifs, à confirmer sur un run à plus grande échelle) :

| | Refus (n=32) | HellaSwag acc_norm (n=5) |
|---|---|---|
| Modèle original | 21,9 % | 0,80 |
| Modèle ablaté | 0,0 % | 0,00 |

Ce contraste va dans le sens attendu par la littérature (l'ablation dégrade la capacité générale du modèle), mais n=5 est bien trop petit pour en tirer une conclusion : un seul échantillon supplémentaire correctement répondu changerait le score de 20 points. Le run à pleine échelle (50 échantillons par tâche sur HellaSwag, GSM8K et MMLU, pour les deux modèles) a été lancé mais interrompu par l'extinction thermique de la machine avant d'obtenir des chiffres exploitables.

## État actuel et limites assumées

- Phase 0 (cadrage) et Phase 1 (ablation) : terminées et validées sur GPU réel.
- Phase 2 (mesure capacité/refus) : outillage fonctionnel et validé mécaniquement, mais **aucun résultat chiffré fiable à ce jour**. La contrainte thermique de la machine impose de revoir la stratégie d'exécution (limiter GSM8K, échelonner les runs, ou réduire le nombre d'échantillons par session).
- Phase 3 (détection d'un modèle ablaté, volet défensif) et Phase 4 (packaging final) : non commencées.

## Prochaines étapes

1. Relancer la mesure Phase 2 en sessions plus courtes (par exemple limiter GSM8K à 20 échantillons, ou le retirer temporairement au profit de HellaSwag/MMLU qui sont nettement moins coûteux) pour rester sous le seuil d'extinction thermique.
2. Une fois des chiffres fiables obtenus, produire la figure signature du projet : taux de refus résiduel vs score de capacité, pour visualiser le compromis réel plutôt que de le supposer.
3. Volet défensif (Phase 3) : probing sur les activations pour détecter qu'un modèle donné a été ablaté.
