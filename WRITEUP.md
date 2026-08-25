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

### 8. Direction NaN sélectionnée silencieusement (bug critique, invalide le premier résultat Phase 1)

En préparant une mesure Phase 2 plus légère, un test de cohérence simple (demander au modèle ablaté la capitale de la France) a révélé que celui-ci générait uniquement des tokens `<unk>` en boucle. Inspection des poids sauvegardés : `embed_tokens`, tous les `self_attn.o_proj` et tous les `mlp.down_proj` étaient **entièrement composés de NaN**.

Cause identifiée par diagnostic direct sur les activations collectées : la couche 0 (`resid_pre`, c'est-à-dire l'embedding brut du dernier token, avant tout bloc transformer) a une différence de moyenne nuisible/bénin **exactement nulle**, parce que le dernier token de la séquence est le même marqueur de début de tour assistant pour tous les prompts (ajouté par `add_generation_prompt=True`). Normaliser un vecteur de norme nulle produit un NaN. Le tri des candidats par `abs(direction.mean())`, non protégé contre les NaN, a classé ce candidat en position #0. Pire : la métrique de refus (recherche de phrases comme « I cannot ») ne détectait pas le texte vide comme un cas particulier, elle l'a donc compté à tort comme « pas de refus », validant en apparence une direction totalement cassée.

**Conséquence : le résultat « 0,00 % de refus résiduel » annoncé plus bas pour la Phase 1 était un faux positif.** Le modèle n'avait pas cessé de refuser, il avait cessé de générer quoi que ce soit de cohérent.

Corrections apportées :
- `compute_refusal_directions` exclut désormais les couches à norme quasi nulle avant normalisation, et filtre tout NaN résiduel avant le tri.
- Ajout de `is_degenerate`/`degenerate_rate` (texte vide ou trop court) dans le module de mesure de refus, pour ne plus jamais confondre « absence de réponse » et « absence de refus ».
- La sélection de la meilleure direction dans `argos.cli` écarte désormais tout candidat majoritairement dégénéré, avec erreur explicite si aucune direction valide ne subsiste.
- Tests de non-régression ajoutés (`tests/test_direction.py`, `tests/test_eval.py`).

## Résultats obtenus

### Phase 1 : ablation complète (rejouée avec le correctif, validée)

Premier run (avant correctif) : configuration par défaut, 20 directions candidates, ~1h05 de calcul. Avait annoncé un taux de refus résiduel de 0,00 % dès la première direction testée (candidat #0). Ce résultat était un artefact du bug décrit ci-dessus (obstacle 8) : la direction retenue était NaN, le modèle ne générait plus rien d'exploitable. **Invalidé, retiré.**

Second run (après correctif), même configuration (256 instructions d'entraînement, 32 de test, 20 directions candidates), durée comparable (~40 minutes pour le scoring des directions) :

- Direction retenue : **candidat #4** (et non plus #0 : le classement change une fois la couche 0 dégénérée exclue).
- Taux de refus résiduel sur les 32 instructions de test : **0,00 %**.
- Taux de générations dégénérées (texte vide/trop court) sur ces mêmes 32 instructions : **0,00 %**, ce qui confirme qu'il ne s'agit pas d'un nouveau cas de corruption.
- Vérification directe des poids sauvegardés : **0 tenseur sur 458 contient un NaN ou un Inf** (contre 53/458 entièrement NaN sur le run précédent).
- Test de cohérence qualitatif (« Quelle est la capitale de la France ? ») : réponse correcte et cohérente (*« The capital of France is Paris. »*), confirmant que le modèle raisonne toujours normalement sur une question neutre.

### Phase 2 : mesure capacité vs refus (résultat valide)

Les premiers chiffres obtenus lors des smoke tests (contraste HellaSwag entre modèle original et modèle « ablaté ») reflétaient la même corruption que la Phase 1 : un modèle qui ne génère que des tokens `<unk>` obtient logiquement 0 sur toute tâche de capacité. Ces chiffres ont été retirés, ils ne mesuraient rien de réel.

Une fois le modèle ablaté corrigé (Phase 1 rejouée) disponible, la mesure a été relancée en deux temps : d'abord une version allégée (HellaSwag et quatre sujets MMLU, n=100, GSM8K exclu par défaut pour éviter la surchauffe), puis une version étendue avec échantillon plus large (n=150 sur HellaSwag/MMLU) et GSM8K inclus en petit échantillon (n=20), en surveillant activement la température GPU (restée sous 56°C tout du long, contre l'extinction thermique du run précédent). Refus mesuré sur les 32 instructions de test habituelles, avec intervalle de confiance de Wilson à 95 %.

| | Original | Ablaté | Écart |
|---|---|---|---|
| Refus (n=32) | 21,9 % [11,0 ; 38,8] | 0,0 % [0,0 ; 10,7] | net, IC non chevauchants |
| HellaSwag acc (n=150) | 0,487 ± 0,041 | 0,480 ± 0,041 | −0,01, bruit |
| HellaSwag acc_norm (n=150) | 0,680 ± 0,038 | 0,673 ± 0,038 | −0,01, bruit |
| MMLU algèbre abstraite (n=100) | 0,46 ± 0,050 | 0,43 ± 0,050 | −0,03, bruit |
| MMLU maths lycée (n=150) | 0,46 ± 0,041 | 0,467 ± 0,041 | +0,01, bruit |
| MMLU scénarios moraux (n=150) | 0,28 ± 0,037 | 0,273 ± 0,037 | −0,01, bruit |
| MMLU droit professionnel (n=150) | 0,507 ± 0,041 | 0,467 ± 0,041 | −0,04, ≈ 1 erreur-type |
| GSM8K exact-match strict (n=20) | 0,60 ± 0,112 | 0,65 ± 0,109 | +0,05, bruit |
| GSM8K exact-match flexible (n=20) | 0,70 ± 0,105 | 0,70 ± 0,105 | 0,00 |

Lecture : le refus chute nettement et de façon statistiquement significative (intervalles à 95 % qui ne se recouvrent quasiment pas). Sur les sept mesures de capacité, un seul écart (droit professionnel) approche une erreur-type ; tous les autres, y compris GSM8K, la tâche de raisonnement multi-étapes la plus susceptible de révéler une dégradation, sont dans le bruit. C'est un résultat notable au regard de la littérature (Labonne rapporte une perte de performance nécessitant un DPO correctif après ablation) : sur ce modèle et cette direction précise, le compromis refus/capacité semble, à cette échelle de mesure, quasi nul, GSM8K inclus.

**Limites à ne pas perdre de vue** : les erreurs-types restent notables (4 points sur les tâches à n=150, 11 points sur GSM8K à n=20), donc une dégradation réelle mais modeste pourrait rester invisible ; aucune correction pour comparaisons multiples (7 mesures testées) ; une seule direction d'ablation a été évaluée, pas de balayage systématique couche par couche.

## État actuel et limites assumées

- Phase 0 (cadrage) : terminée.
- Phase 1 (ablation) : **terminée et validée**, pipeline corrigé, couvert par des tests de non-régression, résultat rejoué et vérifié (poids sains, sortie cohérente).
- Phase 2 (mesure capacité/refus) : **terminée pour la direction retenue** : HellaSwag, 4 sujets MMLU et GSM8K, refus quasi éliminé, aucune perte de capacité détectable au-delà du bruit sur les 7 mesures. Reste : balayage multi-directions.
- Phase 3 (détection d'un modèle ablaté, volet défensif) et Phase 4 (packaging final) : non commencées.

## Prochaines étapes

1. Balayer plusieurs directions/couches candidates (pas seulement celle retenue par Phase 1) pour tracer la vraie figure signature du projet : refus résiduel vs capacité, en fonction du choix de direction.
2. Augmenter l'échantillon (n plus grand par tâche) pour resserrer les erreurs-types et confirmer que l'absence de dégradation tient à plus grande échelle.
3. Volet défensif (Phase 3) : probing sur les activations pour détecter qu'un modèle donné a été ablaté.

## Leçon retenue

Le bug le plus coûteux de ce projet n'était pas dans la partie « intelligente » (calcul de la direction de refus), mais dans l'absence de garde-fou sur une métrique d'évaluation trop naïve : une chaîne vide ne contient aucune des phrases de refus recherchées, donc elle passait pour un succès. Rétrospectivement, la rigueur méthodologique (vérifier qu'un modèle produit toujours une sortie cohérente avant d'interpréter un score) aurait dû être mise en place dès la Phase 1, pas seulement pour la Phase 2.
