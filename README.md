# ARGOS

**Agentic Red-teaming & Guardrail-stripped Offensive System**

[![CI](https://github.com/shm0m/argos/actions/workflows/ci.yml/badge.svg)](https://github.com/shm0m/argos/actions/workflows/ci.yml)

ARGOS étudie comment le comportement de refus d'un LLM open-weight est encodé dans son espace d'activation, et dans quelle mesure on peut le neutraliser (*abliteration*) **sans dégrader sa capacité de raisonnement**.

La technique d'abliteration (Arditi et al., 2024 ; Labonne, 2024) montre qu'une direction unique du residual stream porte l'essentiel du comportement de refus, et que l'orthogonaliser suffit à le supprimer. Les implémentations existantes s'arrêtent à la démonstration : elles suppriment le refus, constatent une perte de performance, et la compensent après coup par un fine-tuning DPO correctif.

ARGOS pose une question différente : **peut-on caractériser et minimiser ce compromis refus/raisonnement de façon mesurable, couche par couche et direction par direction** — plutôt que de le réparer après coup ? Le projet inclut aussi un volet défensif : détecter, à partir des activations, qu'un modèle a subi une ablation.

Modèle cible : **Ministral-3-3B-Instruct** (Mistral AI, checkpoint BF16).

📄 [Note de méthode illustrée](https://claude.ai/code/artifact/5b39802e-f7a7-4159-ae47-9a17c2e3e679) · 📝 [Write-up technique complet (obstacles réels + résultats)](WRITEUP.md)

## Résultats clés

Sur Ministral-3-3B-Instruct, direction de refus retenue automatiquement par le pipeline (Phase 1) :

| | Modèle original | Modèle ablaté |
|---|---|---|
| Taux de refus (n=32, IC95 %) | 21,9 % [11,0 ; 38,8] | **0,0 %** [0,0 ; 10,7] |
| GSM8K exact-match (n=20) | 0,60 / 0,70 | 0,65 / 0,70 |
| HellaSwag / MMLU (n=100-150) | — | écarts ≤ 0,04, dans le bruit |

Balayage sur 8 couches réparties sur toute la profondeur du réseau : le refus dessine une courbe en cloche inversée (19-31 % de refus subsiste aux couches précoces, 0 % aux couches médianes 15-18, remonte à 12,5 % en fin de réseau), tandis que la capacité (HellaSwag) reste constante à toutes les profondeurs (0,48-0,51). Détail complet et discussion des limites dans le [write-up](WRITEUP.md#phase-2--mesure-capacité-vs-refus-résultat-valide).

## Statut

Phases 0 à 3 terminées et validées sur GPU réel (RTX 5070 Ti, 12 Go). Voir [Roadmap](#roadmap).

## Méthode (aperçu)

1. **Collecte d'activations** — faire passer un jeu d'instructions nuisibles et bénignes dans le modèle, capturer le residual stream (entrée/sortie de chaque couche decoder) via des hooks PyTorch natifs.
2. **Direction de refus** — différence de moyennes (nuisible − bénin) par couche, normalisée.
3. **Ablation** — orthogonalisation des poids d'écriture (`embed_tokens`, `self_attn.o_proj`, `mlp.down_proj`) par rapport à la direction sélectionnée.
4. **Mesure** — taux de refus résiduel (ASR) **et** score de capacité (HellaSwag / MMLU / GSM8K) pour chaque couche/direction candidate, afin d'identifier le point qui minimise la perte de raisonnement à neutralisation de refus égale.
5. **Détection** (volet défensif) — probing sur les activations pour distinguer un modèle ablaté d'un modèle non modifié, à direction de refus connue.

Contrairement à l'implémentation originale de Labonne (basée sur TransformerLens, qui ne connaît qu'un jeu figé d'architectures), ARGOS opère directement sur les modules HF (`nn.Linear`) via des hooks PyTorch génériques — nécessaire ici puisque l'architecture `ministral3` de Mistral-3 est trop récente pour être supportée par TransformerLens, et plus généralement réutilisable sur n'importe quel modèle causal HF de la famille Llama/Mistral.

## Roadmap

- [x] Phase 0 — Cadrage, structure du repo
- [x] Phase 1 — Direction de refus, orthogonalisation, validée de bout en bout sur GPU
- [x] Phase 2 — Mesure refus vs capacité (HellaSwag/MMLU/GSM8K) + balayage multi-couches
- [x] Phase 3 — Détection d'un modèle ablaté (volet défensif, MVP à direction connue)
- [x] Phase 4 — Packaging : CLI complète, démo interactive, write-up
- [ ] Phase 5 — CI, détection à direction inconnue

## Éthique et cadre d'usage

Ce projet est une contribution de recherche en interprétabilité et sécurité des LLM, pas un outil prêt à l'emploi pour contourner des garde-fous en production. Les jeux d'instructions « nuisibles » utilisés proviennent d'un benchmark public existant ([mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors), dérivé d'AdvBench) et servent uniquement à caractériser la direction de refus — aucun contenu nuisible généré n'est publié dans ce dépôt. Les résultats visent à documenter la fragilité du safety fine-tuning, dans la continuité des travaux d'Arditi et al. et de Labonne.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

Nécessite un GPU CUDA (validé sur 12 Go de VRAM). Sous Windows, installer torch avec la roue CUDA correspondant au driver plutôt que la roue CPU par défaut de PyPI, ex. :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

## Utilisation

```bash
# Pipeline complet : trouve la meilleure direction et sauvegarde le modele ablate
python -m argos.cli --config configs/ministral-3b.yaml --output results/ablated-model

# Mesure refus vs capacite, modele original vs ablate
python -m argos.measure --baseline <model_id> --ablated results/ablated-model

# Balaye plusieurs couches candidates pour tracer la courbe refus/capacite
python -m argos.sweep --n-directions 8

# Verifie si un modele a ete ablate avec une direction connue
python -m argos.detect --direction results/ablated-model/refusal_direction.pt \
    --baseline <model_id> --candidate results/ablated-model

# Interface de chat locale pour tester le modele ablate
python -m argos.server --model results/ablated-model
```

## Références

- Arditi, A. et al. (2024), [*Refusal in LLMs Is Mediated by a Single Direction*](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction).
- Labonne, M. (2024), [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration).

## Licence

MIT — voir [LICENSE](LICENSE).
