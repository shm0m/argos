# ARGOS

**Agentic Red-teaming & Guardrail-stripped Offensive System**

ARGOS étudie comment le comportement de refus d'un LLM open-weight est encodé dans son espace d'activation, et dans quelle mesure on peut le neutraliser (*abliteration*) **sans dégrader sa capacité de raisonnement**.

La technique d'abliteration (Arditi et al., 2024 ; Labonne, 2024) montre qu'une direction unique du residual stream porte l'essentiel du comportement de refus, et que l'orthogonaliser suffit à le supprimer. Les implémentations existantes s'arrêtent à la démonstration : elles suppriment le refus, constatent une perte de performance, et la compensent après coup par un fine-tuning DPO correctif.

ARGOS pose une question différente : **peut-on caractériser et minimiser ce compromis refus/raisonnement de façon mesurable, couche par couche et direction par direction** — plutôt que de le réparer après coup ? Le projet inclut aussi un volet défensif : détecter, à partir des activations, qu'un modèle a subi une ablation.

Modèle cible : **Ministral-3-3B-Instruct** (Mistral AI, checkpoint BF16).

## Statut

🚧 Projet en cours de construction. Voir [Roadmap](#roadmap) ci-dessous.

## Méthode (aperçu)

1. **Collecte d'activations** — faire passer un jeu d'instructions nuisibles et bénignes dans le modèle, capturer le residual stream (entrée/sortie de chaque couche decoder) via des hooks PyTorch natifs.
2. **Direction de refus** — différence de moyennes (nuisible − bénin) par couche, normalisée.
3. **Ablation** — orthogonalisation des poids d'écriture (`embed_tokens`, `self_attn.o_proj`, `mlp.down_proj`) par rapport à la direction sélectionnée.
4. **Mesure** — taux de refus résiduel (ASR) **et** score de capacité (MMLU / GSM8K / HellaSwag) pour chaque couche/direction candidate, afin d'identifier le point qui minimise la perte de raisonnement à neutralisation de refus égale.
5. **Détection** (volet défensif) — probing sur les activations pour distinguer un modèle abliterated d'un modèle non modifié.

Contrairement à l'implémentation originale de Labonne (basée sur TransformerLens, qui ne connaît qu'un jeu figé d'architectures), ARGOS opère directement sur les modules HF (`nn.Linear`) via des hooks PyTorch génériques — nécessaire ici puisque l'architecture `ministral3` de Mistral-3 est trop récente pour être supportée par TransformerLens, et plus généralement réutilisable sur n'importe quel modèle causal HF de la famille Llama/Mistral.

## Roadmap

- [x] Phase 0 — Cadrage, structure du repo
- [x] Phase 1 — Reproduction de la baseline (direction de refus, orthogonalisation) sur Ministral-3-3B, validée de bout en bout sur GPU (RTX 5070 Ti, 12 Go)
- [ ] Phase 2 — Mesure systématique refus vs raisonnement (multi-couches, multi-directions)
- [ ] Phase 3 — Détection d'un modèle abliterated (volet défensif)
- [ ] Phase 4 — Packaging (CLI, notebook de démo, résultats chiffrés)
- [ ] Phase 5 — Tests, CI

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

Lancer le pipeline complet :

```bash
python -m argos.cli --config configs/ministral-3b.yaml --output results/ablated-model
```

## Références

- Arditi et al., [*Refusal in LLMs Is Mediated by a Single Direction*](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), 2024.
- Labonne, [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration), 2024.

## Licence

MIT — voir [LICENSE](LICENSE).
