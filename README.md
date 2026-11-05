# Évaluation locale de modèles sur GTX 1060

Machine testée : Intel Core i7-8750H (6 cœurs), 8 Gio de RAM, NVIDIA GTX 1060
(3 Gio de VRAM), WSL. Les artefacts lourds sont stockés sur `D:` :
`/mnt/d/dory`; `models/` est un lien vers ce répertoire.

## Installation validée

- Dory : fork `brianchmielNV/llama.cpp` au commit
  `98a74c97ea09ac72f6ceda1559808fc0a2f447ce`.
- GPU : CUDA 12.8 local, compilé pour Pascal SM 6.1. CUDA 13 ne supporte plus
  cette architecture.
- Q5_K_M : export NVIDIA, SHA-256 vérifié.
- Q3_K_M : requantifié localement depuis Q5 ; NVIDIA ne publie pas de Q3.

Le binaire CUDA est `llama.cpp-dory/build-cuda/bin/llama-server`. Avant de
l'utiliser, définir :

```bash
CUDA_HOME=/mnt/d/dory/tools/cuda-12.8-root/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$PWD/llama.cpp-dory/build-cuda/bin:$CUDA_HOME/lib64:/usr/lib/wsl/lib"
```

## Mesures de débit

`llama-bench`, 6 threads, contexte de 128 tokens et génération de 64 tokens :

| Modèle | Backend | Prompt | Génération |
|---|---|---:|---:|
| Dory Q5 | CPU | 14,76 tok/s | 3,34 tok/s |
| Dory Q3 | CPU | 15,04 tok/s | 4,45 tok/s |
| Dory Q3, 193/193 couches | CUDA | 166,01 tok/s | 4,13 tok/s |
| Dory Q5, offload ajusté | CUDA | 113,94 tok/s | 2,79 tok/s |

Le GPU accélère fortement le traitement du prompt. Sur cette GTX 1060, la
génération hybride du Q5 reste plus lente que sur CPU seul.

## IFEval

Les résultats utilisent un échantillon déterministe de 16 des 541 prompts
IFEval, 512 tokens maximum, contexte 4 096 et `temperature=0`. Ce ne sont pas
des scores IFEval officiels complets.

| Modèle | Boucles | Strict | Relâché | Pic RAM | Pic VRAM |
|---|---:|---:|---:|---:|---:|
| Dory Q3 | 1 | 37,5 % | 43,75 % | 3,01 Gio | 2 479 Mio |
| Dory Q3 | 2 | 43,75 % | 50,0 % | 3,62 Gio | 2 805 Mio |
| Dory Q3 | 4 | 37,5 % | 37,5 % | 4,70 Gio | 2 923 Mio |
| Dory Q5 | 2 | 56,25 % | 56,25 % | 5,69 Gio | 2 275 Mio |
| Dory Q5 | 4 | 56,25 % | 56,25 % | 6,18 Gio | 2 275 Mio |

**Recommandation actuelle : Dory Q5 à 2 boucles.** Il égale le Q5 à 4 boucles
sur cet échantillon, avec moins de RAM et de calcul.

## Reproduire les évaluations

`run_ifeval.py` démarre un serveur Dory, évalue les prompts et conserve les
scores stricts/relâchés ainsi que les pics mémoire dans `results/` :

```bash
python3 run_ifeval.py \
  --server llama.cpp-dory/build-cuda/bin/llama-server \
  --model models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --gpu-layers auto --fit-target 256 \
  --loops 2 --limit 16 --max-tokens 512
```

`run_ollama_ifeval.py` utilise la même sélection de prompts via l'API locale
d'Ollama. Il est employé pour comparer `qwen3.5:9b` et `bonsai-2:4b` à Dory.

```bash
python3 run_ollama_ifeval.py qwen3.5:9b
```

Les modèles, build products, jeux de réponses et résultats générés sont exclus
de Git. Le rapport brut CPU est dans [`benchmark.md`](benchmark.md).
