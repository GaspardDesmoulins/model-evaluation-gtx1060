# Évaluation locale de modèles sur GTX 1060

Machine testée : Intel Core i7-8750H (6 cœurs), NVIDIA GTX 1060 (3 Gio de
VRAM), WSL avec 8 Gio de RAM pour les premiers tests, puis 11 Gio pour les
derniers essais FP32. Les artefacts lourds sont stockés sur `D:` :
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
| Bonsai-2 4B Q2_0 | — | 43,75 % | 50,0 % | — | 1 693 Mio |
| Qwen3.5 9B | — | 62,5 % | 68,75 % | — | 1 459 Mio |
| CoDa 1.7B NF4 | 64 étapes | 0 % | 0 % | 5,43 Gio | 2 447 Mio |
| CoDa 1.7B NF4 | 256 étapes | 0 % | 0 % | 5,43 Gio | 2 447 Mio |

**Meilleure qualité mesurée : Qwen3.5 9B.** Il dépasse Dory Q5 à 2 boucles de
6,25 points sur le score strict, mais son premier chargement a pris environ
76 secondes. Dory Q5 à 2 boucles reste l'option plus légère et prévisible :
il égale le Q5 à 4 boucles avec moins de RAM et de calcul. Bonsai consomme
moins de VRAM, mais reste en retrait de 12,5 points sur le score strict.

## CoDa 1.7B (diffusion, code)

CoDa est publié uniquement en FP32 (8,12 Go). Les premiers essais sur 8 Gio
de RAM utilisent `run_coda_benchmark.py` pour le charger en NF4 4 bits à
l'exécution, avec l'attention SDPA math compatible Pascal.

| Mesure | Résultat |
|---|---:|
| Chargement depuis FP32 | 586 s (64 étapes), 524 s (256 étapes) |
| Génération moyenne | 20,0 s (64 étapes), 77,7 s (256 étapes) |
| Pic RAM du processus | 5,43 Gio |
| Pic VRAM | 2 079 Mio |
| Prompts de code | 4 |

À 64 étapes de diffusion et 128 tokens, les sorties sont mitigées : `add` est
correct, `is_palindrome` est presque correct mais omet l'import `re`, et les
deux autres sorties sont invalides ou tronquées. Ce microbenchmark mesure la
faisabilité et la latence ; il ne remplace pas HumanEval, dont le protocole
officiel (768 étapes et BF16) n'a pas été reproduit ici. À 256
étapes, seul `add` est correct : l'augmentation du nombre d'étapes ne résout
pas les sorties vides ou invalides.

Sur le même échantillon IFEval de 16 prompts que les autres modèles, à 64
étapes et 512 tokens, CoDa atteint 0 % strict et relâché. Huit réponses sont
vides et aucune ne satisfait toutes les contraintes. CoDa ne constitue donc
pas une alternative généraliste pratique à Dory, Bonsai ou Qwen sur ce matériel.
À 256 étapes, le score reste nul, avec une génération moyenne de 237 s par
prompt et une durée totale de 1 h 10 min 56 s.

L'export NF4 persistant validé est stocké dans
`/mnt/d/dory/models/CoDA-v0-Instruct-NF4`. Il occupe 2,05 Go et se recharge en
14,4 s, contre 8 min 13 s depuis les poids FP32.

Le dispatch hybride d'Accelerate tronque les logits pendant la génération
diffusion et ne peut pas être utilisé. Avec les 11 Gio de RAM WSL, le modèle
FP32 se charge directement en RAM puis le script déplace manuellement
l'embedding et les 8 premiers blocs sur GPU. Les activations ne traversent la
frontière GPU/CPU qu'une fois ; le générateur diffusion utilise les indices sur
le périphérique des logits, ce qui évite l'erreur de masque.

Ce placement 8 GPU / 20 RAM effectue 64 étapes sans OOM, avec 2 749 Mio de
VRAM et 8 980 Mio de RAM au pic. La génération de 128 tokens prend 146,5 s,
mais cette exécution de validation a produit une réponse vide ; elle valide le
placement GPU, pas la qualité de CoDa. Le script de validation est
`test_coda_fp32_offload.py --manual-gpu-layers 8 --diffusion-test --steps 64
--max-new-tokens 128`.

À 768 étapes, un test FP32 hybride sur le seul prompt `add` a pris 29 min 02 s
de génération, avec 2 749 Mio de VRAM et 8 985 Mio de RAM au pic. La fonction
produite est correcte, mais suivie d'une balise Markdown fermante isolée. Ce
test unique ne constitue pas un benchmark HumanEval.

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
d'Ollama. Il désactive le raisonnement pour obtenir les réponses finales.

```bash
python3 run_ollama_ifeval.py qwen3.5:9b
```

Le runtime CoDa isolé est stocké sur `D:`. Après son installation, lancer :

```bash
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 python3 run_coda_benchmark.py
```

Le benchmark code commun à Dory et Ollama est lancé ainsi :

```bash
python3 run_code_benchmark.py ollama --model qwen3.5:9b --name qwen3.5-9b
```

Pour le même échantillon IFEval que les autres modèles :

```bash
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 python3 run_coda_ifeval.py
```

Pour exporter CoDa en NF4 persistant :

```bash
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 python3 export_coda_nf4.py
```

## Servir les modèles sans Ollama

Lancer les commandes depuis `/home/gaspard/dory`. Chaque serveur écoute
uniquement sur `127.0.0.1` ; attendre la fin du chargement avant les requêtes.
Arrêter le serveur avec `Ctrl+C` avant de lancer un autre modèle afin de libérer
la mémoire GPU.

### Dory Q3 et Q5 avec llama.cpp

Le fork et le build CUDA indiqués dans l'installation sont nécessaires :

```bash
cd /home/gaspard/dory
CUDA_HOME=/mnt/d/dory/tools/cuda-12.8-root/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$PWD/llama.cpp-dory/build-cuda/bin:$CUDA_HOME/lib64:/usr/lib/wsl/lib"
```

Q3, avec les 193 couches sur GPU et 2 boucles, comme dans le benchmark code :

```bash
llama.cpp-dory/build-cuda/bin/llama-server \
  -m models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q3_K_M.gguf \
  --host 127.0.0.1 --port 8081 -t 6 -c 4096 -ngl 193 \
  --jinja --reasoning off \
  --override-kv 'dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop'
```

Q5, avec placement GPU ajusté automatiquement et 2 boucles :

```bash
llama.cpp-dory/build-cuda/bin/llama-server \
  -m models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --host 127.0.0.1 --port 8081 -t 6 -c 4096 -ngl auto \
  --fit on --fit-target 256 --jinja --reasoning off \
  --override-kv 'dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop'
```

Pour les variantes à 1 ou 4 boucles, remplacer `int:2` par `int:1` ou `int:4`.
Pour un essai CPU seul, remplacer les options `-ngl ...` et `--fit ...` par
`-ngl 0 --fit off`, avec l'un ou l'autre fichier GGUF.

L'interface web intégrée de `llama-server` est activée par défaut : une fois
le serveur prêt, ouvrir <http://127.0.0.1:8081/> dans un navigateur local.
Depuis Windows, l'accès à ce port WSL dépend du transfert de localhost.

Dans un autre terminal :

```bash
curl --fail http://127.0.0.1:8081/health
curl --fail http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dory",
    "messages": [{"role": "user", "content": "Write only Python code for add(a, b)."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

### CoDa NF4 sur GPU

Le serveur charge l'export NF4 persistant et expose une API compatible avec
`/v1/chat/completions` :

```bash
cd /home/gaspard/dory
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 \
python3 serve_coda.py --steps 64 --max-tokens 128
```

Attendre le message `CoDa ready`, puis ouvrir <http://127.0.0.1:8000/> dans un
navigateur. La page affiche les échanges, mais chaque question est indépendante :
le serveur ne transmet pas l'historique au modèle. Arrêter le serveur avec
`Ctrl+C`.

Vérifier sa disponibilité :

```bash
curl http://127.0.0.1:8000/health
```

Envoyer une requête de code :

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write only Python code for add(a, b)."}],
    "max_tokens": 128,
    "diffusion_steps": 64
  }'
```

### CoDa FP32 GPU + RAM

Arrêter tout autre modèle utilisant le GPU avant de lancer ce serveur :

```bash
cd /home/gaspard/dory
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 \
python3 serve_coda.py --precision fp32 --gpu-layers 8 --steps 64 --max-tokens 128
```

Après `CoDa ready`, ouvrir <http://127.0.0.1:8000/>. Le serveur charge les
poids FP32 de `/mnt/d/dory/models/CoDA-v0-Instruct` directement en RAM, place
l'embedding et les 8 premiers blocs sur GPU et conserve les 20 autres blocs
sur CPU. `--gpu-layers` permet de réduire le nombre de blocs GPU ; `--model`
peut remplacer le chemin du checkpoint. `--precision nf4` est le mode par
défaut. Les autres options et l'API sont identiques au serveur NF4.

Le placement à 8 blocs a atteint environ 2 749 Mio de VRAM et 8 985 Mio de
RAM du processus avec 11 Gio de RAM WSL au total. Le système peut néanmoins
recourir au swap et lire les poids depuis le disque. Ne pas utiliser les
placements Accelerate pour la génération FP32 : ils ont échoué lors de nos
essais.

### Options et limites du serveur CoDa

| Option | Défaut | Valeurs et limites |
|---|---|---|
| `--precision` | `nf4` | `nf4` ou `fp32`. NF4 charge l'export quantifié sur GPU ; FP32 utilise le placement hybride GPU/RAM. |
| `--model` | Selon la précision | Chemin du checkpoint : `/mnt/d/dory/models/CoDA-v0-Instruct-NF4` en NF4, `/mnt/d/dory/models/CoDA-v0-Instruct` en FP32. Fournir un checkpoint correspondant à la précision ; cette option ne le convertit pas. |
| `--gpu-layers` | `8` | En FP32 uniquement, entier de 0 à 28 pour ce modèle. Même à 0, l'embedding reste sur GPU. Huit blocs ont été testés sur la GTX 1060 ; davantage peut dépasser la VRAM disponible. Ignoré en NF4. |
| `--steps` | `64` | Entier positif, nombre d'étapes de diffusion. Aucun maximum n'est imposé par le serveur ; le temps augmente fortement avec le nombre d'étapes. |
| `--max-tokens` | `128` | Entier positif, limite de tokens générés par requête. Aucun maximum n'est imposé par le serveur ; une valeur élevée consomme davantage de mémoire et de temps. |
| `--temperature` | `0` | Nombre fini positif ou nul en FP32 ; `0` uniquement en NF4, car une température non nulle a provoqué une erreur CUDA. |
| `--host` | `127.0.0.1` | Adresse d'écoute. Garder l'adresse locale : le serveur n'a pas d'authentification. |
| `--port` | `8000` | Port TCP local disponible, par exemple `8001`. Adapter l'URL dans le navigateur. |

La page web affiche le modèle et permet de modifier les étapes, les tokens
maximum et la température pour chaque message. Ces réglages ne modifient pas
les valeurs par défaut du serveur. L'API accepte les champs correspondants
`diffusion_steps`, `max_tokens` et `temperature` avec les mêmes limites ;
`GET /api/config` renvoie les valeurs par défaut et le modèle chargé.

Les limites ci-dessus sont celles du logiciel, pas une garantie que toute
combinaison tienne en mémoire. À 64 étapes / 128 tokens, l'essai FP32 hybride
a pris 146,5 s de génération ; à 768 étapes, environ 29 min pour un seul prompt.
Le serveur traite une génération à la fois, sans transmission de l'historique :
seul le dernier message utilisateur est utilisé. Il ne gère pas CORS pour une
page servie depuis une autre origine.

### Tests CoDa FP32 ponctuels

Pour reproduire les tests ponctuels plutôt que démarrer un serveur, utiliser
les commandes suivantes. Elles exécutent le prompt intégré au script, puis
s'arrêtent et écrasent le `summary.json` du placement correspondant.

FP32 entièrement CPU :

```bash
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 \
python3 test_coda_fp32_offload.py --cpu-only --diffusion-test \
  --steps 64 --max-new-tokens 128
```

FP32 hybride, 8 blocs GPU et 20 blocs CPU, sans offload de poids explicite sur
disque. Pour un essai à 64 étapes :

```bash
cd /home/gaspard/dory
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 \
python3 test_coda_fp32_offload.py --manual-gpu-layers 8 --diffusion-test \
  --steps 64 --max-new-tokens 128
```

`--manual-gpu-layers 8` place l'embedding et les 8 premiers blocs sur GPU,
les 20 blocs restants en RAM ; réduire ce nombre réduit l'occupation GPU.
`--diffusion-test` active la génération, `--steps` règle le nombre d'étapes
et `--max-new-tokens` la longueur maximale générée. Remplacer `--steps 64`
par `--steps 768` pour reproduire l'essai plus long. Sur le prompt intégré,
celui-ci a pris environ 29 minutes de génération.

Les modèles, build products, jeux de réponses et résultats générés sont exclus
de Git. Le rapport brut CPU est dans [`benchmark.md`](benchmark.md).
