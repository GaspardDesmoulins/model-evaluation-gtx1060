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

## Évaluations

Les résultats, les conditions des essais et leurs commandes de reproduction
sont regroupés dans [`benchmark.md`](benchmark.md).

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

Le serveur journalise le chargement dans son terminal et affiche `model loaded`
une fois prêt. Pour suivre l'état et la mémoire GPU totale utilisée, dans un
autre terminal :

```bash
watch -n 2 'curl --silent --show-error --max-time 2 http://127.0.0.1:8081/health; echo; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```

Pendant le chargement, `/health` indique `Loading model` ; à la fin, il
renvoie `ok`. La mémoire affichée concerne tout le GPU, pas uniquement Dory :
elle ne donne pas de pourcentage ni de temps de chargement restant. Arrêter
`watch` avec `Ctrl+C`.

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
`/v1/chat/completions`. Pour créer l'export si nécessaire :

```bash
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 python3 export_coda_nf4.py
```

Lancer le serveur :

```bash
cd /home/gaspard/dory
PYTHONPATH=/mnt/d/dory/tools/coda-python-cu121 \
python3 serve_coda.py --steps 64 --max-tokens 128
```

Attendre le message `CoDa ready`, puis ouvrir <http://127.0.0.1:8000/> dans un
navigateur. La page affiche les échanges, mais chaque question est indépendante :
le serveur ne transmet pas l'historique au modèle. Arrêter le serveur avec
`Ctrl+C`.

Pendant la génération, la page affiche le texte à chaque étape de diffusion.
`[MASK]` représente une position encore masquée ; le texte intermédiaire peut
changer et n'est pas nécessairement rempli de gauche à droite. L'affichage
ajoute du travail, notamment pour transférer les tokens depuis le GPU en NF4.

Le terminal indique les phases de chargement ; en FP32 hybride, il affiche
aussi le nombre de blocs effectivement transférés sur GPU. Ce compteur ne
mesure pas le chargement total en RAM : des pages peuvent être lues plus tard.
Selon le runtime, Transformers affiche également l'avancement des fichiers
de poids.

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

Ce placement exige une marge importante en RAM et en VRAM ; le système peut
recourir au swap et lire les poids depuis le disque. Les mesures et limites
observées figurent dans [`benchmark.md`](benchmark.md).

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
combinaison tienne en mémoire. Le serveur traite une génération à la fois,
sans transmission de l'historique :
seul le dernier message utilisateur est utilisé. Il ne gère pas CORS pour une
page servie depuis une autre origine.

Les modèles, build products, jeux de réponses et résultats générés sont exclus
de Git.
