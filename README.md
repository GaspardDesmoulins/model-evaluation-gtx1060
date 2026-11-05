# Dory sur CPU

Cet espace est réservé à l'échange avec Gemini concernant la mise en place
du modèle Dory de NVIDIA sur un PC sans GPU.

## Échange avec Gemini
```

Pour s'adapter à votre GTX 1060 (3 Go de VRAM) sur votre PC personnel, il est préférable d'abandonner la version Q5_K_M (3,5 Go) qui est trop lourde pour votre carte. Nous allons plutôt télécharger la version Q3_K_M (environ 2,4 Go). Grâce à cette taille réduite, le modèle pourra tenir presque entièrement dans vos 3 Go de VRAM, ce qui maximisera l'accélération de votre carte graphique.
Voici la procédure complète mise à jour, incluant la commande de récupération automatique via Ollama.

Étape 1 : Préparer l'environnement CUDA sous Ubuntu (WSL)

Installez le compilateur CUDA (nvcc) et les bibliothèques système requises pour la compilation :
bash
sudo apt update
sudo apt install -y build-essential cmake git libssl-dev nvidia-cuda-toolkit
Utilisez le code avec précaution.

Étape 2 : Télécharger les poids adaptés (Q3_K_M) avec Ollama

Nous utilisons ollama pull pour récupérer directement la version légère (3-bit) depuis Hugging Face sans lancer le modèle :
bash
ollama pull hf.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF:Q3_K_M
Utilisez le code avec précaution.

Étape 3 : Créer le lien symbolique vers le bon Blob d'Ollama

Puisque vous venez de télécharger le modèle, nous utilisons une commande dynamique (find) pour cibler automatiquement le fichier qui pèse environ 2,4 Go dans le stockage d'Ollama, évitant ainsi d'avoir à chercher manuellement le hash sha256 :
bash
# Recherche automatique du fichier entre 2,1 Go et 2,7 Go et création du lien
BLOB_PATH=$(sudo find /usr/share/ollama/.ollama/models/blobs/ -type f -size +2100M -size -2700M | head -n 1)
ln -s "$BLOB_PATH" ./dory-q3_k_m.gguf
Utilisez le code avec précaution.

Étape 4 : Récupérer le code source de la branche Dory

Initialisez le dépôt llama.cpp et basculez sur le commit de développement dédié à l'architecture en boucle :
bash
git init llama.cpp-dory
cd llama.cpp-dory
git remote add origin https://github.com
git fetch --depth 1 origin 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
git checkout --detach FETCH_HEAD
Utilisez le code avec précaution.

Étape 5 : Compiler llama.cpp avec le support NVIDIA CUDA

Générez les fichiers de configuration avec le flag d'activation CUDA, puis lancez la compilation :
bash
# 1. Configuration CMake avec support CUDA
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_USE_PREBUILT_UI=ON \
    -DLLAMA_OPENSSL=ON

# 2. Build du serveur
cmake --build build --parallel $(nproc) --target llama-server
Utilisez le code avec précaution.

Étape 6 : Lancer le serveur en mode Hybride optimisé

Comme la version Q3_K_M ne pèse que 2,4 Go, nous pouvons tenter de pousser 16 couches (sur les 20 du modèle) directement dans la VRAM de votre carte graphique.
bash
./build/bin/llama-server \
    -m ../dory-q3_k_m.gguf \
    --alias dory \
    --host 127.0.0.1 --port 8080 \
    -ngl 16 \
    -t 4 \
    --ctx-size 4096 --parallel 1 \
    --batch-size 512 --ubatch-size 512 \
    --cache-type-k f16 --cache-type-v f16 \
    --flash-attn off \
    --no-context-shift \
    --jinja --reasoning on \
    --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 --repeat-penalty 1 \
    --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop"
Utilisez le code avec précaution.
Une fois le téléchargement Ollama terminé, dites-moi :
• Si la commande ln -s a bien réussi à lier le fichier (vérifiez avec un rapide ls -lh ../dory-q3_k_m.gguf).
• Si vous observez un message d'erreur "Out of VRAM" au lancement du serveur (si c'est le cas, nous baisserons simplement le paramètre -ngl 16 à 12).

```

## Installation locale validée

- Fork Dory de llama.cpp : `brianchmielNV/llama.cpp`
- Révision : `98a74c97ea09ac72f6ceda1559808fc0a2f447ce`
- Compilation CPU : `dory/llama.cpp-dory/build-cpu/bin/`
- Outils : `llama-bench` et `llama-quantize`

### Modèles

| Quantification | Fichier | Taille | Provenance |
|---|---|---:|---|
| Q5_K_M | `models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` | 3,673,288,864 octets | Export NVIDIA, SHA-256 vérifié |
| Q3_K_M | `models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q3_K_M.gguf` | 1,86 Gio | Requantifié localement depuis le Q5 |

NVIDIA ne distribue actuellement que le Q5_K_M. Le Q3_K_M a donc été créé
localement avec `llama-quantize --allow-requantize`; il sert à réduire
l'occupation mémoire et ses résultats peuvent être moins fidèles qu'un Q3
quantifié directement depuis les poids BF16.

### Benchmark CPU

Machine : Intel Core i7-8750H (6 cœurs physiques), CPU uniquement, 6 threads,
3 répétitions, contexte de 128 tokens et génération de 64 tokens.

| Quantification | Traitement du prompt | Génération |
|---|---:|---:|
| Q5_K_M | 14,76 ± 0,99 tok/s | 3,34 ± 0,06 tok/s |
| Q3_K_M | 15,04 ± 0,35 tok/s | 4,45 ± 0,12 tok/s |

Le rapport brut est disponible dans [`benchmark.md`](benchmark.md).