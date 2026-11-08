## Débit Dory

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| dory ?B Q5_K - Medium          |   3.41 GiB |     3.98 B | CPU        |       6 |           pp128 |         14.76 ± 0.99 |
| dory ?B Q5_K - Medium          |   3.41 GiB |     3.98 B | CPU        |       6 |            tg64 |          3.34 ± 0.06 |
| dory ?B Q3_K - Medium          |   1.86 GiB |     3.98 B | CPU        |       6 |           pp128 |         15.04 ± 0.35 |
| dory ?B Q3_K - Medium          |   1.86 GiB |     3.98 B | CPU        |       6 |            tg64 |          4.45 ± 0.12 |

build: 98a74c97 (11551)

Les mesures CUDA correspondantes, avec une GTX 1060 3 Go, sont de 166,01 tok/s
en prompt et 4,13 tok/s en génération pour Dory Q3 entièrement offloadé, et de
113,94 tok/s / 2,79 tok/s pour Dory Q5 hybride.

## IFEval

Échantillon déterministe de 16 prompts, 512 tokens maximum et température 0.
Ce ne sont pas des scores IFEval officiels complets.

| Modèle | Réglage | Strict | Relâché | Prompt | Génération | Fenêtre IFEval | Pic RAM | Pic VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dory Q3 | 1 boucle | 37,5 % | 43,75 % | 150,60 tok/s | 10,09 tok/s | 8 min 38 s | 3,01 Gio | 2 479 Mio |
| Dory Q3 | 2 boucles | 43,75 % | 50,0 % | — | — | — | 3,62 Gio | 2 805 Mio |
| Dory Q3 | 4 boucles | 37,5 % | 37,5 % | — | — | — | 4,70 Gio | 2 923 Mio |
| Dory Q5 | 2 boucles | 56,25 % | 56,25 % | 40,65 tok/s | 4,46 tok/s | 18 min 05 s | 5,69 Gio | 2 275 Mio |
| Dory Q5 | 4 boucles | 56,25 % | 56,25 % | 20,12 tok/s | 2,80 tok/s | 28 min 51 s | 6,18 Gio | 2 275 Mio |
| Bonsai-2 4B Q2_0 | Ollama | 43,75 % | 50,0 % | — | 15,77 tok/s | ~5 min 35 s | — | 1 693 Mio |
| Qwen3.5 9B | Ollama | 62,5 % | 68,75 % | — | 4,36 tok/s | ~21 min 30 s | — | 1 459 Mio |
| CoDa 1.7B | NF4, 64 étapes | 0 % | 0 % | — | — | 25 min 16 s | 5,43 Gio | 2 447 Mio |
| CoDa 1.7B | NF4, 256 étapes | 0 % | 0 % | — | — | 1 h 10 min 56 s | 5,43 Gio | 2 447 Mio |

CoDa est un modèle de diffusion spécialisé code. À 64 étapes, huit réponses
sur seize sont vides ; son résultat IFEval ne permet donc pas de le
recommander pour les tâches généralistes. Son exécution complète dure 25 min
16 s : 9 min 32 s de chargement, puis 16 générations à 59,0 s de moyenne.

Les scripts Dory et Ollama antérieurs ne sauvegardaient pas la durée
chronométrée. Les débits Dory sont reconstitués à partir des 16 mesures du
serveur ; les débits Ollama sont reconstitués à partir des 16 requêtes valides
du journal système. Les fenêtres Dory excluent l'initialisation du serveur ;
les durées Ollama sont approximatives et excluent le chargement de contrôle.

## CoDa 1.7B : microbenchmark code

Poids officiels FP32 chargés en NF4 4 bits à l'exécution, attention SDPA math,
quatre prompts de code et 128 tokens maximum. Les poids sources font 8,12 Go ;
le chargement est la principale limite pratique.

| Étapes de diffusion | Durée totale | Chargement | Génération moyenne | Pic RAM | Pic VRAM |
|---:|---:|---:|---:|---:|---:|
| 64 | 11 min 06 s | 586 s | 20,0 s | 5,43 Gio | 2 079 Mio |
| 256 | 13 min 55 s | 524 s | 77,7 s | 5,43 Gio | 2 079 Mio |

À 64 étapes, une sortie est correcte, une est presque correcte mais omet
`import re`, et deux sont invalides ou tronquées. À 256 étapes, seule la
fonction `add` est correcte : la qualité ne justifie pas le temps de calcul
supplémentaire.

## Benchmark code commun

Quatre prompts Python, température 0, 128 tokens maximum. Les réponses ont été
inspectées statiquement ; aucun code généré n'a été exécuté.

| Modèle | Réglage | Sorties correctes | Durée totale | Chargement | Génération moyenne | Pic RAM | Pic VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| Dory Q3 | 2 boucles, 193 couches GPU | 4/4 | 1 min 01 s | 30,3 s | 6,56 s | 2,22 Gio | 2 805 Mio |
| Dory Q5 | 2 boucles, offload ajusté | 4/4 | 1 min 29 s | 51,1 s | 8,73 s | 4,68 Gio | 2 275 Mio |
| Qwen3.5 9B | Ollama | 4/4 | 2 min 26 s | 91,8 s | 4,56 tok/s | — | 1 459 Mio |

Qwen est le plus lent à charger et à générer ; Dory Q3 est le plus rapide sur
ce petit jeu de code. Ces réponses sont un contrôle qualitatif réduit, pas un
remplacement de HumanEval.

## Export CoDa NF4 persistant

Le checkpoint FP32 de 8,12 Go a été chargé une fois puis sauvegardé en NF4 sur
`D:`. L'export pèse 2,05 Go et a été validé par rechargement et génération.

| Chargement FP32 → NF4 | Export | Rechargement NF4 | Taille exportée |
|---:|---:|---:|---:|
| 8 min 13 s | 22,2 s | 14,4 s | 2,05 Go |

## Tests CoDa FP32 hybride

Un passage avant FP32 a été validé avec l'embedding et 6 blocs sur GPU, 17
blocs en RAM et 5 blocs offloadés sur `D:`. La génération diffusion échoue
ensuite dans les hooks de dispatch d'Accelerate, incompatibles avec cette
architecture personnalisée.

| Placement | Pic VRAM | Pic RAM | Passage avant | Génération diffusion |
|---|---:|---:|---:|---|
| 6 GPU, 17 RAM, 5 disque | 3 548 Mio | 6,06 Gio | 53,4 s | Échec du dispatch |
| 6 GPU, 22 RAM, 0 disque | 3 548 Mio | 6,71 Gio | 4,85 s | Échec du dispatch |
| 0 GPU, 28 RAM, chargement direct | 0 Mio | 7,45 Gio | 421,1 s | 64 étapes / 128 tokens : 172,8 s |
| 8 GPU, 20 RAM, transferts manuels | 2 749 Mio | 8,77 Gio | 135,0 s | 64 étapes / 128 tokens : 146,5 s |
| 8 GPU, 20 RAM, transferts manuels | 2 749 Mio | 8,77 Gio | 150,7 s | 768 étapes / 128 tokens : 1 741,7 s |

Les deux premiers essais utilisent `Accelerate`, qui retourne 34 logits pour un
masque de 35 positions pendant le débruitage. Le chargement direct en RAM
résout ce défaut ; huit blocs peuvent ensuite être déplacés manuellement sur la
GTX 1060, tandis que les activations sont transférées explicitement une seule
fois vers les blocs CPU. La génération GPU hybride est valide, mais le gain à
64 étapes reste modeste et la qualité de CoDa demeure insuffisante.

Le test à 768 étapes a utilisé un seul prompt demandant une fonction `add`.
Le chargement a pris 37,8 s, puis la génération 29 min 02 s. La réponse
contient une fonction correcte suivie d'une balise Markdown fermante isolée.
Ce n'est pas un benchmark HumanEval ni une comparaison statistique de qualité.
