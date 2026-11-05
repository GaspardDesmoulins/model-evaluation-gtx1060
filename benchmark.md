| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| dory ?B Q5_K - Medium          |   3.41 GiB |     3.98 B | CPU        |       6 |           pp128 |         14.76 ± 0.99 |
| dory ?B Q5_K - Medium          |   3.41 GiB |     3.98 B | CPU        |       6 |            tg64 |          3.34 ± 0.06 |
| dory ?B Q3_K - Medium          |   1.86 GiB |     3.98 B | CPU        |       6 |           pp128 |         15.04 ± 0.35 |
| dory ?B Q3_K - Medium          |   1.86 GiB |     3.98 B | CPU        |       6 |            tg64 |          4.45 ± 0.12 |

build: 98a74c97 (11551)
