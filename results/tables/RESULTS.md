### RMSE (avg over 5 datasets; mean ± std over 3 seeds; lower is better)

| name                | MCAR            | MAR             | MNAR            |
|:--------------------|:----------------|:----------------|:----------------|
| ReMasker++          | 0.1558 ± 0.0033 | 0.1531 ± 0.0099 | 0.1613 ± 0.0023 |
| ReMasker++ (W)      | 0.1607 ± 0.0022 | 0.1604 ± 0.0111 | 0.1660 ± 0.0016 |
| ReMasker (official) | 0.1712 ± 0.0030 | 0.1752 ± 0.0123 | 0.1778 ± 0.0052 |
| HyperImpute         | 0.1512 ± 0.0028 | 0.1379 ± 0.0045 | 0.1569 ± 0.0008 |
| MissForest          | 0.1680 ± 0.0006 | 0.1738 ± 0.0086 | 0.1716 ± 0.0019 |
| MICE                | 0.2375 ± 0.0018 | 0.2232 ± 0.0090 | 0.2448 ± 0.0034 |
| GAIN                | 0.2101 ± 0.0040 | 0.2055 ± 0.0054 | 0.2209 ± 0.0074 |
| MIWAE               | 0.2672 ± 0.0253 | 0.2812 ± 0.0028 | 0.2821 ± 0.0202 |
| Sinkhorn            | 0.2210 ± 0.0032 | 0.2367 ± 0.0110 | 0.2250 ± 0.0018 |
| SoftImpute          | 0.2632 ± 0.0023 | 0.2446 ± 0.0059 | 0.2675 ± 0.0080 |
| Mean                | 0.2178 ± 0.0002 | 0.2279 ± 0.0026 | 0.2269 ± 0.0026 |

### WD (avg over 5 datasets; mean ± std over 3 seeds; lower is better)

| name                | MCAR            | MAR             | MNAR            |
|:--------------------|:----------------|:----------------|:----------------|
| ReMasker++          | 0.0594 ± 0.0009 | 0.0582 ± 0.0053 | 0.0643 ± 0.0020 |
| ReMasker++ (W)      | 0.0510 ± 0.0010 | 0.0520 ± 0.0055 | 0.0568 ± 0.0012 |
| ReMasker (official) | 0.0725 ± 0.0035 | 0.0875 ± 0.0045 | 0.0737 ± 0.0045 |
| HyperImpute         | 0.0554 ± 0.0010 | 0.0520 ± 0.0069 | 0.0526 ± 0.0031 |
| MissForest          | 0.0838 ± 0.0016 | 0.0936 ± 0.0076 | 0.0866 ± 0.0033 |
| MICE                | 0.0546 ± 0.0006 | 0.0522 ± 0.0035 | 0.0597 ± 0.0009 |
| GAIN                | 0.1104 ± 0.0083 | 0.1154 ± 0.0063 | 0.1168 ± 0.0045 |
| MIWAE               | 0.1174 ± 0.0188 | 0.1277 ± 0.0073 | 0.1290 ± 0.0206 |
| Sinkhorn            | 0.1260 ± 0.0009 | 0.1398 ± 0.0026 | 0.1299 ± 0.0013 |
| SoftImpute          | 0.0820 ± 0.0023 | 0.0980 ± 0.0081 | 0.1025 ± 0.0200 |
| Mean                | 0.1677 ± 0.0005 | 0.1741 ± 0.0023 | 0.1740 ± 0.0016 |

### DS_XGB (avg over 5 datasets; mean ± std over 3 seeds; higher is better)

| name                | MCAR            | MAR             | MNAR            |
|:--------------------|:----------------|:----------------|:----------------|
| ReMasker++          | 0.6907 ± 0.0173 | 0.7874 ± 0.0903 | 0.6536 ± 0.0295 |
| ReMasker++ (W)      | 0.6854 ± 0.0162 | 0.7877 ± 0.0898 | 0.6607 ± 0.0197 |
| ReMasker (official) | 0.6766 ± 0.0139 | 0.7599 ± 0.1088 | 0.6540 ± 0.0303 |
| HyperImpute         | 0.6918 ± 0.0160 | 0.7884 ± 0.0903 | 0.6638 ± 0.0177 |
| MissForest          | 0.6816 ± 0.0123 | 0.7794 ± 0.0920 | 0.6661 ± 0.0143 |
| MICE                | 0.6184 ± 0.0363 | 0.7404 ± 0.1118 | 0.5949 ± 0.0290 |
| GAIN                | 0.6588 ± 0.0265 | 0.7695 ± 0.0940 | 0.6452 ± 0.0252 |
| MIWAE               | 0.6523 ± 0.0042 | 0.7565 ± 0.0964 | 0.6093 ± 0.0331 |
| Sinkhorn            | 0.6661 ± 0.0240 | 0.7669 ± 0.0918 | 0.6369 ± 0.0115 |
| SoftImpute          | 0.6435 ± 0.0133 | 0.7592 ± 0.0958 | 0.6207 ± 0.0334 |
| Mean                | 0.6719 ± 0.0325 | 0.7774 ± 0.0881 | 0.6706 ± 0.0145 |

### DS_LIN (avg over 5 datasets; mean ± std over 3 seeds; higher is better)

| name                | MCAR            | MAR             | MNAR            |
|:--------------------|:----------------|:----------------|:----------------|
| ReMasker++          | 0.5936 ± 0.0142 | 0.6329 ± 0.0407 | 0.5740 ± 0.0137 |
| ReMasker++ (W)      | 0.5862 ± 0.0143 | 0.6279 ± 0.0442 | 0.5710 ± 0.0133 |
| ReMasker (official) | 0.5847 ± 0.0097 | 0.6213 ± 0.0407 | 0.5667 ± 0.0156 |
| HyperImpute         | 0.6010 ± 0.0114 | 0.6325 ± 0.0336 | 0.5786 ± 0.0134 |
| MissForest          | 0.5855 ± 0.0065 | 0.6231 ± 0.0364 | 0.5695 ± 0.0121 |
| MICE                | 0.5069 ± 0.0096 | 0.5840 ± 0.0533 | 0.4880 ± 0.0124 |
| GAIN                | 0.5614 ± 0.0121 | 0.6075 ± 0.0371 | 0.5439 ± 0.0223 |
| MIWAE               | 0.5023 ± 0.0316 | 0.5714 ± 0.0596 | 0.4835 ± 0.0297 |
| Sinkhorn            | 0.5451 ± 0.0013 | 0.6000 ± 0.0414 | 0.5431 ± 0.0150 |
| SoftImpute          | 0.5302 ± 0.0098 | 0.5926 ± 0.0313 | 0.5086 ± 0.0073 |
| Mean                | 0.5645 ± 0.0041 | 0.6110 ± 0.0280 | 0.5524 ± 0.0133 |

### Average rank across 15 (dataset × mechanism) settings (1 = best)

| name                |   rmse |    wd |   ds_xgb |   overall |
|:--------------------|-------:|------:|---------:|----------:|
| HyperImpute         |   1.53 |  2.93 |     3.13 |      2.53 |
| ReMasker++ (W)      |   3.2  |  1.93 |     2.73 |      2.62 |
| ReMasker++          |   1.6  |  3.27 |     3.6  |      2.82 |
| MissForest          |   4.07 |  6.2  |     4.6  |      4.96 |
| ReMasker (official) |   4.6  |  5.4  |     5.87 |      5.29 |
| MICE                |   8.67 |  2.73 |     8.73 |      6.71 |
| GAIN                |   7    |  8.4  |     6.87 |      7.42 |
| Sinkhorn            |   7.73 |  8.87 |     7.6  |      8.07 |
| Mean                |   8    | 10.8  |     5.47 |      8.09 |
| SoftImpute          |   9.67 |  6.6  |     8.6  |      8.29 |
| MIWAE               |   9.93 |  8.87 |     8.8  |      9.2  |

### Number of settings (out of 15) where each method is best

|                     |   rmse |   wd |   ds_xgb |
|:--------------------|-------:|-----:|---------:|
| ReMasker++          |      6 |    0 |        3 |
| ReMasker++ (W)      |      0 |    4 |        3 |
| ReMasker (official) |      0 |    0 |        1 |
| HyperImpute         |      9 |    9 |        3 |
| MissForest          |      0 |    0 |        1 |
| MICE                |      0 |    2 |        1 |
| GAIN                |      0 |    0 |        0 |
| MIWAE               |      0 |    0 |        1 |
| Sinkhorn            |      0 |    0 |        0 |
| SoftImpute          |      0 |    0 |        0 |
| Mean                |      0 |    0 |        2 |

### Downstream utility: mean change vs. classifier/regressor trained on complete data (AUROC for raisin/wine, R² for yacht/airfoil/compression; closer to 0 is better)

| name                |   XGBoost Δ vs oracle |   Linear Δ vs oracle |
|:--------------------|----------------------:|---------------------:|
| ReMasker++          |               -0.2076 |              -0.0954 |
| ReMasker++ (W)      |               -0.2069 |              -0.1005 |
| ReMasker (official) |               -0.2214 |              -0.1047 |
| HyperImpute         |               -0.2035 |              -0.0915 |
| MissForest          |               -0.2092 |              -0.1028 |
| MICE                |               -0.2669 |              -0.1693 |
| GAIN                |               -0.227  |              -0.1246 |
| MIWAE               |               -0.2455 |              -0.1765 |
| Sinkhorn            |               -0.2282 |              -0.1328 |
| SoftImpute          |               -0.2437 |              -0.1517 |
| Mean                |               -0.2115 |              -0.1196 |

Oracle scores (complete data):

| dataset     |   XGBoost (oracle) |   Linear (oracle) |
|:------------|-------------------:|------------------:|
| airfoil     |             0.9162 |            0.5099 |
| compression |             0.9294 |            0.6041 |
| raisin      |             0.9225 |            0.9282 |
| wine        |             0.8252 |            0.8096 |
| yacht       |             0.9975 |            0.626  |

### Ablation (avg over 5 datasets × 3 mechanisms × 3 seeds; Δ = mean per-setting relative change)

| variant                       |   RMSE |     WD |    XGB |   ΔRMSE % vs ReMasker |   ΔWD % vs ReMasker |   ΔRMSE MNAR % |
|:------------------------------|-------:|-------:|-------:|----------------------:|--------------------:|---------------:|
| ReMasker (official)           | 0.1748 | 0.0779 | 0.6968 |                0      |              0      |         0      |
| Batched ReMasker (ours, A)    | 0.1761 | 0.0816 | 0.7052 |               -0.1521 |              0.0207 |        -1.8451 |
| A + pattern embedding (B)     | 0.2072 | 0.0899 | 0.6888 |               17.7172 |             20.942  |        17.0095 |
| A + Wasserstein loss (D)      | 0.1908 | 0.0694 | 0.7001 |                7.4281 |            -13.2616 |         4.4558 |
| A + random mask 30% (control) | 0.1679 | 0.053  | 0.7062 |               -6.3062 |            -32.7939 |        -3.8511 |
| A + copy-mask 50% (C)         | 0.1593 | 0.0655 | 0.7121 |               -9.9728 |            -20.1817 |        -9.216  |
| A + MC inference (E)          | 0.166  | 0.0823 | 0.7061 |               -6.1296 |              1.7175 |        -6.5241 |
| A + progressive (E+F)         | 0.1739 | 0.0912 | 0.7089 |               -1.9181 |             12.3579 |         0.4792 |
| ReMasker++                    | 0.1567 | 0.0607 | 0.7106 |              -11.4962 |            -25.3711 |        -9.6729 |
| ReMasker++ + MC inference (E) | 0.1602 | 0.0791 | 0.7098 |               -9.3331 |             -2.1624 |        -8.9823 |
| ReMasker++ + progressive (F)  | 0.1639 | 0.0611 | 0.7076 |               -8.0409 |            -23.9217 |        -4.4518 |
| ReMasker++ (W)                | 0.1624 | 0.0533 | 0.7113 |               -8.7754 |            -34.552  |        -7.3374 |

### Uncertainty quality (Spearman between predicted uncertainty and absolute error on missing cells)

| name                          |   unc_spearman |   rmse_conf25 |   rmse_unconf25 |   error ratio (least/most confident) |
|:------------------------------|---------------:|--------------:|----------------:|-------------------------------------:|
| A + MC inference (E)          |         0.1715 |        0.1624 |          0.1754 |                               1.0797 |
| A + progressive (E+F)         |         0.2237 |        0.1518 |          0.1905 |                               1.2554 |
| ReMasker++                    |         0.4699 |        0.0668 |          0.2316 |                               3.4676 |
| ReMasker++ (W)                |         0.4761 |        0.0714 |          0.2355 |                               3.2991 |
| ReMasker++ + MC inference (E) |         0.1878 |        0.1564 |          0.1707 |                               1.0916 |
| ReMasker++ + progressive (F)  |         0.435  |        0.0909 |          0.2334 |                               2.5673 |
| pp_c0.5+mc                    |         0.1578 |        0.1582 |          0.1672 |                               1.0569 |
| pp_c0.5+prog                  |         0.1949 |        0.1483 |          0.1756 |                               1.1843 |
| pp_c1_w0.3_v+mc               |         0.2182 |        0.1547 |          0.186  |                               1.2022 |
| pp_c1_w0.3_v+prog             |         0.4372 |        0.0947 |          0.2399 |                               2.534  |
| pp_m0.3+mc                    |         0.2463 |        0.1663 |          0.2015 |                               1.2119 |
| pp_m0.3+prog                  |         0.3064 |        0.1393 |          0.2087 |                               1.4981 |
| pp_p+mc                       |         0.1382 |        0.1993 |          0.2165 |                               1.0866 |
| pp_p+prog                     |         0.1929 |        0.1863 |          0.2269 |                               1.2181 |
| pp_w1+mc                      |         0.2198 |        0.1669 |          0.1978 |                               1.1852 |
| pp_w1+prog                    |         0.2816 |        0.1502 |          0.2151 |                               1.4323 |

ReMasker++ (learned uncertainty head) by mechanism:

| mech   |   unc_spearman |   rmse_conf25 |   rmse_unconf25 |
|:-------|---------------:|--------------:|----------------:|
| MAR    |         0.4499 |        0.0513 |          0.2343 |
| MCAR   |         0.46   |        0.0701 |          0.2251 |
| MNAR   |         0.4998 |        0.079  |          0.2353 |

### Stress test: RMSE under MNAR with increasing missing rate (mean over seeds)

|                                                  |    0.3 |    0.5 |    0.7 |
|:-------------------------------------------------|-------:|-------:|-------:|
| ('airfoil', 'HyperImpute')                       | 0.2075 | 0.2475 | 0.3    |
| ('airfoil', 'Mean')                              | 0.2938 | 0.2923 | 0.2925 |
| ('airfoil', 'MissForest')                        | 0.2373 | 0.2791 | 0.3191 |
| ('airfoil', 'ReMasker (official)')               | 0.2475 | 0.2663 | 0.3102 |
| ('airfoil', 'ReMasker++')                        | 0.2288 | 0.2534 | 0.2784 |
| ('airfoil', 'ReMasker++ + MC inference (E)')     | 0.234  | 0.253  | 0.2771 |
| ('airfoil', 'ReMasker++ + progressive (F)')      | 0.2492 | 0.2866 | 0.3137 |
| ('compression', 'HyperImpute')                   | 0.1494 | 0.1871 | 0.2278 |
| ('compression', 'Mean')                          | 0.2257 | 0.2257 | 0.2268 |
| ('compression', 'MissForest')                    | 0.1763 | 0.1947 | 0.2327 |
| ('compression', 'ReMasker (official)')           | 0.1717 | 0.219  | 0.2793 |
| ('compression', 'ReMasker++')                    | 0.1579 | 0.1888 | 0.2135 |
| ('compression', 'ReMasker++ + MC inference (E)') | 0.1633 | 0.1864 | 0.2099 |
| ('compression', 'ReMasker++ + progressive (F)')  | 0.1703 | 0.2111 | 0.2561 |
| ('raisin', 'HyperImpute')                        | 0.0684 | 0.1085 | 0.1381 |
| ('raisin', 'Mean')                               | 0.1838 | 0.1743 | 0.1667 |
| ('raisin', 'MissForest')                         | 0.0827 | 0.0943 | 0.116  |
| ('raisin', 'ReMasker (official)')                | 0.0894 | 0.1044 | 0.1425 |
| ('raisin', 'ReMasker++')                         | 0.0721 | 0.0894 | 0.1191 |
| ('raisin', 'ReMasker++ + MC inference (E)')      | 0.072  | 0.0901 | 0.119  |
| ('raisin', 'ReMasker++ + progressive (F)')       | 0.0742 | 0.094  | 0.1209 |

### Mean wall-clock time per imputation run (s, 1 CPU thread)

| method                        |   time |
|:------------------------------|-------:|
| Mean                          |    0   |
| MICE                          |    1.1 |
| GAIN                          |    2   |
| SoftImpute                    |    3.8 |
| MIWAE                         |   12.6 |
| HyperImpute                   |   15.3 |
| MissForest                    |   18.8 |
| ReMasker (official)           |   35.7 |
| ReMasker++ + MC inference (E) |   56.3 |
| ReMasker++                    |   56.3 |
| ReMasker++ + progressive (F)  |   56.3 |
| pp_c1_w0.3_v+mc               |   58.7 |
| pp_c1_w0.3_v+prog             |   58.7 |
| ReMasker++ (W)                |   58.7 |
| pp_m0.3+mc                    |   61   |
| A + random mask 30% (control) |   61   |
| pp_m0.3+prog                  |   61   |
| A + progressive (E+F)         |   61.5 |
| Batched ReMasker (ours, A)    |   61.5 |
| A + MC inference (E)          |   61.5 |
| A + copy-mask 50% (C)         |   61.8 |
| pp_c0.5+mc                    |   61.8 |
| pp_c0.5+prog                  |   61.8 |
| A + pattern embedding (B)     |   62.1 |
| pp_p+mc                       |   62.1 |
| pp_p+prog                     |   62.1 |
| A + Wasserstein loss (D)      |   64.1 |
| pp_w1+mc                      |   64.1 |
| pp_w1+prog                    |   64.1 |
| Sinkhorn                      |  328.1 |
