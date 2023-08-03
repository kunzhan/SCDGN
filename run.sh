# !/bin/bash

python main.py --dataname cora --beta=3 --epochs=50 --hid_dim=512 --knn=21 --time=100
# ACC:74.79$\pm$0.38 NMI:56.86$\pm$0.42 ARI:52.61$\pm$0.33 F1:70.42$\pm$0.48

python main.py --dataname citeseer --beta=7 --epochs=50 --hid_dim=512 --knn=150 --time=160 
# ACC:69.62$\pm$0.02 NMI:44.35$\pm$0.03 ARI:45.43$\pm$0.02 F1:65.50$\pm$0.06

# python main.py --dataname amap --epochs=20 --knn=19 --hid_dim=512 --time=10.0 --beta=5.0
# 78.91$\pm$0.00 NMI:72.53$\pm0.02 ARI:63.41$\pm$0.01 F1:75.27$\pm$0.01

python main.py --dataname bat --beta 0.7 --epochs=25 --hid_dim=64 --knn=21 --time=200.0 
# ACC:74.73 $\pm$ 0.23 NMI:52.63 $\pm$ 0.11 ARI:47.65 $\pm$ 0.18 F1:74.49 $\pm$0.26

python main.py --dataname eat --beta=6 --gamma=1.5 --epochs=30 --hid_dim=512 --knn=155 --time=15
# ACC:51.88 $\pm$ 0.00 NMI:32.49 $\pm$ 0.21 ARI:23.86 $\pm$ 0.08 F1:47.62 $\pm$ 0.02
