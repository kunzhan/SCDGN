# Self-Contrastive Graph Diffusion Network
Contrastive learning has been proven to be a successful approach in graph self-supervised learning. Augmentation techniques and sampling strategies are crucial in contrastive learning, but in most existing works, augmentation techniques require careful design, and their sampling strategies can only capture a small amount of intrinsic supervision information. Additionally, the existing methods require complex designs to obtain two different representations of the data. To overcome these limitations, we propose a novel framework called the Self-Contrastive Graph Diffusion Network (SCGDN). Our framework consists of two main components: the Attentional Module (AttM) and the Diffusion Module (DiFM). AttM aggregates higher-order structure and feature information to get an excellent embedding, while DiFM balances the state of each node in the graph through Laplacian diffusion learning and allows the cooperative evolution of adjacency and feature information in the graph. Unlike existing methodologies, SCGDN is an augmentation-free approach that avoids "sampling bias" and semantic drift, without the need for pre-training.  We conduct a high-quality sampling of samples based on structure and feature information. If two nodes are neighbors, they are considered positive samples of each other. If two disconnected nodes are also unrelated on $k$NN graph, they are considered negative samples for each other. The contrastive objective reasonably uses our proposed sampling strategies, and the redundancy reduction term minimizes redundant information in the embedding and can well retain more discriminative information. In this novel framework, the graph self-contrastive learning paradigm gives expression to a powerful force. SCGDN effectively balances between preserving high-order structure information and avoiding overfitting. The results manifest that SCGDN can consistently generate outperformance over both the contrastive methods and the classical methods.

## Requirements
> Dependencies (with python >= 3.7): Main dependencies are
```
torch==1.8.1
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.6
torch-sparse==0.6.9
torch-spline-conv==1.2.1
torchdiffeq==0.2.1
```
> Commands to install all the dependencies in a new conda environment
```bash
conda create --name grand python=3.7
conda activate grand

pip install ogb pykeops
pip install torch==1.8.1
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

# or
conda create --name grand --file env.txt
```

```bash
mkdir best log
```

```py
python main.py --dataname cora --beta=3 --epochs=50 --hid_dim=512 --knn=21 --time=100
# ACC:74.79$\pm$0.38 NMI:56.86$\pm$0.42 ARI:52.61$\pm$0.33 F1:70.42$\pm$0.48

python main.py --dataname citeseer --beta=7 --epochs=50 --hid_dim=512 --knn=150 --time=160 
# ACC:69.62$\pm$0.02 NMI:44.35$\pm$0.03 ARI:45.43$\pm$0.02 F1:65.50$\pm$0.06

python main.py --dataname amap --epochs=20 --knn=19 --hid_dim=512 --time=10.0 --beta=5.0
# 78.91$\pm$0.00 NMI:72.53$\pm0.02 ARI:63.41$\pm$0.01 F1:75.27$\pm$0.01

python main.py --dataname bat --beta 0.7 --epochs=25 --hid_dim=64 --knn=21 --time=200.0 
# ACC:74.73 $\pm$ 0.23 NMI:52.63 $\pm$ 0.11 ARI:47.65 $\pm$ 0.18 F1:74.49 $\pm$0.26

python main.py --dataname eat --beta=6 --gamma=1.5 --epochs=30 --hid_dim=512 --knn=155 --time=15
# ACC:51.88 $\pm$ 0.00 NMI:32.49 $\pm$ 0.21 ARI:23.86 $\pm$ 0.08 F1:47.62 $\pm$ 0.02

python main.py --dataname corafull --beta=2 --epochs=12 --hid_dim=1024 --knn=73 --time=5 --gpu=-1
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@InProceedings{MaMM2023,
  author    = {Yixuan Ma and Kun Zhan},
  booktitle = {ACM MM},
  title     = {Self-contrastive graph diffusion network},
  year      = {2023},
  volume    = {31},
}
```
# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)
