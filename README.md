

## Dependencies and Installion
```
mp-api==0.35.1
pymatgen==2023.8.10
numpy==1.24.1
pandas==2.0.3
scikit-learn==1.3.2
torch==2.1.0+cu118
torch-cluster==1.6.3+pt21cu118
torch-geometric==2.6.1
torch-scatter==2.1.2+pt21cu118
torch-sparse==0.6.18+pt21cu118
torchaudio==2.1.0+cu118
torchvision==0.16.0+cu118
tqdm==4.67.1
transformers==4.24.0
xgboost==2.1.4
```

We recommend creating a new conda environment to install the dependencies:
```
conda env remove --name cross
conda create -y -n cross python=3.9
conda activate cross
pip install transformers==4.24.0
```

## Datasets

We use the open computational database from [Materials Project](https://next-503gen.materialsproject.org).
To ensure maximum data consistency and eliminate structural ambiguities before model learning, a strict filtering pipeline is used to exclude compounds with disordered lattices, and incomplete property tags. This process yields a finalized high-throughput dataset of approximately 1,600 pristine ABX3 perovskite compounds. For each compound instance originated from identical crystallographic structures, the curated data consists of a XRD profile, K-edge XAFS spectra, and the corresponding targets: formation energy, Fermi energy, and bandgap.

## Quick Start
Extract the data archive [data.zip](https://drive.google.com/file/d/19HUld80pnwEvODCQux0NhWW3jGv922rZ/view?usp=sharing) and maintain the following directory structure:
```text
InterLLC/
├── data/
├── res/
├── results/
└── main.py
```

To train a multimodal InterLLC model using XRD and XAFS.

```
python main.py [<args>] [-h | --help]
```

e.g. To train a InterLLC model.

```
python main.py --data abx3 --fun train
```

To train a unimodal XRD or XAFS model .

```
python main.py --mode single --spectrum xrd --xrd_model cnn9
python main.py --mode single --spectrum xafs --xas_model mlp
```















[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLinaZhaoAIGroup%2FInterLLC&count_bg=%2F7928CA&title_bg=%2F555555&icon=&icon_color=%2FE7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
