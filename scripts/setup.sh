conda create -y -n graph2smiles python=3.6 tqdm
conda activate graph2smiles
conda install -y pytorch=1.6.0 torchvision cudatoolkit=10.1 torchtext -c pytorch
conda install -y rdkit -c conda-forge

# pip dependencies
pip install gdown OpenNMT-py==1.2.0 networkx==2.5 selfies==1.0.3
