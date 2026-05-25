# AI_for_chemistry_Project
Methyl cation afinity (MCA) and methyl anion affinity (MAA) prediction using different ML models
# Setup

## 1. Clone the repo
git clone https://github.com/ton-user/ton-repo.git
cd ton-repo

## 2. Download the dataset
The dataset (~1.1 GB) is hosted externally. Run the following script to download and extract it:

python download_data.py

This will create `df_elec.parquet` (or `data/QMdata4ML/df_elec.csv.gz`) in the working directory.

## 3. Run the model
python gnn_esnuel_siteaware_v4_nofilter.py --data_path df_elec.parquet
