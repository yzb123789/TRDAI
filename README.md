# Unified modelling of cancer druggability and therapeutic discovery from multi-omics and mechanistic knowledge
Here we present TRDAI, a unified framework that couples cancer druggability prediction with gene–drug interaction modelling by integrating multi-omics profiles, language-derived mechanistic knowledge and drug molecular structures.

## Framework Architecture
![TADE Architecture](https://github.com/yzb123789/TRDAI/blob/main/TRDAI.png)

---

## Reproducibility (Highly Recommended)
To ensure the immediate reproducibility of the core results and performance metrics reported in our manuscript, we provide an executable Code Ocean capsule. This environment is pre-configured with all necessary dependencies and the specific computing environment required for training.

---

## Reproducibility (Highly Recommended)


---

## Environment Setup
If you prefer to run the analysis locally, we recommend using a Linux environment (Ubuntu 22.04) configured with Python 3.10 and CUDA 11.8.

### 1. Requirements
* PyTorch == 2.1.0
* DGL == 2.1.0 (with CUDA 11.8 support)
* RDKit == 2022.3.3
* dgllife == 0.2.8

### 2. Installation
```bash
#
pip install category-encoders==2.6.4
pip install yacs==0.1.8
pip install tensorboard==2.2.2
pip install qhoptim==1.1.0
pip install scikit-learn==1.3.2
pip install numpy==1.21.4
pip install scipy==1.10.1
pip install protobuf==3.19.1
pip install pandas==2.0.3
pip install dgl==2.1.0+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dgllife==0.2.8
pip install rdkit==2022.3.3
pip install torchdata==0.7.1
pip install tqdm==4.66.5
pip install pydantic
pip install einops==0.8.0
pip install prettytable==3.11.0
```

## Usage
Druggable Gene Prediction: Refer to ./code/WCADANet/main.py to execute model training across pan-cancer datasets.
Gene Scoring & Evaluation: Refer to ./code/WCADANet/score.py to calculate druggability scores for candidate genes or to evaluate model performance on COSMIC、CIVIC、CGI test sets.
Gene-Drug Interaction Model Training: Refer to ./code/BiSLANet/train.py to train the Bi-SLA network for GDI prediction.
Cross-Dataset Validation: Use ./code/BiSLANet/test.py to evaluate the model on curated GDI test sets from COSMIC, CIViC, and CGI databases.


