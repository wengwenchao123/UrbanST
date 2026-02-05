<div align="center">
  <h3><b> UrbanST: A Fair and Scalable rban Spatio-Temporal Forecasting Benchmark and Toolkit. </b></h3>
</div>

---

##  Installation and Quick Start

UrbanST is an improvement based on BasicTS, and its installation and usage can refer to the tutorial of [BasicTS v0.5.8](https://github.com/GestaltCogTeam/BasicTS/tree/v0.5.8).

## 🔧 Install Dependencies

### PyTorch

BasicTS is very flexible regarding PyTorch versions. You can https://pytorch.org/get-started/previous-versions/ according to your Python version. We recommend using `pip` for installation.

### Example Setups

#### Example 1: Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 (Recommended)

```bash
# Install Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Example 2: Python 3.9 + PyTorch 1.10.0 + CUDA 11.1

```bash
# Install Python
conda create -n BasicTS python=3.9
conda activate BasicTS
# Install PyTorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

##  Datasets

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/file/d/1Jz6-aAD5Lh2sciJfqUKcWpPkg1EU8rb4/view?usp=drive_link).

Unzip the downloaded dataset files to the main file directory, the same directory as run.py.

## Train Model

You can run the following command to train your model:

    ```bash
    python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'
    ```

Replace `${DATASET_NAME}` and `${MODEL_NAME}` with any supported models and datasets. For example, to run Graph WaveNet on the METR-LA dataset:

    ```bash
    python experiments/train.py -c baselines/GWNet/METR-LA.py --gpus '0'
    ```

## How to Evaluate Your Model

You can run the following command to evaluate your model:

```bash
python experiments/evaluate.py -cfg {CONFIG_FILE}.py -ckpt {CHECKPOINT_PATH}.pth -g 0
```


##  Acknowledgement

UrbanST is developed based on [BasicTS]https://github.com/GestaltCogTeam/BasicTS/tree/v0.5.8)), an easy-to-use and powerful open-source neural network training framework.

