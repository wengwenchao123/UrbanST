<div align="center">
  <h3><b> UrbanST: A Fair and Scalable rban Spatio-Temporal Forecasting Benchmark and Toolkit. </b></h3>
</div>

---

##  Installation and Quick Start

UrbanST is an improvement based on BasicTS, and its installation and usage can refer to the tutorial of [BasicTS v0.5.8](https://github.com/GestaltCogTeam/BasicTS/tree/v0.5.8).

##  Datasets

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/file/d/1Jz6-aAD5Lh2sciJfqUKcWpPkg1EU8rb4/view?usp=drive_link).

Unzip the downloaded dataset files to the main file directory, the same directory as run.py.

## Model Training

```bash
python run.py --dataset {DATASET_NAME} --mode {MODE_NAME}
```
Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD8`, `PEMSD7(L)`, `PEMSD7(M)`

such as `python run.py --dataset PEMSD4`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the `pre-trained` folder.


##  Acknowledgement

BasicTS is developed based on [BasicTS](https://github.com/cnstark/easytorch), an easy-to-use and powerful open-source neural network training framework.

