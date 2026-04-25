# IAENet
The repo is the official implementation for the paper: " Early Warning of Intraoperative Adverse Events via Transformer-Driven Multi-Label Learning". AAAI 2026！

***Abstract*** - Early warning of intraoperative adverse events plays a vital role in reducing surgical risk and improving patient safety. While deep learning has shown promise in predicting the single adverse event, several key challenges remain: overlooking adverse event dependencies, underutilizing heterogeneous clinical data, and suffering from the class imbalance inherent in medical datasets. To address these issues, we construct the first Multi-label Adverse Events dataset (MuAE) for intraoperative adverse events prediction, covering six critical events. Next, we propose a novel Transformerbased multi-label learning framework (IAENet) that combines an improved Time-Aware Feature-wise Linear Modulation (TAFiLM) module for static covariates and dynamic variables robust fusion and complex temporal dependencies modeling. Furthermore, we introduce a Label-Constrained Reweighting Loss (LCRLoss) with co-occurrence regularization to effectively mitigate intra-event imbalance and enforce structured consistency among frequently co-occurring events. Extensive experiments demonstrate that IAENet consistently outperforms strong baselines on 5, 10, and 15-minute early warning tasks, achieving improvements of +5.05%, +2.82%, and +7.57% on average F1 score. These results highlight the potential of IAENet for supporting intelligent intraoperative decision-making in clinical practice.

![the IAENet model structure](E:\Program Files (x86)\Git\mygit\IAENet-pytorch\pig\the IAENet model structure.png)

Figure 1: An overview of the IAENet framework for time-series in multi-label classification. Given an sample about vital sign series and static covariate in MuAE dataset $\mathbf{x} = \{\mathbf{x}_{d_0},..., \mathbf{x}_{d_{14}}, \mathbf{x}_{s_0},..., \mathbf{x}_{s_4} \}$, our goal is to predict whether the values in the following $\triangle$ time steps will be normal or abnormal $\mathbf{y} = \{\mathbf{y}_{1}, \mathbf{y}_{2}, ..., \mathbf{y}_{6} \}$. Firstly, the preprocessed dynamic variables and static covariates are respectively fused through the TAFiLM module for feature early fusion. Then, a Transformer encoder is employed to capture temporal correlations among multivariate variables. Finally, the model is trained using the proposed LCRLoss, which combines a batch-wise label frequency-weighted BCE loss with a co-occurrence constraint term based on label dependencies.

## Usage

1. Install Python 3.11. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

- MuAE dataset: the first multi-label dataset for early warning of intraoperative adverse events, comprising six adverse event types derived from the VitalDB dataset. It consists of data from 836 patiens across multiple hospitals, providing time-series data on various physiological variables such as heart rate, blood pressure, and oxygen saturation. You can access the dataset at VitalDB.

  
  
  The processed dataset, including the necessary preprocessing steps, will be made publicly available in the future to facilitate reproducibility and extend the impact of our work.


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# classification
bash ./scripts/classification/IAENet.sh
```

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wang2026early,
  title={Early Warning of Intraoperative Adverse Events via Transformer-Driven Multi-Label Learning},
  author={Wang, Xueyao and Cai, Xiuding and Shang, Honglin and Zhu, Yaoyao and Yao, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  series={40},
  number={31},
  pages={26615--26623},
  year={2026}
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## Acknowledgement

This library is constructed based on the following repos:

- Time-Series-Library: <https://github.com/thuml/Time-Series-Library>

We extend our sincere thanks for their excellent work and repositories!