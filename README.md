# Pretrained Audio Neural Network Fine-tuning Comparisons

Based on the research by Qiuqiang Kong: [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://ieeexplore.ieee.org/abstract/document/9229505).

Our work investigates the potential benefits of using a Residual Neural Network rather than the suggested Convolutional architecture, comparing performance after fine-tuning on the GTZAN dataset. Our research into this can be seen [here](https://expo.baulab.info/2024-Spring/joshiarnav/).

Setup:
```
$ git clone https://github.com/samphillippo/PANNInvestigation.git
$ cd PANNInvestigation
$ pip install torch matplotlib librosa torchlibrosa
```

- Download either the "Cnn14_mAP=0.431.pth" or "ResNet38_mAP=0.434.pth" model [here](https://zenodo.org/records/3987831).

- Download the GTZAN dataset [here](https://huggingface.co/datasets/marsyas/gtzan).

- Finetune your model with `python FineTuneGTZAN.py <workspace_path> <GTZAN_dataset_name> <pretrained_model_name>`

- Measure the confusion of your tuned model with `python MeasureConfusion.py <workspace_path> <GTZAN_dataset_name> <model_name>`
