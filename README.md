# VinciCoder: Unifying Multimodal Code Generation via Coarse-to-fine Visual Reinforcement Learning




## Training 
Our SFT stage utilize ms-swift, please follow the official document for training.

Our RL based on Easyr1, please first modify the configurations in ```./examples/qwen3vl_8b_vincicder.sh``` and run the following scripts
```
bash ./examples/qwen3vl_8b_vincicder.sh
```




## Acknowledgement
The training frameworks are based on the [ms-swift](https://github.com/modelscope/ms-swift) and [EasyR1](https://github.com/hiyouga/EasyR1). Thanks for these great works and open sourcing!