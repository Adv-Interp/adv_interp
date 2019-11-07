# PyTorch Code for ["Adversarial Interpolation Training"](https://openreview.net/pdf?id=Syejj0NYvr)


## Usage
### Setup
The training environment can be setup as follows:
#### Create a virtual environment [optional, recommended]
```
virtualenv -p python3 /YOUR_PATH_TO_VENVS/adv_interp_env
source /YOUR_PATH_TO_VENVS/adv_interp_env/bin/activate
```
#### Clone repo and setup dependencies
```
git clone URL_TO_REPO
cd REPO
python setup.py install
```

### Train
Specify the path for saving checkpoints in ```adv_interp_train.sh```, and then run
```
sh ./adv_interp_train.sh
```

### Evaluate
Specify the corresponding model path and attack method in ```eval.sh``` and then run
```
sh ./eval.sh
```

### Evaluate Pretrained Model
A model trained on CIFAR10 using Adversarial Interpolation Training is [here](https://drive.google.com/file/d/1NWYmLAArzstzaknO1L0ZMni0_8UaSpTo/view?usp=sharing).
Download it to ```./pre_trained_adv_interp_models/``` and then run
```
sh ./eval_pretrain.sh
```