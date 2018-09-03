<br><br><br>

# Interactive Deep Colorization in PyTorch

This is our PyTorch reimplementation for interactive image colorization. The code was written by [Richard Zhang](https://github.com/richzhang) and [Jun-Yan Zhu](https://github.com/junyanz).

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch 0.4 and dependencies from http://pytorch.org

- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Alternatively, all dependencies can be installed by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/richzhang/colorization-pytorch
cd colorization-pytorch
```

### Interactive colorization train/test
- Dataset preparation: Download the ILSVRC 2012 dataset and run the following script to prepare data
```python make_ilsvrc_dataset.py --in_path /PATH/TO/ILSVRC12```

- Train a model:
```bash ./scripts/train_siggraph.sh```

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.`

- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
```
The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

## Acknowledgments
This code borrows from the [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
