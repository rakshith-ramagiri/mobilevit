# MobileViT
A PyTorch implementation of MobileViT as presented in the paper ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/abs/2110.02178).

## Training
Check whether all dependencies are satisfied. This :point_down: will run a random `torch tensor` through the three MobileViT architectures and print their sizees (# of model parameters).

```python
python3 train.py check
```


## Citation
```
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

## Credits
Code adopted from [MobileViT-PyTorch](https://github.com/chinhsuanwu/mobilevit-pytorch).
