# Weakly Supervised Polyp Frame Detection

> [**MICCAI'22 Early Accept**] [**Contrastive Transformer-based Multiple Instance Learning for Weakly Supervised Polyp Frame Detection**](https://arxiv.org/pdf/2203.12121.pdf)
>
> by [Yu Tian](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), [Fengbei Liu](https://fbladl.github.io/), Yuyuan Liu, Chong Wang, Yuanhong Chen, Johan W Verjans,  [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).
>

![image](https://user-images.githubusercontent.com/19222962/193112248-dbc4489b-4618-4c93-8b20-3ed671c0f4d3.png)

### Dataset

Please download the pre-processed i3d features of the dataset through this [link](https://drive.google.com/file/d/15VlBw0erQmP6HmYIGYaqyfCBMKzpA5R2/view?usp=drive_link) 
>
The original colonoscopy videos can be found in this [link](https://drive.google.com/file/d/1PTQdluckHm7aeVzgRHuoTTpz-Sum7xmF/view?usp=sharing).

### Training 
After downloading the dataset and extracting the I3D features using this [**repo**](https://github.com/Tushar-N/pytorch-resnet3d), simply run the following command: 
```shell
python main_transformer.py
```

### Inference 
For inference, after setting the path of the best checkpoint, then run the following command: 
```shell
python inference.py
```

### Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{tian2022contrastive,
  title={Contrastive Transformer-Based Multiple Instance Learning for Weakly Supervised Polyp Frame Detection},
  author={Tian, Yu and Pang, Guansong and Liu, Fengbei and Liu, Yuyuan and Wang, Chong and Chen, Yuanhong and Verjans, Johan and Carneiro,   Gustavo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={88--98},
  year={2022},
  organization={Springer}
}
```

If you use the dataset, please also consider citing the papers below:
```bibtex
@inproceedings{ma2021ldpolypvideo,
  title={Ldpolypvideo benchmark: A large-scale colonoscopy video dataset of diverse polyps},
  author={Ma, Yiting and Chen, Xuejin and Cheng, Kai and Li, Yang and Sun, Bin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={387--396},
  year={2021},
  organization={Springer}
}
```
```bibtex
@article{borgli2020hyperkvasir,
  title={HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy},
  author={Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, Sigrun L and Randel, Kristin Ranheim and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc Tien Dang and others},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={1--14},
  year={2020},
  publisher={Nature Publishing Group}
}
```
---
