# alquimodelia
Tesselo's Model Alquemy

## About
Package to enable easy usage of Tesselo's common model

## Features: 
- [x] U-Net models        
    - [x] 3D and 2D architetures
    - [] Make variable number of layers
- [] Pixel Model:
    - [] Make as a class
- [] Models:
    - [] ResNet
    - [] LSTM
- [] Test Models:
    - [] Working samples to test models
    - [] Visualize results
    - [] Visualize inner layers

## Installation:
```bash
pip install git+https://github.com/tesselo/alquimodelia
```
or
```bash
cd path/to/repo
pip install .
```

## Usage examples:

- UNet implementations:  
    - [3D_UNET](#3D_UNET)  

<br>

### 3D_UNET


```python
from alquimodelia import alquimodelia as alqm

UNET_3D_model = alqm.ModelMagia('3D_UNET',
    timesteps=12,
    width=512,
    height=512,
    padding=0,
    num_bands=10,
    num_classes=3,
    activation_final=None)
UNET_3D_model = UNET_3D_model.unet()
```

[[back to usage examples]](#usage-examples)

<br>