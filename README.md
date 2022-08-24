# alquimodelia
Tesselo's Model Alquemy

## About
Package to enable easy usage of Tesselo's common model

## Features: 
- [x] U-Net models        
    - [x] 3D and 2D architetures
    - [x] Make variable number of layers
- [x] Pixel Model:
    - [x] Make as a class
- [ ] Models:
    - [x] ResNet
    - [x] LSTM
    - [ ] Outside packages
- [ ] Test Models:
    - [ ] Working samples to test models
    - [ ] Visualize results
    - [ ] Visualize inner layers

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
    - [3D_RESNET](#3D_RESNET)  
    - [1D_LSTM](#LSTM)


<br>

### 3D_UNET


```python
from alquimodelia.unet_arch import UNet2D, UNet3D

UNet3D_model = UNet3D(
        n_filters=16,
        number_of_conv_layers=None,
        kernel_size=3,
        batchnorm=True,
        padding_style="same",
        activation="relu",
        kernel_initializer="he_normal",
        timesteps=12,
        width=600,
        height=600,
        padding=None,
        num_bands=10,
        num_classes=4,
        activation_final=None,
        data_format="channels_last",
)
```


<br>

### 3D_RESNET


```python
from alquimodelia.resnet_arch import ResNet2D, ResNet3D

ResNet3D_model = ResNet3D(
        n_filters=16,
        timesteps=12,
        width=600,
        height=600,
        num_bands=10,
        num_classes=4,
        activation_final=None,
        data_format="channels_last",
)
```

<br>

### LSTM

```python
from alquimodelia.rnn_lstm_arch import RnnLSTM

RnnLSTM_model = RnnLSTM(
        timesteps=48,
        num_bands=10,
        num_classes=12,
        activation_final="softmax",
        data_format="channels_last",
        lstm_units=(120, 80),
)
```


[[back to usage examples]](#usage-examples)

<br>