<br />
<p align="center">
  <h2 align="center">From Hearing to Seeing: Linking Auditory and Visual Place Perceptions with Soundscape-to-Image Generative Artificial Intelligence</h2>
</p>

## Table of Contents

* [Citation](#citation)
* [About the Project](#about-the-project)
* [Code Usage](#code-usage)
* [Folder Structure](#folder-structure)
* [Contact](#contact)

<!-- Citation -->
## Citation
If you use this algorithm in your research or applications, please cite this source:


<!-- ABOUT THE PROJECT -->
## About The Project

 Our proposed Soundscape-to-Image Stable Diffusion model, which contains a Low-Resolution Diffusion Model and a Super-Resolution Diffusion Model, can then translate those semantic audio vectors into visual representations of place effectively. We evaluated our proposed model by using both machine-based and human-centered approaches and proved that the generated street view images align with our common perceptions, and accurately create several key street elements of the original soundscapes. It also demonstrates that soundscapes provide sufficient visual information places. 

 ## Code Usage

Environment: Python 3.9 or newer  
See <em>requirements.txt</em>

after install imagen_pytorch, change its diffusion decoder into 343   

put you train audio set and image set in ./extract/audio and ./extract/image
train them by train.py 
and put you test audio in testaudio
sample you result by sample.py



## Folder Structure 
The folders and files are organized as follows.   
```
project
|-- extract
|   |-- audio
|   |-- image
|-- torchvggish
|-- torchvggish
|-- torchvggish
|-- train.py
|-- sample.ipynb
```

## Contact

Yonggai Zhuang: start128@163.com