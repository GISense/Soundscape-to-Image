<br />
<p align="center">
  <h2 align="center">From Hearing to Seeing: Linking Auditory and Visual Place Perceptions with Soundscape-to-Image Generative Artificial Intelligence</h2>
</p>

 <p align="center">
    <img src="Figures.png" alt="framework" >
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

```
@article{ZHUANG2024102122,
title = {From hearing to seeing: Linking auditory and visual place perceptions with soundscape-to-image generative artificial intelligence},
journal = {Computers, Environment and Urban Systems},
volume = {110},
pages = {102122},
year = {2024},
issn = {0198-9715},
doi = {https://doi.org/10.1016/j.compenvurbsys.2024.102122},
url = {https://www.sciencedirect.com/science/article/pii/S0198971524000516},
author = {Yonggai Zhuang and Yuhao Kang and Teng Fei and Meng Bian and Yunyan Du},
keywords = {Soundscape, Street view images, Sense of place, Stable diffusion, Generative AI, LLMs},
}
```

<!-- ABOUT THE PROJECT -->
## About The Project

 Our proposed Soundscape-to-Image Stable Diffusion model, which contains a Low-Resolution Diffusion Model and a Super-Resolution Diffusion Model, can then translate those semantic audio vectors into visual representations of place effectively. We evaluated our proposed model by using both machine-based and human-centered approaches and proved that the generated street view images align with our common perceptions, and accurately create several key street elements of the original soundscapes. It also demonstrates that soundscapes provide sufficient visual information places. 

 <p align="center">
    <img src="structure.png" alt="framework" >
</p>

 ## Code Usage

Environment: Python 3.9 or newer  
See <em>requirements.txt</em>

we combine our audio encoder with imagen decoder.

recommand to first train the audio encoder 
after install imagen_pytorch, change its diffusion decoder max_text_len into 343,change max_seq_len into 768
<p align="center">
    <img src="change1.png" alt="framework" >
</p>
<p align="center">
    <img src="change2.png" alt="framework" >
</p>

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
|-- testaudio
|-- testresult
|-- train.py
|-- sample.py
```

## Contact

Yonggai Zhuang: start128@163.com