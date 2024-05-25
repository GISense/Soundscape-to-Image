<br />
<p align="center">
  <h2 align="center">From Hearing to Seeing: Linking Auditory and Visual Place Perceptions with Soundscape-to-Image Generative Artificial Intelligence</h2>
</p>

 <p align="center">
    <img src="./githubimage/Figures.png" alt="framework" >
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

People experience the world through multiple senses simultaneously, contributing to our sense of place. Prior quantitative geography studies have mostly emphasized human visual perceptions, neglecting human auditory perceptions at place due to the challenges in characterizing the acoustic environment vividly. Also, few studies have synthesized the two-dimensional (auditory and visual) perceptions in understanding human sense of place. To bridge these gaps, we propose a Soundscape-to-Image Stable Diffusion model, a generative Artificial Intelligence (AI) model supported by Large Language Models (LLMs), aiming to visualize soundscapes through the generation of street view images. By creating audio-image pairs, acoustic environments are first represented as high-dimensional semantic audio vectors. Our proposed Soundscape-to-Image Stable Diffusion model, which contains a Low-Resolution Diffusion Model and a Super-Resolution Diffusion Model, can then translate those semantic audio vectors into visual representations of place effectively. We evaluated our proposed model by using both machine-based and human-centered approaches and proved that the generated street view images align with our common perceptions, and accurately create several key street elements of the original soundscapes. It also demonstrates that soundscapes provide sufficient visual information places. This study stands at the forefront of the intersection between generative AI and human geography, demonstrating how human multi-sensory experiences can be linked. We aim to enrich geospatial data science and AI studies with human experiences. It has the potential to inform multiple domains such as human geography, environmental psychology, and urban design and planning, as well as advancing our knowledge of human-environment relationships. 

 <p align="center">
    <img src="./githubimage/structure.png" alt="framework" >
</p>

 ## Code Usage

Environment: Python 3.9 or newer  
See <em>requirements.txt</em>

we combine our audio encoder with imagen decoder.
recommand to first train the audio encoder 

1. install imagen_pytorch, change its diffusion decoder max_text_len into 343,change max_seq_len into 768
2. put you train audio set and image set in ./extract/audio and ./extract/image
3. train by train.py 
4. and put you test audio in testaudio
5. sample you result by sample.py



## Folder Structure 
The folders and files are organized as follows.   
put you traing audio and image in the extract folder, 
and after training, put you test audio in the testaudio folder

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