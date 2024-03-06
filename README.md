<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Critical area prediction in train stations with DEtection TRansformer (DETR)</h3>

  <p align="center">
    Chair of Modelling and Simulation - Technical University of Munich.
    <br />
    Patrick Berggold
    <br />
    <a href=#docs><strong>Explore the docs »</strong></a>
    <!-- <strong>[Explore the docs »](#documentation)</strong> -->
    <br />
    <br />
    <a href="mailto:patrick.berggold@tum.de">Report Bug</a>
    ·
    <a href="mailto:patrick.berggold@tum.de">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#docs">Documentation</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

Project title: Critical area prediction in train stations with DEtection TRansformer (DETR)

In this work, we propose a deep learning-based approach to realistically and instantly predict critical areas in the design of train station platforms. These critical areas, represented by rectangles in the platform layout, 
may form during an evacuation process, when many passengers try to reach the exits all at once, thus creating dangerous overcrowding situations on the platform that may even turn into stampedes and mass panic. 
Therefore, we aim at supporting particularly the early stages of the train station design project by proposing a methodology that utilizes an object detector (namely an adapted version of DETR) to interactively assess platform floorplans 
whenever a new design variant emerges. This is conventionally done with pedestrian simulators which entail runtimes of minutes or even hours (depending on the building complexity), as well as several manual export and conversion steps. 
Consequently, it is simply unfeasible to interrogate every single building design variant that emerges in the design and planning process. 

In contrast, the neural network - trained on a synthetic dataset - can deliver predictions in real-time and interactively if connected to the BIM model as depicted in Figure 1. However, to reproduce simulation results, it is necessary to not only use the 
floorplans as inputs to the neural network, but also consider simulator information (e.g. number of agents, number of stairs and escalators, agent velocity distribution, etc.) as essential components. In this repository, we showcase
some potential input options to emulate simulator results, supporting the design and planning of train station platforms.

![Figure 1: Methodology of our approach.](https://github.com/patrickberggold/CriticalAreaDetection/tree/main/pics/methodology.png)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Start by confirming Git is installed on your computer

```sh
git --version
```

You should get an output similar to the one below, otherwise download and install [Git](https://git-scm.com/downloads).

```sh
git version 1.2.3
```

Next clone the repository:

```sh
git clone https://github.com/patrickberggold/CriticalAreaDetection
```

### Folder Strcture

Once you clone the repository, make sure the folder structure matches the directory tree shown below.

📦CriticalAreaDetection  
┣ 📂checkpoints  
┣ 📂ExampleDataset
┃ ┣ 📂inputs
┃ ┣ 📂targets
┣ 📂models
┃ ┣ 📜Detr_custom.py
┣ 📂ExampleDataset
┃ ┣ 📂inputs
┃ ┣ 📂targets
┣ 📜helper.py
┣ 📜main.py
┣ 📜metrics_calc.py
┣ 📜ObjDetDatamodule.py
┣ 📜ObjDetDataset.py
┣ 📜ObjDetModule.py
┣ 📜README.md
┗ 📜requirements.txt


## Initial Setup

#### Python version

<!-- 3.10.13 -->
Before using this script, make sure that you download and install Python 3.10 [here](https://www.python.org/downloads/). Afterwards, you need to create a virtual Python environment using the command below.

```
python3 -m venv /path/and/name/of/your/virtual/environment/
```

You can find some useful information on virtual environments [here](https://docs.python.org/3/library/venv.html#creating-virtual-environments).

**Warning**: This code does not support Python 2.0. If you are using an older version of Revit, the default may be python 2.0. Make sure you select Python 3.0 before running the python node.

After activating your virtual environment, install all the required python libraries by running the following command (shown for both Pip and Anaconda).

- Pip

```
pip install -r requirements.txt
```

- Anaconda

```
conda install --file requirements.txt
```

<!-- USAGE EXAMPLES -->

## Usage

#### Synthetic Dataset
We provide an example dataset of fifty samples to showcase the synthetic platform dataset, comprising the colored floorplan images of the platforms (exported from a parametric BIM model) and the resulting critical 
areas as targets. The filenames of the input images describe the platform and simulation setup, containing the following information (in order):

Number of tracks (**T**), number of agents (**A**), obstacle presence (**C**) (0=false, 2=true), number of central escalators (**E**), width of central stairs (**S**) (in meters), 
orientation of central ascent units (**O**) (0=outwards, 1=inwards), number of escalators on the sides (**ES**) and width of the stairs on the sides (**SS**) (in meters). Additionally,
**VC** and **VS** denote the general presence of any vertical ascent units (meaning escalators and stairs) either in the center, or sides, or both.


#### Neural Network Architecture
All possible configurations are incorporated in the main.py file, including training, inference, hyperparameter choice, etc. Therfore, to choose a specific configuration, it must be specified in 
CONFIG and TRAIN_CONFIG dictionaries. Specifically, the adaption of the DETR is based on different input options to the neural network, where each option provides the (augmented) input image of
the floorplan as input. Additionally, simulator information is provided as well in the following way:

* The **vanilla** option performs a forward pass similar to the original DETR implementation.
* The **vanilla_imgAugm** option varies the brightness of each color in the input image depending on the number of agents.
* Options **before_encoder** and **after_encoder** encode the number of agents as a specified number of learnable embeddings, and provide those embeddings before or after the encoder, respectively, 
during the forward pass.
* Options **before_encoder+** and **after_encoder+** encode the number of agents, the number of vertical ascent units on the sides and in the center and boolean obstacle presence as a specified number of learnable embeddings,
and provide those embeddings before or after the encoder, respectively, during the forward pass.

Natually, some of these options are quite use case-specific, but we aim at underscoring the vast possibilities and versatility of adding supplementary information to enhance the predictive capabilities of the network.

<!-- CONTACT -->

## Contact

If you have any questions with regards to our research or the usage of this project, please don't hesitate to contact me via e-mail.

Patrick Berggold - patrick.berggold@tum.de

Project Link: [https://github.com/patrickberggold/PedSimAutomation](https://github.com/patrickberggold/PedSimAutomation)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
