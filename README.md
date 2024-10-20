![Banner image!](/images/SlicerTutorialBanner.jpeg)


# Slicer_scripting_tutorial
A repository built for the slicer tutorial run by Marcus Milantoni and Edward Wang. This tutorial gives a demonstration of how to use python with scripting & using the [slicerutility package](https://github.com/Marcus-Milantoni/Slicer_Utility) created by the same authors.

## Table of Contents
- [License](#License)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Setup](#Setup)
- [Tutorial](#Description)
- [Example patients](#Examples)
- [Acknowledgements](#Acknowledgements)

## License
The following repository is under MIT license. For more information, visit [LICENSE](/LICENSE).

## Requirements
- numpy
- matplotlib

## Installation
**!!!! Please install this package in a Slicer python environment through 3D Slicer's python terminal !!!!**

### Install slicerutil
``` python
import pip

pip.main(['install', 'slicerutil'])
```

### Import the package
After installing the package use the following to import:
``` python
import slicerutil as su
```

or if you want to use the package in function mode (without oop):
``` python
import slicerutil.basic_functions as bf
```

## Setup

### Downloading the 3D slicer application & supporting software

Please follow the steps provided bellow:
1. Visit [slicer](https://download.slicer.org) to download the application.
2. Visit [anaconda](https://www.anaconda.com/download) to download python3/ jupyter notebook.
3. Visit [Visual Studio Code](https://code.visualstudio.com/Download) to download the source-code editor (optional).
4. From the Extensions Manager widget, download the SlicerJupyter, MeshToLabelMap, PETDICOMExtension (if working with PET DICOMS), SlicerRT (if working with radiotherapy data).
    - The Slicer application needs to restart to install the extensions.

### Set up the SlicerJupyter

1. Using the search widget in Slicer, open the SlicerJupyter extension by searching for JupyterKernel.


    ![The Slicer application on the SlicerJupyter Modules!](/images/SlicerJupyterScreenCapture.png)
2. Click the "Start Jupyter Server" button. A JupyterLab notebook will open when the setup is complete.
3. Click the "Jupyter server in external Python environment" and copy the command to clipboard.
4. Open the anaconda prompt (Terminal if on mac) and paste the command.
5. (Optional) open an external environment (Visual Studio Code) and select the Slicer kernel!

### Install python packages

1. Open the Python console by clicking the python logo on the widget bar.
2. Import pip with
    ~~~ python
    import pip
    ~~~
3. install the packages with
    ~~~ python
    pip.main(['install', 'slicerutil'])
    ~~~

## Tutorial
#### [cervical_cancer_tutorial](Tutorials/cervical_cancer_tutorial.ipynb)
This tutorial was created to demonstrate the basic_functions module on the cervical cancer example patient. This tutorial runs through most of the functions that are used in Slicer Python image processing. 

## Examples
A list of all the the [example patients provided](example_patients).
#### [Brain Pre-op](example_patients/Brain_resection)
This dataset is provided by the Brain Resection Multimodal Imaging Database (ReMIND) on the Cancer Imaging Archive. The data includes a patient that was surgically treated with image-guided tumor resection between 2018 and 2022. The preoperative T1 & T2 MRIs are used as well as the segmentation of the tumor. More information can be found [here](https://www.cancerimagingarchive.net/collection/remind/).

Juvekar, P., Dorent, R., Kögl, F., Torio, E., Barr, C., Rigolo, L., Galvin, C., Jowkar, N., Kazi, A., Haouchine, N., Cheema, H., Navab, N., Pieper, S., Wells, W. M., Bi, W. L., Golby, A., Frisken, S., & Kapur, T. (2023). The Brain Resection Multimodal Imaging Database (ReMIND) (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/3RAG-D070

#### [Cervical Cancer](example_patients/cervical_cancer)
This dataset is provided by the CC-TUMOR-HETEROGENITY collection on the Cancer Imaging Archive. The dataset includes a PET/CT DICOM of a patient with cervical cancer, and structure RTstructs created using TotalSegmentator. More information can be found [here](https://www.cancerimagingarchive.net/collection/cc-tumor-heterogeneity/).
	
Mayr, N., Yuh, W. T. C., Bowen, S., Harkenrider, M., Knopp, M. V., Lee, E. Y.-P., Leung, E., Lo, S. S., Small Jr., W., & Wolfson, A. H. (2023). Cervical Cancer – Tumor Heterogeneity: Serial Functional and Molecular Imaging Across the Radiation Therapy Course in Advanced Cervical Cancer (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/ERZ5-QZ59

## Acknowledgements
The slicer scripting tutorial would not be possible without the following open source software:
- [Slicer](https://github.com/Slicer/Slicer)
- [Numpy](https://github.com/numpy/numpy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
