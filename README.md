# Slicer_tutorial
A repository built for the slicer tutorial run by Marcus Milantoni and Edward Wang.


![Banner image!](/images/SlicerTutorialBanner.jpeg)

## Table of Contents
- [Requirements](#Requirements)
- [Setup](#Setup)
- [Code Description](#Description)
- [Example patients](#Examples)

## Requirements
- numpy
- matplotlib

The full list can be found in [requirements.txt](/requirements.txt)

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
2. Import pip with ''import pip''.
3. install the packages with ''pip.main(['install', 'my_package'])''.
    - Find the requirements in the [requirements.txt](/requirements.txt) file.


## Examples
A list of all the the [example patients provided](example_patients).
#### [Brain Pre-op](example_patients/Brain_resection)
This dataset is provided by the Brain Resection Multimodal Imaging Database (ReMIND) on the Cancer Imaging Archive. The data includes a patient that was surgically treated with image-guided tumor resection between 2018 and 2022. The preoperative T1 & T2 MRIs are used as well as the segmentation of the tumor. More information can be found [here](https://www.cancerimagingarchive.net/collection/remind/).

Juvekar, P., Dorent, R., Kögl, F., Torio, E., Barr, C., Rigolo, L., Galvin, C., Jowkar, N., Kazi, A., Haouchine, N., Cheema, H., Navab, N., Pieper, S., Wells, W. M., Bi, W. L., Golby, A., Frisken, S., & Kapur, T. (2023). The Brain Resection Multimodal Imaging Database (ReMIND) (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/3RAG-D070

#### [CT Phantom](example_patients/CT_Phantom)
This dataset is provided by CT-Phantom4Radiomics on the Cancer Imaging Archive. The dataset includes an abdominal CT phantom printed based on real patient data, as well as segmentations. More information can be found [here](https://www.cancerimagingarchive.net/collection/ct-phantom4radiomics/)

Schaer, R., Bach, M., Obmann, M., Flouris, K., Konukoglu, E., Stieltjes, B., Müller, H., Aberle, C., Jimenez del Toro, O. A., & Depeursinge, A. (2023). Task-Based Anthropomorphic CT Phantom for Radiomics Stability and Discriminatory Power Analyses (CT-Phantom4Radiomics) [Data set]. https://doi.org/10.7937/a1v1-rc66
