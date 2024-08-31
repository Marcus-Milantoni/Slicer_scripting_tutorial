
# Setup

## Downloading the 3D slicer application & supporting software

Please follow the steps provided bellow:
1. Visit [slicer](https://download.slicer.org) to download the application.
2. Visit [anaconda](https://www.anaconda.com/download) to download python3/ jupyter notebook.
3. Visit [Visual Studio Code](https://code.visualstudio.com/Download) to download the source-code editor (optional).
4. From the Extensions Manager widget, download the SlicerJupyter, MeshToLabelMap, PETDICOMExtension (if working with PET DICOMS), SlicerRT (if working with radiotherapy data).
    - The Slicer application needs to restart to install the extensions.

## Set up the SlicerJupyter

1. Using the search widget in Slicer, open the SlicerJupyter extension by searching for JupyterKernel.
2. 
![The Slicer application on the SlicerJupyter Modules!](/images/SlicerJupyterScreenCapture.png)
3. Click the "Start Jupyter Server" button. A JupyterLab notebook will open when the setup is complete.
4. Click the "Jupyter server in external Python environment" and copy the command to clipboard.
5. Open the anaconda prompt (Terminal if on mac) and paste the command.
6. (Optional) open an external environment (Visual Studio Code) and select the Slicer kernel!

## Install python packages

1. Open the Python console by clicking the python logo on the widget bar.
2. Import pip with ~~~import pip~~~.
3. install the packages with ~~~pip.main(['install', 'my_package'])~~~.
    - Find the requirements in the [requirements.txt](/requirements.txt) file.
