import slicer
import numpy as np
from typing import Union
from utils import check_type, log_and_raise, PresetColormaps, PetColormaps
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ImageVolume:
    """
    Class to handle the volume nodes in 3D Slicer.
    
    Attributes
    ----------
    volumeNode : slicer.vtkMRMLScalarVolumeNode
        The volume node in Slicer.
    name : str
        The name of the volume node.
    NumPyArray : np.ndarray
        The NumPy array associated with the volume node.
    shape : tuple
        The shape of the NumPy array.
        
    Methods
    -------
    description() -> str
        Get the description of the volume node.
    update_slicer_view() -> None
        Update the volume node in Slicer with the NumPy array.
    make_copy(newName: str) -> slicer.vtkMRMLScalarVolumeNode
        Make a copy of the volume node.
    set_as_foreground(backgroundNode: slicer.vtkMRMLScalarVolumeNode, opacity: float = 0.75) -> None
        Set the volume node as the foreground image.
    set_cmap(cmap: Union[PresetColormaps, PetColormaps]) -> None
        Set the colormap of the volume node.
    edit_name(newName: str) -> None
        Edit the name of the volume node.
    check_segment_shape_match(segmentationArray: np.ndarray) -> bool
        Check if the shape of the segmentation array matches the volume node.   
    crop_volume_from_segment(segmentationArray: np.ndarray, updateScene: bool = True) -> None
        Crop the volume node from the segmentation array.
    resample_scalar_volume_BRAINS(referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, nodeName: str, interpolatorType: str = 'NearestNeighbor') -> slicer.vtkMRMLScalarVolumeNode
        Resample a scalar volume node based on a reference volume node using the BRAINSResample module.
    quick_visualize(cmap: str ='gray', indices=None) -> None
        Visualize axial, coronal, and sagittal slices of a 3D image array.
    zero_image_where_mask_present(binaryMaskArray: np.ndarray, updateScene: bool = True) -> None
        Zero the image where the mask is present.
    save_volume(outputDir : str, fileType : str = 'nrrd', additionalSaveInfo: str = None) -> None
        Save the volume node.
    """
    preset_colormaps = ['CT_BONE', 'CT_AIR', 'CT_BRAIN', 'CT_ABDOMEN', 'CT_LUNG', 'PET', 'DTI']
    pet_colormaps = ['PET-Heat','PET-Rainbow2']


    def __init__(self, volumeNode: slicer.vtkMRMLScalarVolumeNode, name: str = None) -> None:
        self.volumeNode = volumeNode
        if name == None:
            self.name = self.volumeNode.GetName()
        else:
            self.name = name
            self.volumeNode.SetName(name)
        self.NumPyArray = slicer.util.arrayFromVolume(self.volumeNode)
        self.shape = self.NumPyArray.shape
        self.maxValue = np.max(self.NumPyArray)
        self.minValue = np.min(self.NumPyArray)
        self.spacing = self.volumeNode.GetSpacing()
        self.origin = self.volumeNode.GetOrigin()
        self.ID = self.volumeNode.GetID()
        self.getName = self.volumeNode.GetName()


    def description(self) -> str:
        """
        Get the description of the volume node.
        """
        return f"Name: {self.name}, Shape: {self.shape}"


    def update_slicer_view(self) -> None:
        """
        Update the volume node in Slicer with the NumPy array.
        """
        slicer.util.updateVolumeFromArray(self.volumeNode, self.NumPyArray)


    def make_copy(self, newName:str) -> slicer.vtkMRMLScalarVolumeNode:
        """
        Make a copy of the volume node.
        
        Parameters
        ----------
        newName : str
            The name of the new volume node.
        """
        check_type(newName, str, 'newName')    
        try:
            logger.info(f"Making a copy of the volume node with name {newName}")
            copiedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', newName)
            copiedNode.Copy(self.volumeNode)
            return copiedNode
        except Exception as e:
            log_and_raise(logger, "An error occurred in makeCopy", type(e))


    def set_as_foreground(self, backgroundNode: slicer.vtkMRMLScalarVolumeNode, opacity=0.75) -> None:
        """
        Set the volume node as the foreground image.
        
        Parameters
        ----------
        backgroundNode : slicer.vtkMRMLScalarVolumeNode
            The background volume node.
        opacity : float
            The opacity of the volume node.
        """
        check_type(backgroundNode, slicer.vtkMRMLScalarVolumeNode, 'backgroundNode')
        if not 0 <= opacity <= 1:
            raise ValueError("The opacity parameter must be between 0 and 1.")
        try:
            logger.info(f"Setting the volume node as the foreground image with opacity {opacity}")
            slicer.util.setSliceViewerLayers(background=backgroundNode, foreground=self.volumeNode, foregroundOpacity=opacity)
        except Exception as e:
            log_and_raise(logger, "An error occurred in setAsForeground", type(e))


    def set_cmap(self, cmap: Union[PresetColormaps, PetColormaps]) -> None:
        """
        Set the colormap of the volume node.
        
        Parameters
        ----------
        cmap : Union[PresetColormaps, PetColormaps]
            The colormap to set.
        """
        if not isinstance(cmap, PresetColormaps) and not isinstance(cmap, PetColormaps):
            raise ValueError("The cmap parameter must be a valid colormap.")
        try:
            logger.info(f"Setting the colormap to {cmap.value}")
            if isinstance(cmap, PresetColormaps):
                slicer.modules.volumes.logic().ApplyVolumeDisplayPreset(self.volumeNode.GetVolumeDisplayNode(), cmap.value)
            else:
                ColorNode = slicer.mrmlScene.GetFirstNodeByName(cmap.value)
                self.volumeNode.GetVolumeDisplayNode().SetAndObserveColorNodeID(ColorNode.GetID())
                self.volume_node.GetVolumeDisplayNode().AutoWindowLevelOn()
        except Exception as e:
            log_and_raise(logger, "An error occurred in setCmap", type(e))


    def edit_name(self, newName: str) -> None:
        """
        Edit the name of the volume node.

        Parameters
        ----------
        newName : str
            The new name of the volume node.
        """
        check_type(newName, str, 'newName')
        try:
            logger.info(f"Editing the name of the volume node to {newName}")
            self.volumeNode.SetName(newName)
            self.name = newName
        except Exception as e:
            log_and_raise(logger, "An error occurred in editName", type(e))


    def check_segment_shape_match(self, segmentationArray: np.ndarray) -> bool:
        """
        Check if the shape of the segmentation array matches the volume node.
        
        Parameters
        ----------
        segmentationArray : np.ndarray
            The segmentation array to check.
        """
        check_type(segmentationArray, np.ndarray, 'segmentationArray')
        try:
            logger.info(f"Checking if the shape of the segmentation array matches the volume node")
            if segmentationArray.shape == self.shape:
                return True
            else:
                return False
        except Exception as e:
            log_and_raise(logger, "An error occurred in checkSegmentShapeMatch", type(e))


    def crop_volume_from_segment(self, segmentationArray: np.ndarray, updateScene: bool = True) -> None:
        """
        Crop the volume node from the segmentation array.
        
        Parameters
        ----------
        segmentationArray : np.ndarray
            The segmentation array to crop the volume node from.
        updateScene : bool
            Whether to update the Slicer scene.
        """
        check_type(segmentationArray, np.ndarray, 'segmentationArray')
        if self.check_segment_shape_match(segmentationArray):
            try:
                logger.info(f"Cropping the volume node from the segmentation array")
                self.NumPyArray = self.NumPyArray * segmentationArray 
                if updateScene:
                    self.update_slicer_view()
            except Exception as e:
                log_and_raise(logger, "An error occurred in cropVolumeFromSegment", type(e))    
        else:
            logger.warning("The shape of the segmentation array does not match the volume node.")
    

    def resample_scalar_volume_BRAINS(self, referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, nodeName: str, interpolatorType: str = 'NearestNeighbor') -> slicer.vtkMRMLScalarVolumeNode:
        """
        Resamples a scalar volume node based on a reference volume node using the BRAINSResample module.

        This function creates a new scalar volume node in the Slicer scene, resamples the input volume node
        to match the geometry (e.g., dimensions, voxel size, orientation) of the reference volume node, and
        assigns the specified node name to the newly created volume node if possible. If the specified node
        name is already in use or not provided, a default name is assigned by Slicer.

        Parameters
        ----------
        referenceVolumeNode : slicer.vtkMRMLScalarVolumeNode
            The reference volume node to resample the input volume node to.
        nodeName : str
            The name to assign to the resampled volume node.
        interpolatorType : str, optional
            The interpolator type to use for resampling. Default is 'NearestNeighbor'.
        """
        check_type(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode, 'referenceVolumeNode')
        check_type(nodeName, str, 'nodeName')
        check_type(interpolatorType, str, 'interpolatorType')
        if not interpolatorType in ['NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline', 'WindowedSinc']:
            raise ValueError("The interpolatorType parameter must be one of 'NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline', or 'WindowedSinc'.")
        parameters = {}
        parameters["inputVolume"] = self.volumeNode
        try:
            resampledNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', nodeName)
        except:
            resampledNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        parameters["outputVolume"] = resampledNode
        parameters["referenceVolume"] = referenceVolumeNode
        parameters["interpolatorType"] = interpolatorType
        # Execute
        resampler = slicer.modules.brainsresample
        cliNode = slicer.cli.runSync(resampler, None, parameters)  
        # Process results
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)  
        # success
        slicer.mrmlScene.RemoveNode(cliNode)
        return resampledNode


    def quick_visualize(self, cmap: str ='gray', indices=None) -> None:
        """
        Visualizes axial, coronal, and sagittal slices of a 3D image array.

        Parameters
        ----------
        cmap : str, optional
            The colormap to use. Default is 'gray'.
        indices : dict, optional
            The indices of the slices to visualize. Default is None.
        """
        check_type(cmap, str, 'cmap')
        if indices is not None:
            if not isinstance(indices, dict):
                raise TypeError("indices must be a dictionary.")
            if not all(key in indices for key in ['axial', 'coronal', 'sagittal']):
                raise ValueError("indices must contain 'axial', 'coronal', and 'sagittal' keys.")
            if not all(isinstance(indices[key], list) and len(indices[key]) == 3 for key in ['axial', 'coronal', 'sagittal']):
                raise ValueError("Each key in indices must have a list of 3 integers.")
        # Automatically select indices if not provided
        if indices is None:
            indices = {
                'axial': [self.shape[0] // 4, self.shape[0] // 2, 3 * self.shape[0] // 4],
                'coronal': [self.shape[1] // 4, self.shape[1] // 2, 3 * self.shape[1] // 4],
                'sagittal': [self.shape[2] // 4, self.shape[2] // 2, 3 * self.shape[2] // 4],
            }
        plt.figure(figsize=(10, 10))
        # Axial slices
        for i, idx in enumerate(indices['axial'], 1):
            plt.subplot(3, 3, i)
            plt.imshow(self.NumPyArray[idx, :, :], cmap=cmap)
            plt.title(f"Axial slice {idx}")
        # Coronal slices
        for i, idx in enumerate(indices['coronal'], 4):
            plt.subplot(3, 3, i)
            plt.imshow(self.NumPyArray[:, idx, :], cmap=cmap)
            plt.title(f"Coronal slice {idx}")
        # Sagittal slices
        for i, idx in enumerate(indices['sagittal'], 7):
            plt.subplot(3, 3, i)
            plt.imshow(self.NumPyArray[:, :, idx], cmap=cmap)
            plt.title(f"Sagittal slice {idx}")
        plt.tight_layout()
        plt.show()   


    def zero_image_where_mask_present(self, binaryMaskArray: np.ndarray, updateScene: bool = True) -> None:
        """
        Zero the image where the mask is present.

        Parameters
        ----------
        binaryMaskArray : np.ndarray
            The binary mask array.
        updateScene : bool, optional
            Whether to update the Slicer scene. Default is True.
        """
        if not isinstance(binaryMaskArray, np.ndarray):
            raise ValueError("Both imageArray and binaryMaskArray must be NumPy arrays.")
        self.NumPyArray = np.where(binaryMaskArray == 1, 0, self.NumPyArray)
        if updateScene:
            self.update_slicer_view()

    
    def save_volume(self, outputDir : str, fileType : str = 'nrrd', additionalSaveInfo: str = None) -> None:
        """
        Save the volume node.

        Parameters
        ----------
        outputDir : str
            The directory to save the volume node to.
        fileType : str, optional
            The file type to save the volume node as. Default is 'nrrd'.
        additionalSaveInfo : str, optional
            Additional information to add to the file name. Default is None.
        """
        check_type(outputDir, str, 'outputDir')
        check_type(fileType, str, 'fileType')
        if not fileType in [".nii", ".nrrd"]:
            raise ValueError("The outputFileType parameter must be either '.nii' or '.nrrd'.")
        if not isinstance(additionalSaveInfo, None) or not isinstance(additionalSaveInfo, str):
            raise ValueError("The additionalSaveInfo parameter must be a string or None.")
        
        if not os.path.exists(outputDir):
            logger.debug(f"Creating the output folder {outputDir}")
            os.makedirs(outputDir)
        
        if additionalSaveInfo == None:
            slicer.util.saveNode(self.volumeNode, os.path.join(outputDir, f"{self.name}.{fileType}"))
        else:
            slicer.util.saveNode(self.volumeNode, os.path.join(outputDir, f"{additionalSaveInfo}.{fileType}"))

