import slicer, vtk
from DICOMLib import DICOMUtils
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import ScreenCapture
from .util import check_type, log_and_raise
from enum import Enum
from typing import Union
import Segment


# Setup the logger (can customize)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class TempNodeManager:
    def __init__(self, node_class, node_name) -> None:
        self.scene = slicer.mrmlScene
        self.node_class = node_class
        self.node_name = node_name

    def __enter__(self) -> slicer.vtkMRMLNode:
        self.node = self.scene.AddNewNodeByClass(self.node_class, self.node_name)
        return self.node

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.scene.RemoveNode(self.node)



class PresetColormaps(Enum):
    CT_BONE = 'CT_BONE'
    CT_AIR = 'CT_AIR'
    CT_BRAIN = 'CT_BRAIN'
    CT_ABDOMEN = 'CT_ABDOMEN'
    CT_LUNG = 'CT_LUNG'
    PET = 'PET'
    DTI = 'DTI'



class PetColormaps(Enum):
    PET_HEAT = 'PET-Heat'
    PET_RAINBOW2 = 'PET-Rainbow2'
    LABELS = 'Labels'
    FULL_RAINBOW = 'Full-Rainbow'
    GREY = 'Grey'
    RAINBOW = 'Rainbow'
    INVERTED_GREY = 'Inverted-Grey'
    FMRI = 'fMRI'



class ImageVolume:
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
        return f"Name: {self.name}, Shape: {self.shape}"


    def update_slicer_view(self) -> None:
        slicer.util.updateVolumeFromArray(self.volumeNode, self.NumPyArray)


    def make_copy(self, newName:str) -> slicer.vtkMRMLScalarVolumeNode:
        check_type(newName, str, 'newName')    
        try:
            logger.info(f"Making a copy of the volume node with name {newName}")
            copiedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', newName)
            copiedNode.Copy(self.volumeNode)
            return copiedNode
        except Exception as e:
            log_and_raise(logger, "An error occurred in makeCopy", type(e))


    def set_as_foreground(self, backgroundNode: slicer.vtkMRMLScalarVolumeNode, opacity=0.75) -> None:
        check_type(backgroundNode, slicer.vtkMRMLScalarVolumeNode, 'backgroundNode')
        if not 0 <= opacity <= 1:
            raise ValueError("The opacity parameter must be between 0 and 1.")
        try:
            logger.info(f"Setting the volume node as the foreground image with opacity {opacity}")
            slicer.util.setSliceViewerLayers(background=backgroundNode, foreground=self.volumeNode, foregroundOpacity=opacity)
        except Exception as e:
            log_and_raise(logger, "An error occurred in setAsForeground", type(e))


    def set_cmap(self, cmap: Union[PresetColormaps, PetColormaps]) -> None:
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
        check_type(newName, str, 'newName')
        try:
            logger.info(f"Editing the name of the volume node to {newName}")
            self.volumeNode.SetName(newName)
            self.name = newName
        except Exception as e:
            log_and_raise(logger, "An error occurred in editName", type(e))


    def check_segment_shape_match(self, segmentationArray: np.ndarray) -> bool:
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
        if not isinstance(binaryMaskArray, np.ndarray):
            raise ValueError("Both imageArray and binaryMaskArray must be NumPy arrays.")
        self.NumPyArray = np.where(binaryMaskArray == 1, 0, self.NumPyArray)
        if updateScene:
            self.update_slicer_view()

    
    def save_volume(self, outputDir : str, fileType : str = 'nrrd', additionalSaveInfo: str = None) -> None:
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



class SegmentationNode:

    def __init__(self, SegmentationNodeObject: slicer.vtkMRMLSegmentationNode, name: str = None):
        self.segmentationNode = SegmentationNodeObject
        self._segmentation = self.segmentationNode.GetSegmentation()
        self.nodeID = self.segmentationNode.GetID()
        self.segments = [] # List to store Segment Instances
        if name == None:
            self.name = self.segmentationNode.GetName()
        else:
            self.name = name
            self.segmentationNode.SetName(self.name)
        
        # Initialize the segmentation Node with all created segments
        for i in range(self._segmentation.GetNumberOfSegments()):
            segmentObject = self._segmentation.GetNthSegment(i)
            self.segments.append(Segment(self, segmentObject))


    def description(self) -> str:
        return f"Name: {self.name}, Number of segments: {len(self.segments)}"


    def get_name(self) -> str:
        return self.name


    def get_segments(self) -> tuple:
        return tuple(self.segments) 


    def get_segment_names(self) -> tuple:
        segment_names = [segment.getName() for segment in self.segments]
        return tuple(segment_names)


    def get_number_of_segments(self) -> int:
        return len(self.segments)


    def get_segment_by_segment_name(self, segmentName: str) -> Segment:
        for segment in self.segments:
            if segment.getName() == segmentName:
                return segment
        logger.warning(f"No segment with the name {segmentName} was found.")


    def edit_name(self, newName : str) -> None:
        self.name = newName
        self.segmentationNode.SetName(newName)


    def _add_segment(self, segmentObject: Segment, segmentName: str = None) -> None:
        if segmentName == None:
            self.segments.append(Segment(self, segmentObject))
        else:
            self.segments.append(Segment(self, segmentObject, segmentName))


    def remove_segment(self, segment: Segment) -> None:
        check_type(segment, Segment, 'segment')
        self.segments.remove(segment)
        self._segmentation.RemoveSegment(segment.getSegmentID())
        del segment


    def remove_segment_by_name(self, segmentName: str) -> None:
        for segment in self.segments:
            if segment.getName() == segmentName:
                self.remove_segment(segment)
                return
        logger.warning(f"No segment with the name {segmentName} was found.")


    def clear_segments(self) -> None:
        for segment in self.segments:
            self._segmentation.RemoveSegment(segment.getSegmentID())
        self.segments.clear()


    def add_blank_segmentation(self, segmentName: str) -> str:
        check_type(segmentName, str, 'segmentName')
        try:
            logger.info(f"Adding a blank segment with name {segmentName}")
            segmentID = self._segmentation.GetSegmentIdBySegmentName(segmentName)
            if segmentID == None:            
                segmentID = self._segmentation.AddEmptySegment(segmentName)
                segmentObject = self._segmentation.GetNthSegment(segmentID)
                self._add_segment(segmentObject, segmentName)
                return segmentID
            else:
                logger.warning(f"A segment with the name {segmentName} already exists.")
                return segmentID
        except Exception as e:
            log_and_raise(logger, "An error occurred in addBlankSegmentation", type(e))


    def add_segment_from_array(self, segmentName: str, segmentArray: np.ndarray, referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, color: tuple = None) -> str:
        check_type(segmentName, str, 'segmentName')
        check_type(segmentArray, np.ndarray, 'segmentArray')
        check_type(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode, 'referenceVolumeNode')
        check_type(color, (tuple, type(None)), 'color')
        if color is not None:
            if len(color) != 3 or not all(isinstance(color[i], (int, float)) for i in range(3)):
                raise TypeError("The color parameter must be a tuple of three integers or floats.")
        else:
            color = tuple(np.random.rand(3))
        try:
            with TempNodeManager(slicer.mrmlScene, 'vtkMRMLScalarVolumeNode', segmentName) as tempVolumeNode:
                tempVolumeNode.CopyOrientation(referenceVolumeNode)
                tempVolumeNode.SetSpacing(referenceVolumeNode.GetSpacing())
                tempVolumeNode.CreateDefaultDisplayNodes()
                displayNode = tempVolumeNode.GetDisplayNode()
                displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRainbow')
                slicer.util.updateVolumeFromArray(tempVolumeNode, segmentArray)
                tempImageData = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(tempVolumeNode)
                segmentID = self.segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(tempImageData, segmentName, color)
            return segmentID
        except Exception as e:
            log_and_raise(logger, "An error occurred in addSegmentFromArray", type(e))




