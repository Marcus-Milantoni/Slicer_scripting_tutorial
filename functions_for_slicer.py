# Created by Marcus Milantoni and Edward Wang. This script contains basic functions for image processing in Slicer.

import slicer, vtk
from DICOMLib import DICOMUtils
import numpy as np
from numpy import random as rnd
import os
import logging
import matplotlib.pyplot as plt


# Setup the logger (can customize)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_DICOM(dicomDataDir):
    """
    This function loads DICOM data into Slicer. This function uses DICOMutils to handle the data types.

    Parameters
    ----------
    dicomDataDir : str
                The directory containing the DICOM data to load.
    
    Returns
    -------
    list
        The list of all loaded node IDs.
    """
    if not isinstance(dicomDataDir, str):
        logger.error("The dicomDataDir parameter must be a string.")
        raise TypeError("The dicomDataDir parameter must be a string.")
    if not os.path.isdir(dicomDataDir):
        logger.error("The dicomDataDir parameter must be a valid directory.")
        raise ValueError("The dicomDataDir parameter must be a valid directory.")

    try:
        loadedNodeIDs = []  # this list will contain the list of all loaded node IDs

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            logger.debug(f"Importing DICOM data from directory: {dicomDataDir}")
            DICOMUtils.importDicom(dicomDataDir, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                logger.debug(f"Loading patient with UID: {patientUID}")
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

        return loadedNodeIDs

    except Exception:
        logger.exception("An error occurred in load_DICOM")
        raise


def get_segment_array(segmentID, segmentationNode=None, referenceVolumeNode=None):
    """
    This function returns the full size segmentation array from a given segment ID.
    
    Parameters
    ----------
    segmentID : str
                The segment ID to get the full size segmentation from.
    
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to get the segment from.
    
    referenceVolumeNode : vtkMRMLVolumeNode, default: None
                The reference volume node to get the segment from.
    
    Returns
    -------
    numpy.ndarray
        The full size segmentation array.
    """
    if not isinstance(segmentID, str):
        logger.error("The segmentID parameter must be a string.")
        raise TypeError("The segmentID parameter must be a string.")
    if segmentationNode is not None and not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        logger.error("The segmentationNode parameter must be of type vtkMRMLSegmentationNode or None.")
        raise TypeError("The segmentationNode parameter must be of type vtkMRMLSegmentationNode or None.")
    if referenceVolumeNode is not None and not isinstance(referenceVolumeNode, slicer.vtkMRMLVolumeNode):
        logger.error("The referenceVolumeNode parameter must be of type vtkMRMLVolumeNode or None.")
        raise TypeError("The referenceVolumeNode parameter must be of type vtkMRMLVolumeNode or None.")

    if segmentationNode is None:
        logger.debug("No segmentationNode provided, using the first vtkMRMLSegmentationNode from the scene.")
        segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    if referenceVolumeNode is None:
        logger.debug("No referenceVolumeNode provided, using the first vtkMRMLScalarVolumeNode from the scene.")
        referenceVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")

    try:
        logger.debug(f"Getting the segment array for segment ID: {segmentID}")
        segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentID, referenceVolumeNode)
        return segmentArray
    
    except Exception:
        logger.exception("An error occurred in get_segment_array")
        raise


def create_volume_node(volumeArray, referenceNode, volumeName):
    """
    This function creates a volume node from a numpy array.

    Parameters
    ----------
    volumeArray : numpy.ndarray
                The numpy array to create the volume node from.
    referenceNode : vtkMRMLScalarVolumeNode
                The reference node to create the volume node from.
    volumeName : str
                The name of the volume node to create.
    
    Returns
    -------
    vtkMRMLScalarVolumeNode
        The volume node created from the numpy array.
    """
    if not isinstance(volumeArray, np.ndarray):
        logger.error("The volumeArray parameter must be a numpy array.")
        raise TypeError("The volumeArray parameter must be a numpy array.")
    if not isinstance(referenceNode, slicer.vtkMRMLScalarVolumeNode):
        logger.error("The referenceNode parameter must be a vtkMRMLScalarVolumeNode.")
        raise TypeError("The referenceNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(volumeName, str):
        logger.error("The volumeName parameter must be a string.")
        raise TypeError("The volumeName parameter must be a string.")
    
    try:
        logger.debug(f"Creating a new volume node with name: {volumeName}")
        doseNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', volumeName)
        doseNode.CopyOrientation(referenceNode)
        doseNode.SetSpacing(referenceNode.GetSpacing())
        doseNode.CreateDefaultDisplayNodes()
        displayNode = doseNode.GetDisplayNode()
        displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRainbow')
        slicer.util.updateVolumeFromArray(doseNode, volumeArray)

        return doseNode
    
    except Exception:
        logger.exception("An error occurred in create_volume_node")
        raise


def add_segmentation_array_to_node(segmentationNode, segmentationArray, segmentName, referenceVolumeNode, color=None):
    """
    This function adds a segmentation to a node from a numpy array.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the segmentation to.
    segmentationArray : numpy.ndarray
                The numpy array to add to the segmentation node.
    segmentName : str
                The name of the segmentation to add.
    referenceVolumeNode : vtkMRMLScalarVolumeNode
                The reference volume node to add the segmentation to.
    color : tuple, default: rnd.rand(3)
                The color of the segmentation to add.
        
    Returns
    -------
    str
        The segment ID of the added segmentation.
    """
    if color is None:
        color = tuple(rnd.rand(3))

    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        logger.error("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(segmentationArray, np.ndarray):
        logger.error("The segmentationArray parameter must be a numpy array.")
        raise TypeError("The segmentationArray parameter must be a numpy array.")
    if not isinstance(segmentName, str):
        logger.error("The segmentName parameter must be a string.")
        raise TypeError("The segmentName parameter must be a string.")
    if not isinstance(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode):
        logger.error("The referenceVolumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        raise TypeError("The referenceVolumeNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(color, tuple):
        logger.error("The color parameter must be a tuple.")
        raise TypeError("The color parameter must be a tuple.")
    
    try:
        logger.debug(f"Creating a temporary volume node for segmentation array.")
        tempVolumeNode = create_volume_node(segmentationArray, referenceVolumeNode, "TempNode")
        tempImageData = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(tempVolumeNode)
        slicer.mrmlScene.RemoveNode(tempVolumeNode)
        logger.debug(f"Adding the segmentation to the segmentation node.")
        segmentID = segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(tempImageData, segmentName, color)

        return segmentID
    
    except Exception:
        logger.exception("An error occurred in add_segmentation_array_to_node")
        raise


def check_intersection_binary_mask(mask_1_array, mask_2_array, num_voxels_threshold=1):
    """
    Checks if two binary masks intersect.

    Parameters
    ----------
    mask_1_array : numpy.ndarray
             The first binary mask.
    mask_2_array : numpy.ndarray
             The second binary mask.
    num_voxels_threshold : int, default: 1
                           The number of voxels that must be intersecting for the function to return True.
               
    Returns
    -------
    bool
        True if the masks intersect, False if they do not intersect.
    """
    if not isinstance(mask_1_array, np.ndarray):
        logger.error("The mask_1_array parameter must be a numpy array.")
        raise TypeError("The mask_1_array parameter must be a numpy array.")
    if not isinstance(mask_2_array, np.ndarray):
        logger.error("The mask_2_array parameter must be a numpy array.")
        raise TypeError("The mask_2_array parameter must be a numpy array.")
    if not isinstance(num_voxels_threshold, int):
        logger.error("The num_voxels_threshold parameter must be an integer.")
        raise TypeError("The num_voxels_threshold parameter must be an integer.")
        
    try:
        intersection = np.logical_and(mask_1_array, mask_2_array)
        if np.sum(intersection) >= num_voxels_threshold:
            return True
        else:
            return False
    
    except Exception:
        logger.exception("An error occurred in check_intersection_binary_mask")
        raise


def margin_editor_effect(inputName, newName, segmentationNode, volumeNode, operation='Grow', MarginSize=10.0):
    """
    This function dilates or shrinks a segment in a segmentation node. This function automatically copies the function before applying the margin effect (makes a new segment).

    Parameters
    ----------
    inputName : str
                The name of the segment to dilate or shrink.
    newName : str
                The name of the new segment.
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node containing the segment to dilate or shrink.
    volumeNode : vtkMRMLScalarVolumeNode
                The volume node that the segmentation node is based on.
    operation : str, default: 'Grow'
                The operation to perform on the segment. Must be 'Grow' or 'Shrink'.
    MarginSize : float, default: 10.0
                The number of voxels to dilate or shrink the segment by.

    Returns
    -------
    str
        The segment ID of the new segment.
    """
    if not isinstance(inputName, str):
        logger.error("The inputName parameter must be a string.")
        raise TypeError("The inputName parameter must be a string.")
    if not isinstance(newName, str):
        logger.error("The newName parameter must be a string.")
        raise TypeError("The newName parameter must be a string.")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        logger.error("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
        logger.error("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(operation, str):
        logger.error("The operation parameter must be a string.")
        raise TypeError("The operation parameter must be a string.")
    if not isinstance(MarginSize, float):
        logger.error("The MarginSize parameter must be a float.")
        raise TypeError("The MarginSize parameter must be a float.")
    
    try:
        logger.debug(f"Copying segment {inputName} to {newName}")
        copy_segmentation(segmentationNode, inputName, newName)

        logger.debug(f"Setting up segment editor widget")
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setSourceVolumeNode(volumeNode)
        segmentEditorNode.SetOverwriteMode(2)  # i.e. "allow overlap" in UI

        newSegmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(newName)
        segmentEditorNode.SetSelectedSegmentID(newSegmentID)

        logger.debug(f"Setting up margin effect")
        segmentEditorWidget.setActiveEffectByName("Margin")
        effect = segmentEditorWidget.activeEffect()
    
        logger.debug(f"Applying margin effect")
        if operation in ['Grow', 'Shrink']:
            effect.setParameter("Operation", operation)
        else:
            logger.error("Invalid operation. Operation must be 'Grow' or 'Shrink.'")
            raise ValueError("Invalid operation. Operation must be 'Grow' or 'Shrink.'")
        
        if (isinstance(MarginSize, float) or isinstance(MarginSize, int)) and MarginSize > 0:
            effect.setParameter("MarginSizeMm", MarginSize)
        else:
            logger.error("Invalid MarginSize. MarginSize must be a positive number.")
            raise ValueError("Invalid MarginSize. MarginSize must be a positive number.")
        effect.self().onApply() 

        return newSegmentID 

    except TypeError:
        logger.exception("A type error occurred in margin_editor_effect")
        raise
    except ValueError:
        logger.exception("A value error occurred in margin_editor_effect")
        raise
    except Exception:
        logger.exception("An error occurred in margin_editor_effect")
        raise
        

def make_blank_segment(segmentationNode, segmentName):
    """
    This function creates a blank segment in a segmentation node.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the segment to.
    segmentName : str
                The name of the segment to add.

    Returns
    -------
    str
        The segment ID of the new segment.
    """
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        logger.error("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(segmentName, str):
        logger.error("The segmentName parameter must be a string.")
        raise TypeError("The segmentName parameter must be a string.")
    
    try:
        logger.debug(f"Creating a blank segment with name {segmentName}")
        segmentID = segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
        return segmentID
    
    except Exception:
        logger.exception("An error occurred in make_blank_segment")
        raise


def copy_segmentation(segmentationNode, segmentName, newSegmentName):
    """
    This function copies a segment in a segmentation node.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to copy the segment in.
    segmentName : str
                The name of the segment to copy.
    newSegmentName : str
                The name of the new segment.

    Returns
    -------
    str
        The segment ID of the new segment.
    """
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(segmentName, str):
        raise TypeError("The segmentName parameter must be a string.")
    if not isinstance(newSegmentName, str):
        raise TypeError("The newSegmentName parameter must be a string.")
    
    try:
        logger.debug(f"Setting up segment editor widget")
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        segmentEditorWidget.setSegmentationNode(segmentationNode)
        volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
        segmentEditorWidget.setSourceVolumeNode(volumeNode)
        # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
        segmentEditorNode.SetOverwriteMode(2)  # i.e. "allow overlap" in UI

        # Get the segment IDs
        logger.debug(f"getting the segment IDs")
        segmentationNode.AddSegmentFromClosedSurfaceRepresentation(vtk.vtkPolyData(), newSegmentName)
        targetSegmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(newSegmentName)
        modifierSegmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        
        logger.debug(f"Setting the parameters for the logical operators effect")
        segmentEditorNode.SetSelectedSegmentID(targetSegmentID)
        segmentEditorWidget.setActiveEffectByName("Logical operators")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("Operation", "COPY")  # change the operation here
        effect.setParameter("ModifierSegmentID", modifierSegmentID)
        effect.self().onApply()

        return targetSegmentID
    
    except Exception:
        logger.exception("An error occurred in copy_segmentation")
        raise


def combine_masks_from_array(mask1, mask2, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Combines two binary masks. This function uses numpy's bitwise_or function to combine the masks. The function can also add the combined mask to a segmentation node.

    Parameters
    ----------
    mask1 : numpy.ndarray
             The first binary mask.
    mask2 : numpy.ndarray
             The second binary mask.
    addSegmentationToNode : bool, default: False
                Specification to add the combined mask to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to add the combined mask.
    volumeNode : vtkMRMLScalarVolumeNode, default: None
                The volume node that the segmentation node is based on. 
    segmentName : str, default: None
                The name of the segment to add the combined mask to.
               
    Returns
    -------
    numpy.ndarray
        The combined binary mask.
    """
    if not isinstance(mask1, np.ndarray):
        raise TypeError("The mask1 parameter must be a numpy.ndarray.")
    if not isinstance(mask2, np.ndarray):
        raise TypeError("The mask2 parameter must be a numpy.ndarray.")
    
    logger.debug(f"Combining masks")
    combined_mask = np.bitwise_or(mask1, mask2)
    
    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(segmentName, str):
            raise TypeError("The segment_name parameter must be a string.")

        logger.debug(f"Adding the combined mask to the segmentation node {segmentationNode}")     
        add_segmentation_array_to_node(segmentationNode, combined_mask, segmentName, volumeNode)
    
    return combined_mask


def bitwise_and_from_array(mask1, mask2, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Combines two binary masks where the masks overlap. This function uses numpy's bitwise_and function to combine the masks. The function can also add the combined mask to a segmentation node.

    Parameters
    ----------
    mask1 : numpy.ndarray
             The first binary mask.
    mask2 : numpy.ndarray
             The second binary mask.
    addSegmentationToNode : bool, default: False
                Specification to add the combined mask to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the combined mask to.
    volumeNode : vtkMRMLScalarVolumeNode
                The volume node that the segmentation node is based on. 
    segment_name : str
                The name of the segment to add the combined mask to.
               
    Returns
    -------
    numpy.ndarray
        The combined binary mask.
    """
    try:
        if isinstance(mask1, np.ndarray) and isinstance(mask2, np.ndarray):
            logger.debug(f"Combining masks")
            combined_mask = np.bitwise_and(mask1, mask2)
            
            if addSegmentationToNode:
                if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                    raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
                if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
                    raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
                if not isinstance(segmentName, str):
                    raise TypeError("The segment_name parameter must be a string.")

                logger.debug(f"Adding the combined mask to the segmentation node {segmentationNode}")     
                add_segmentation_array_to_node(segmentationNode, combined_mask, segmentName, volumeNode)
            
            return combined_mask
        
        else:
            if not isinstance(mask1, np.ndarray):
                raise TypeError("The mask1 parameter must be a numpy.ndarray.")
            if not isinstance(mask2, np.ndarray):
                raise TypeError("The mask2 parameter must be a numpy.ndarray.")
        
    except TypeError:
        logger.exception("A type error occurred in bitwise_and_from_array")
        raise
    except ValueError:
        logger.exception("A value error occurred in bitwise_and_from_array")
        raise


def remove_mask_superior_to_slice(maskArray, slice_number, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Crops the input maskArray to the region superior of the specified slice number. This function can also add the cropped mask to a segmentation node.
        
    Parameters
    ----------  
    maskArray : numpy.ndarray
                  The binary mask that is getting cropped.
    slice_number : int
                   The slice number used for the crop.
    addSegmentationToNode : bool, default: False
                Specification to add the cropped mask to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped mask to.
    volumeNode : vtkMRMLScalarVolumeNode, default: None
                The volume node that the segmentation node is based on.
    segmentName : str, default: None
                The name of the segment to add the cropped mask to.
    
    Returns
    -------
    numpy.ndarray
        The mask array cropped.
    """
    if not isinstance(maskArray, np.ndarray):
        logger.error("The maskArray parameter must be a numpy.ndarray.")
        raise TypeError("The maskArray parameter must be a numpy.ndarray.")
    if not isinstance(slice_number, int):
        logger.error("The slice_number parameter must be an integer.")
        raise TypeError("The slice_number parameter must be an integer.")
    if slice_number < 0 or slice_number >= maskArray.shape[0]:
        logger.error("The slice_number parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")
        raise IndexError("The slice_number parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")

    logger.debug(f"Cropping the mask to the region superior to slice {slice_number}")
    main_copy = np.copy(maskArray)
    main_copy[slice_number + 1:] = 0

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(segmentName, str):
            raise TypeError("The segment_name parameter must be a string.")

        logger.debug(f"Adding the cropped mask to the segmentation node {segmentationNode}")     
        add_segmentation_array_to_node(segmentationNode, main_copy, segmentName, volumeNode)
    
    return main_copy


def remove_mask_inferior_to_slice(maskArray, slice_number, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Crops the input maskArray to the region inferior to the specified slice number. This function can also add the cropped mask to a segmentation node.
        
    Parameters
    ----------  
    maskArray : numpy.ndarray
                  The binary mask that is getting cropped.
    slice_number : int
                   The slice number used for the crop.
    addSegmentationToNode : bool, default: False
                Specification to add the cropped mask to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped mask to.
    volumeNode : vtkMRMLScalarVolumeNode, default: None
                The volume node that the segmentation node is based on.
    segmentName : str, default: None
                The name of the segment to add the cropped mask to.
    
    Returns
    -------
    numpy.ndarray
        The mask array cropped.
    """
    if not isinstance(maskArray, np.ndarray):
        logger.error("The maskArray parameter must be a numpy.ndarray.")
        raise TypeError("The maskArray parameter must be a numpy.ndarray.")
    if not isinstance(slice_number, int):
        logger.error("The slice_number parameter must be an integer.")
        raise TypeError("The slice_number parameter must be an integer.")
    if slice_number < 0 or slice_number >= maskArray.shape[0]:
        logger.error("The slice_number parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")
        raise IndexError("The slice_number parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")

    logger.debug(f"Cropping the mask to the region inferior to slice {slice_number}")
    main_copy = np.copy(maskArray)
    main_copy[:slice_number] = 0

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(segmentName, str):
            raise TypeError("The segment_name parameter must be a string.")

        logger.debug(f"Adding the cropped mask to the segmentation node {segmentationNode}")     
        add_segmentation_array_to_node(segmentationNode, main_copy, segmentName, volumeNode)
    
    return main_copy


def project_segmentation_vertically_from_array(maskArray, numberOfSlicesToCombine, numberOfSlicesToProject=5, projectInferior=False, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    This function projects a binary mask array in the vertical direction (either uperiorly or posteriorly). The function can also add the projected mask to a segmentation node.

    Parameters
    ----------
    maskArray : numpy.ndarray
                The binary mask array to project.
    nuberOfSlicesToCombine : int
                The number of slices to combine before projection.
    numberOfSlicesToProject : int, default: 5
                The number of slices to project.
    projectInferior : bool, default: True
                Specification to project the inferior direction.
    addSegmentationToNode : bool, default: False
                Specification to add the projected segment to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to add the projected segment to.
    volumeNode : vtkMRMLScalarVolumeNode, default: None
                The volume node that the segmentation node is based on. 
    segmentName : str, default: None
                The name of the segment to add the projected mask to.

    Returns
    -------
    numpy.ndarray
                The numpy array with the projected segmentation
    """
    if not isinstance(maskArray, np.ndarray):
        logger.error("The maskArray parameter must be a numpy.ndarray.")
        raise TypeError("The maskArray parameter must be a numpy.ndarray.")
    if not isinstance(numberOfSlicesToCombine, int):
        logger.error("The numberOfSlicesToCombine parameter must be an integer.")
        raise TypeError("The numberOfSlicesToCombine parameter must be an integer.")
    if not numberOfSlicesToCombine > 0:
        logger.error("The numberOfSlicesToCombine parameter must be greater than 0")
        raise ValueError("The numberOfSlicesToCombine parameter must be greater than 0")
    if not isinstance(numberOfSlicesToProject, int):
        logger.error("The numberOfSlicesToProject parameter must be an integer.")
        raise TypeError("The numberOfSlicesToProject parameter must be an integer.")
    
    slices_w_mask = []
    main_copy = np.copy(maskArray)
    for index, slice in enumerate(maskArray):
        if 1 in slice:
            slices_w_mask.append(index)

    try:
        if projectInferior:
            logger.debug(f"Projecting inferiorly")
            bottom_slice = min(slices_w_mask)

            logger.debug(f"Calculating the last slices")
            last_slices = main_copy[bottom_slice : bottom_slice + numberOfSlicesToCombine]
            result = last_slices[0]
            all_slices_to_change = np.arange(bottom_slice - numberOfSlicesToProject, bottom_slice + numberOfSlicesToProject)

            logger.debug(f"Creating the array")
            for slice in last_slices[1:]:
                result = np.bitwise_or(result, slice)        
            
            for index in all_slices_to_change:
                main_copy[index] = result

            logger.info(f"Finished creating the array")

        if not projectInferior:
            logger.debug(f"Projecting superiorly")
            top_slice = max(slices_w_mask)
            
            logger.debug(f"Calculating the first slices")
            first_slices = main_copy[top_slice - numberOfSlicesToCombine + 1 : top_slice + 1]
            result = first_slices[0]
            all_slices_to_change = np.arange(top_slice - numberOfSlicesToCombine + 1, top_slice + numberOfSlicesToProject + 1)

            logger.debug(f"Creating the array")
            for slice in first_slices[1:]:
                result = np.bitwise_or(result, slice)        

            for index in all_slices_to_change:
                main_copy[index] = result
            
            logger.info(f"Finished creating the array")

        if addSegmentationToNode:
            if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
            if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
                raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
            if not isinstance(segmentName, str):
                raise TypeError("The segment_name parameter must be a string.")
            
            logger.debug(f"Adding the segmentation array to the node {segmentationNode}")
            add_segmentation_array_to_node(segmentationNode, main_copy, segmentName, volumeNode)
        
        return main_copy

    except Exception as e:
        logger.exception(e)
        raise


def create_binary_mask_between(binaryMaskArrayLeft, binaryMaskArrayRight, fromMedial=True):
    """
    Creates a binary mask between two binary masks. The function can create the binary mask from the medial or lateral direction.
    
    Parameters
    ----------
    binaryMaskArrayLeft : numpy.ndarray
                The binary mask array to the left.
    binaryMaskArrayRight : numpy.ndarray
                The binary mask array to the right.
    fromMedial : bool, default: True
                Specification to create the binary mask from the medial direction.
                
    Returns
    -------
    numpy.ndarray
        The binary mask between the two binary masks.
    """
    if not isinstance(binaryMaskArrayLeft, np.ndarray):
        raise Exception("The binaryMaskArrayLeft parameter must be a numpy.ndarray.")
    if not isinstance(binaryMaskArrayRight, np.ndarray):
        raise Exception("The binaryMaskArrayRight parameter must be a numpy.ndarray.")
    if not isinstance(fromMedial, bool):
        raise Exception("The fromMedial parameter must be a boolean.")

    try:
        logger.debug("Making a copy of the masks.")
        left_copy = np.copy(binaryMaskArrayLeft)
        right_copy = np.copy(binaryMaskArrayRight)
        mask_w_ones = np.ones_like(left_copy)

        if fromMedial:
            logger.debug("Creating the mask from the medial direction.")
            most_medial_left = []
            most_medial_right = []

            for slice in right_copy:
                logger.debug("Finding the most medial slice in the right mask.")
                for index in range(np.shape(slice)[1]):
                    column = slice[:, index]
                    most_medial_in_slice = 0
                    if 1 in column and index > most_medial_in_slice:
                        most_medial_in_slice = index
                    most_medial_right.append(most_medial_in_slice)

            for slice in left_copy:
                logger.debug("Finding the most medial slice in the left mask.")
                for index in range(np.shape(slice)[1]):
                    column = slice[:, index]
                    if 1 in column:
                        most_medial_in_slice = index
                        most_medial_left.append(most_medial_in_slice)
                        break

            logger.debug("Developing the new mask.")
            mask_w_ones[:, :, min(most_medial_left):] = 0
            mask_w_ones[:, :, :max(most_medial_right) + 1] = 0

        else:
            logger.debug("Creating the mask from the lateral direction.")
            most_lateral_left = []
            most_lateral_right = []

            for slice in right_copy:
                logger.debug("Finding the most lateral slice in the right mask.")
                for index in range(np.shape(slice)[1]):
                    column = slice[:, index]
                    if 1 in column:
                        most_lateral_right.append(index)
                        break

            for slice in left_copy:
                logger.debug("Finding the most lateral slice in the left mask.")
                for index in range(np.shape(slice)[1]):
                    column = slice[:, index]
                    most_lateral_in_slice = 0
                    if 1 in column and index > most_lateral_in_slice:
                        most_lateral_in_slice = index
                    most_lateral_left.append(most_lateral_in_slice)

            logger.debug("Developing the new mask.")
            mask_w_ones[:, :, max(most_lateral_left) + 1:] = 0
            mask_w_ones[:, :, :max(most_lateral_right)] = 0

        return mask_w_ones

    except Exception as e:
        logger.exception(e)
        raise


def crop_anterior(binaryMaskToCrop, referenceMask, fromAnterior=True):
    """
    Crops a binary mask from the anterior.
    
    Parameters
    ----------
    binaryMaskToCrop : numpy.ndarray
                The binary mask to be cropped.
    referenceMask : numpy.ndarray
                The reference mask to be used for cropping.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
                
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    if not isinstance(binaryMaskToCrop, np.ndarray):
        raise Exception("The binaryMaskToCrop parameter must be a numpy.ndarray.")
    if not isinstance(referenceMask, np.ndarray):
        raise Exception("The referenceMask parameter must be a numpy.ndarray.")
    if not isinstance(fromAnterior, bool):
        raise Exception("The fromAnterior parameter must be a boolean.")

    try:
        logger.debug("Making a copy of the masks.")
        binary_mask = np.copy(binaryMaskToCrop)
        referenceMask_copy = np.copy(referenceMask)

        if fromAnterior:
            lo
            most_anterior = []

            for slice in referenceMask_copy:
                for index in range(np.shape(slice)[0]):
                    row = slice[index, :]
                    if 1 in row:
                        most_anterior.append(index)
                        break

            binary_mask[:, min(most_anterior):, :] = 0

        else:
            most_posterior = []

            for slice in referenceMask_copy:
                for index in range(np.shape(slice)[0]):
                    row = slice[index, :]
                    most_posterior_in_slice = 0
                    if 1 in row and index > most_posterior_in_slice:
                        most_posterior_in_slice = index
                    most_posterior.append(most_posterior_in_slice)

            binary_mask[:, :max(most_posterior), :] = 0

        return binary_mask

    except Exception as e:
        logger.exception(e)


def remove_multiple_rois_from_mask(binary_mask_array, Tuple_of_masks_to_remove, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    Removes a region of interest from a binary mask.
    
    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    Tuple_of_masks_to_remove : tuple[numpy.ndarray]
                The masks to be removed from the binary mask.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volume_name : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    add_segmentation_to_node : bool, default: False
                Specification to add the cropped binary mask to a node. 
    
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask_array, np.ndarray) and isinstance(Tuple_of_masks_to_remove, tuple) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
            for index, mask in enumerate(Tuple_of_masks_to_remove):
                if isinstance(mask, np.ndarray) and mask.shape == binary_mask.shape:
                    binary_mask = binary_mask - mask
                else:
                    raise Exception(f"The Tuple_of_masks_to_remove parameter must contain only numpy.ndarrays, The index {index} is not a numpy array.")
    
            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, binary_mask, volume_name, volumeNode, color=rnd.rand(3))
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volume_name, str):
                        raise Exception("The volume_name parameter must be a string.")
        
            return(binary_mask)
        
        else:
            if isinstance(binary_mask_array, np.ndarray) and isinstance(Tuple_of_masks_to_remove, tuple):
                raise Exception("The add_segmentation_to_node parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray):
                raise Exception("The Tuple_of_masks_to_remove parameter must be a tuple.")
            elif isinstance(Tuple_of_masks_to_remove, tuple):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def save_images(outputFolder, output_file_type=".nii"):
    """
    Save all loaded images to the output folder.

    Parameters
    ----------
    outputFolder : str
        The folder to save the images to.

    Returns
    -------
    None
    """
    patientUIDs = slicer.dicomDatabase.patients()
    for patientUID in patientUIDs:
        loadedNodeIDs = DICOMUtils.loadPatientByUID(patientUID)
        for loadedNodeID in loadedNodeIDs:
            # Check if we want to save this node
            node = slicer.mrmlScene.GetNodeByID(loadedNodeID)
            # Only export images
            if not node or not node.IsA('vtkMRMLScalarVolumeNode'):
                continue
            # Construct filename
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            seriesItem = shNode.GetItemByDataNode(node)
            studyItem = shNode.GetItemParent(seriesItem)
            patientItem = shNode.GetItemParent(studyItem)
            filename = shNode.GetItemAttribute(patientItem, 'DICOM.PatientID')
            filename += '_' + shNode.GetItemAttribute(seriesItem, 'DICOM.Modality')
            filename = slicer.app.ioManager().forceFileNameValidCharacters(filename) + output_file_type
            # Save node
            print(f'Write {node.GetName()} to {os.path.join(outputFolder, filename)}') 
            slicer.util.saveNode(node, os.path.join(outputFolder, filename))


def save_rtstructs_as_nii(output_folder, segmentation_to_save):
    """
    Save the segmentation to the output folder as a .nii file.
    
    Parameters
    ----------
    output_folder : str
                The folder to save the .nii file to.
    segmentation_to_save : list[str]
                The segmentations to save.
                
    Returns
    -------
    None
    """
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    segmentation = segmentationNode.GetSegmentation()

    # Create a vtkStringArray outside the loop
    segmentIds = vtk.vtkStringArray()

    for segment in segmentation_to_save:
        segmentation_id_to_save = segmentation.GetSegmentIdBySegmentName(segment)
        # Add each segment ID to the vtkStringArray
        segmentIds.InsertNextValue(segmentation_id_to_save)

    # Call the export function once, after the loop
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsBinaryLabelmapRepresentationToFiles(output_folder, segmentationNode, segmentIds, ".nii", False)


def max_distance(matrix):
    """
    Finds the maximum distance between two pixels in a 2D binary mask.
    
    Parameters
    ----------
    matrix : numpy.ndarray
                The binary mask to find the maximum distance in. Must be a 2D array.
    
    Returns
    -------
    float
        The maximum distance between two pixels in the binary mask.
    """
    coords = np.where(matrix == 1)
    if not coords[0].size or not coords[1].size:
        return 0  # Return 0 if no pixels are found

    min_x, max_x = np.min(coords[0]), np.max(coords[0])
    min_y, max_y = np.min(coords[1]), np.max(coords[1])

    max_dist = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    return max_dist


def find_first_aortic_arch(aortic_matrix):
    """
    Finds the first slice with an aortic arch in a binary mask array.
    
    Parameters
    ----------
    aortic_matrix : numpy.ndarray
                The binary mask array to find the first aortic arch in.
    
    Returns
    -------
    int
        The first slice with an aortic arch.
    """
    try:
        if isinstance(aortic_matrix, np.ndarray):
            aortic_copy = np.copy(aortic_matrix)

            for index, slice in enumerate(aortic_copy):
                if 1 in slice:
                    max_dist = max_distance(slice)
                    if max_dist > 70:
                        return index
                    
        else:
            raise Exception("The aortic_matrix parameter must be a numpy.ndarray.")
        
    except Exception as e:
        logger.exception(e)


def crop_inferior_from_slice_number(binary_mask_array, slice_number, include_slice=True):
    """
    Crops the given binary mask array to remove all slices inferior to the specified slice number.
    This function allows for selective cropping where the user can choose to include or exclude the slice
    at the specified slice number in the cropped output. The operation is performed in-place, modifying
    the input array.

    Parameters
    ----------
    binary_mask_array : numpy.ndarray
        The binary mask to be cropped. It should be a 3D array where each slice along the first dimension
        represents a 2D binary mask.
    slice_number : int
        The slice number from which the cropping should start. Slices inferior to this number will be
        modified based on the value of `include_slice`.
    include_slice : bool, optional
        A flag to determine whether the slice specified by `slice_number` should be included in the
        cropping operation. If True, the specified slice and all slices superior to it will remain
        unchanged, while all inferior slices will be set to 0. If False, the specified slice will also
        be set to 0, along with all inferior slices. The default is True.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask. The returned array is the same instance as the input `binary_mask_array`
        with modifications applied in-place.

    Raises
    ------
    Exception
        If `binary_mask_array` is not a numpy.ndarray or `slice_number` is not an integer, or if
        `include_slice` is not a boolean value, an exception is raised with an appropriate error message.
    """
    try:
        if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and (include_slice == True or include_slice == False):
            binary_mask = np.copy(binary_mask_array)
            
            if include_slice:
                for index, slice in enumerate(binary_mask):
                    if index >= slice_number:
                        slice[slice == 1] = 0

            elif not include_slice:
                for index, slice in enumerate(binary_mask):
                    if index > slice_number:
                        slice[slice == 1] = 0

            return(binary_mask)
        
        else:
            if isinstance(binary_mask_array, np.ndarray):
                raise Exception("The slice_number parameter must be an integer.")
            elif isinstance(slice_number, int):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_superior_to_slice(main_mask, slice_number):
    """
    Crops the given binary mask array to keep only the region superior to the specified slice number.
    This function modifies the input array in-place, setting all slices inferior to the specified slice
    number to zero, effectively removing them from the binary mask.

    Parameters
    ----------
    main_mask : numpy.ndarray
        The binary mask to be cropped. It should be a 3D array where each slice along the first dimension
        represents a 2D binary mask.
    slice_number : int
        The slice number above which the binary mask should be kept. All slices at and below this number
        will be set to 0.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask. The returned array is the same instance as the input `main_mask` with
        modifications applied in-place.

    Raises
    ------
    Exception
        If `main_mask` is not a numpy.ndarray or `slice_number` is not an integer, an exception is raised
        with an appropriate error message.
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(slice_number, int):
            main_copy = np.copy(main_mask)
            slice_shape = main_mask.shape[1:]
            dtype_img = main_mask.dtype

            for index, slice in enumerate(main_copy):
                if index <= slice_number:
                    np.putmask(slice, np.ones(slice_shape, dtype=dtype_img), 0)
            
            return main_copy
        
        else:
            if not isinstance(main_mask, np.ndarray):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")
            if not isinstance(slice_number, int):
                raise Exception("The slice_number parameter must be an integer.")
    
    except Exception as e:
        logger.exception(e)


def crop_posterior_from_slice_number(binary_mask_array, slice_number, tuple_of_masks, fromAnterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    This function crops a binary mask posteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binary_mask_array.

    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    slice_number : int
                The slice number to crop from.
    tuple_of_masks : tuple[numpy.ndarray]
                The masks to be used for cropping.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volume_name : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    add_segmentation_to_node : bool, default: False
                Specification to add the cropped binary mask to a node.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """

    try:
        if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(tuple_of_masks, tuple) and isinstance(fromAnterior, bool) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
    
            if fromAnterior:
                most_anterior =  []

                for mask in tuple_of_masks:
                    if isinstance(mask, np.ndarray):
                        slice = mask[slice_number]
                        for index in range(np.shape(slice)[0]):
                            row = slice[index,:]
                            if 1 in row:
                                most_anterior.append(index)
                                break

                print(min(most_anterior))
                binary_mask[:, : min(most_anterior),:] = 0


            elif fromAnterior == False:
                most_posterior =  []

                for mask in tuple_of_masks:
                    if isinstance(mask, np.ndarray):
                        slice = mask[slice_number]
                        for index in range(np.shape(slice)[0]):
                            row = slice[index,:]
                            most_posterior_in_slice = 0
                            if 1 in row:
                                if index > most_posterior_in_slice:
                                    most_posterior_in_slice = index
                            most_posterior.append(most_posterior_in_slice)

                binary_mask[:,: max(most_posterior),:] = 0
                    
            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, binary_mask, volume_name, volumeNode, color=rnd.rand(3))
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volume_name, str):
                        raise Exception("The volume_name parameter must be a string.")

        
            return(binary_mask)

        else:
            if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(tuple_of_masks, tuple):
                raise Exception("The fromAnterior parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int):
                raise Exception("The tuple_of_masks parameter must be a tuple.")
            elif isinstance(tuple_of_masks, tuple):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_anterior_from_slice_number(binary_mask_array, slice_number, tuple_of_masks, fromAnterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    This function crops a binary mask anteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binary_mask_array.
    
    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    slice_number : int
                The slice number to crop from.
    tuple_of_masks : tuple[numpy.ndarray]
                The masks to be used for cropping.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volume_name : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    add_segmentation_to_node : bool, default: False
                Specification to add the cropped binary mask to a node.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(tuple_of_masks, tuple) and isinstance(fromAnterior, bool) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
    
            if fromAnterior:
                most_anterior =  []

                for mask in tuple_of_masks:
                    if isinstance(mask, np.ndarray):
                        slice = mask[slice_number]
                        for index in range(np.shape(slice)[0]):
                            row = slice[index,:]
                            if 1 in row:
                                most_anterior.append(index)
                                break

                binary_mask[:, min(most_anterior):,:] = 0


            elif fromAnterior == False:
                most_posterior =  []

                for mask in tuple_of_masks:
                    if isinstance(mask, np.ndarray):
                        slice = mask[slice_number]
                        for index in range(np.shape(slice)[0]):
                            row = slice[index,:]
                            most_posterior_in_slice = 0
                            if 1 in row:
                                if index > most_posterior_in_slice:
                                    most_posterior_in_slice = index
                            most_posterior.append(most_posterior_in_slice)

                binary_mask[:,: max(most_posterior),:] = 0
                    
            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, binary_mask, volume_name, volumeNode, color=rnd.rand(3))
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volume_name, str):
                        raise Exception("The volume_name parameter must be a string.")

        
            return(binary_mask)

        else:
            if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(tuple_of_masks, tuple):
                raise Exception("The fromAnterior parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int):
                raise Exception("The tuple_of_masks parameter must be a tuple.")
            elif isinstance(tuple_of_masks, tuple):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_posterior_from_slice_number(binary_mask_array, slice_number, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    This function crops a binary mask posteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binary_mask_array.
    
    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    slice_number : int
                The slice number to crop from.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volume_name : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    add_segmentation_to_node : bool, default: False
                Specification to add the cropped binary mask to a node.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
            binary_mask[:,:slice_number,:] = 0

            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, binary_mask, volume_name, volumeNode, color=rnd.rand(3))
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volume_name, str):
                        raise Exception("The volume_name parameter must be a string.")

        
            return(binary_mask)

        else:
            if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int):
                raise Exception("The add_segmentation_to_node parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray):
                raise Exception("The slice_number parameter must be an integer.")
            
    except Exception as e:
        logger.exception(e)  


def get_most_superior_slice(tuple_of_masks):
    """
    Finds the most superior slice in a tuple of binary masks.
    
    Parameters
    ----------
    tuple_of_masks : tuple[numpy.ndarray]
                The tuple of binary masks to find the most superior slice in.
    
    Returns
    -------
    int
        The most superior slice.
    """
    try:
        if isinstance(tuple_of_masks, tuple):
            most_superior = 0
            for mask in tuple_of_masks:
                if isinstance(mask, np.ndarray):
                    for index, slice in enumerate(mask):
                        if 1 in slice and index > most_superior:
                            most_superior = index
                else:
                    raise Exception(f"The tuple_of_masks parameter must contain only numpy.ndarrays. The index {index} is not a numpy array.")
            return most_superior
        else:
            raise Exception("The tuple_of_masks parameter must be a tuple.")
    except Exception as e:
        logger.exception(e)



def crop_posterior_from_distance(binary_mask, referenceMask, number_of_pixels, fromAnterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    Crops a binary mask from the anterior direction from a specified distance.
    
    Parameters
    ----------
    binary_mask : numpy.ndarray
                The binary mask to be cropped.
    referewnce_mask : numpy.ndarray
                The reference mask to be used for cropping.
    number_of_pixels : int
                The number of pixels to crop from.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volume_name : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    add_segmentation_to_node : bool, default: False
                Specification to add the cropped binary mask to a node.
                
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask, np.ndarray) and isinstance(referenceMask, np.ndarray) and isinstance(number_of_pixels, int) and (fromAnterior == True or fromAnterior == False):
            binary_mask = np.copy(binary_mask)


            if fromAnterior:
                most_anterior =  []

                for slice in referenceMask:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        if 1 in row:
                            most_anterior.append(index)
                            break

                binary_mask[:, : min(most_anterior) - number_of_pixels,:] = 0


            elif fromAnterior == False:
                most_posterior =  []

                for slice in referenceMask:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        most_posterior_in_slice = 0
                        if 1 in row:
                            if index > most_posterior_in_slice:
                                most_posterior_in_slice = index
                        most_posterior.append(most_posterior_in_slice)

                binary_mask[:,: max(most_posterior) - number_of_pixels,:] = 0

            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, binary_mask, volume_name, volumeNode, color=rnd.rand(3))
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volume_name, str):
                        raise Exception("The volume_name parameter must be a string.")

            return binary_mask
    
        else:
            if isinstance(binary_mask, np.ndarray) and isinstance(referenceMask, np.ndarray) and isinstance(number_of_pixels, int):
                raise Exception("The fromAnterior parameter must be a boolean.")
            elif isinstance(binary_mask, np.ndarray) and isinstance(referenceMask, np.ndarray):
                raise Exception("The number_of_pixels parameter must be an integer.")
            elif isinstance(referenceMask, np.ndarray) and isinstance(number_of_pixels, int):
                raise Exception("The binary_mask parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def get_border_from_width_of_mask_saggital(mask, percentage):
    """
    This function gets a border to be used in cropping from a percentage of the mask.
    The percentage is measured from the patient right side of the mask (left looking at the image).

    Parameters
    ----------
    mask : numpy.ndarray
                The mask to get the border from.
    percentage : float (0-1)
                The percentage of the mask to get the border from.

    Returns
    -------
    int
        The border to be used in cropping.    
    """

    try:
        if isinstance(mask, np.ndarray) and isinstance(percentage, float) and (0 <= percentage <= 1):
            mask_copy = np.copy(mask)

            sagittal_slices_w_mask = []

            for index in range(mask_copy.shape[2]):
                if np.any(mask_copy[:, :, index] == 1):
                    sagittal_slices_w_mask.append(index)

            right_bound = min(sagittal_slices_w_mask)
            left_bound = max(sagittal_slices_w_mask)

            return int((left_bound-right_bound)*percentage  + right_bound)
        
        else:
            if isinstance(mask, np.ndarray):
                raise Exception("The percentage parameter must be a float between 0 and 1.")
            elif isinstance(percentage, float):
                raise Exception("The mask parameter must be a numpy.ndarray.")
    except Exception as e:
        logger.exception(e)


def create_binary_mask_between_slices(referenceMask_array, left_bound, right_bound, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    Creates a binary mask between two slice numbers.
    
    Parameters
    ----------
    left_bound : int
                The left bound of the binary mask.
    right_bound : int
                The right bound of the binary mask.
                
    Returns
    -------
    numpy.ndarray
        The binary mask between the two binary masks.
    """
    try:
        if isinstance(left_bound, int) and isinstance(right_bound, int):
            mask_w_ones = np.ones_like(referenceMask_array)

            mask_w_ones[:,:, : right_bound] = 0
            mask_w_ones[:,:, left_bound :] = 0


            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, mask_w_ones, volume_name, volumeNode, color=rnd.rand(3))
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volume_name, str):
                        raise Exception("The volume_name parameter must be a string.")
            
            return(mask_w_ones)
    
        else:
            if isinstance(left_bound, int):
                raise Exception("The right_bound parameter must be an integer.")
            elif isinstance(right_bound, int):
                raise Exception("The left_bound parameter must be an integer.")

    except Exception as e:
        logger.exception(e)


def get_middle_of_sup_and_post_from_midline(mask_array, percentage):
    """
    This function gets the middle of the superior and posterior part of a mask from the midline. The mask is a 3D array.
    The middle is defined as the middle of the superior and posterior part of the mask from the midline.

    Parameters
    ----------
    mask_array : numpy.ndarray
                The mask to get the middle from.
    percentage : float (0-1)
                The percentage of the mask to get the middle from.

    Returns
    -------
    int
        The middle of the superior and posterior part of the mask from the midline.
    """

    try:
        if isinstance(mask_array, np.ndarray) and isinstance(percentage, float) and (0 <= percentage <= 1):
            mask_copy = np.copy(mask_array)

            midline = int(mask_copy.shape[2] // 2)
            transverse_slices_w_mask = []

            midline_slice = mask_copy[:, :, midline]
            for index in range(mask_copy.shape[0]):
                if np.any(midline_slice[index, :] == 1):
                    transverse_slices_w_mask.append(index)

            return int((max(transverse_slices_w_mask) - min(transverse_slices_w_mask))*percentage + min(transverse_slices_w_mask))
        
        else:
            if isinstance(mask_array, np.ndarray):
                raise Exception("The percentage parameter must be a float between 0 and 1.")
            elif isinstance(percentage, float):
                raise Exception("The mask_array parameter must be a numpy.ndarray.")
            else:
                raise Exception("The mask_array parameter must be a numpy.ndarray and the percentage parameter must be a float between 0 and 1.")

    except Exception as e:
        logger.exception(e)


def get_border_from_width_of_mask_coronal(mask, percentage):
    """
    This function gets a border to be used in cropping from a percentage of the mask.
    The percentage is measured from the patient right side of the mask (left looking at the image).

    Parameters
    ----------
    mask : numpy.ndarray
                The mask to get the border from.
    percentage : float (0-1)
                The percentage of the mask to get the border from.

    Returns
    -------
    int
        The border to be used in cropping.    
    """

    try:
        if isinstance(mask, np.ndarray) and isinstance(percentage, float) and (0 <= percentage <= 1):
            mask_copy = np.copy(mask)

            sagittal_slices_w_mask = []

            for index in range(mask_copy.shape[1]):
                if np.any(mask_copy[:, index, :] == 1):
                    sagittal_slices_w_mask.append(index)

            right_bound = min(sagittal_slices_w_mask)
            left_bound = max(sagittal_slices_w_mask)

            return int((left_bound-right_bound)*percentage  + right_bound)
        
        else:
            if isinstance(mask, np.ndarray):
                raise Exception("The percentage parameter must be a float between 0 and 1.")
            elif isinstance(percentage, float):
                raise Exception("The mask parameter must be a numpy.ndarray.")
    except Exception as e:
        logger.exception(e)


def get_ct_and_pet_volume_nodes():
    """
    This function gets the volume nodes of the CT and PET images.

    Returns
    -------
    tuple[slicer.vtkMRMLVolumeNode]
        The volume nodes of the CT and PET images.
    """
    try:
        logger.info(f"sorting through nodes.")
        ct_node_found = False
        pet_node_found = False
        for Volume_Node in slicer.util.getNodesByClass("vtkMRMLVolumeNode"):
            if ('ct ' in Volume_Node.GetName().lower() or 'ct_' in Volume_Node.GetName().lower())and not ct_node_found:
                ct_node_found = True
                ct_node = Volume_Node
                logger.info(f"CT node found: {ct_node.GetName()}")
            elif ('suvbw' in Volume_Node.GetName().lower() or 'standardized_uptake_value_body_weight' in Volume_Node.GetName().lower() or 'pet ' in Volume_Node.GetName().lower()) and not pet_node_found:
                pet_node_found = True
                pet_node = Volume_Node
                logger.info(f"Pet node found: {pet_node.GetName()}")
            if ct_node_found and pet_node_found:
                break
        
        if ct_node_found and pet_node_found:
            return(ct_node, pet_node)
        else:
            raise Exception("CT and PET nodes not found.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def segmentation_by_auto_threshold(segmentation_name, segmentationNode, pet_node, threshold='OTSU'):
    """Function to apply the autosegmentation threshold to a new segmentation in slicer defined by the segmentation name.
    
    Parameters
    ----------
    segmentation_name : str
        The name of the segmentation to be created.
    threshold : str, optional
        The thresholding method to be used. Default is 'OTSU'.
    segmentationNode : vtkMRMLSegmentationNode, optional
        The segmentation node to be used. Default is the segmentationNode.
    pet_node : vtkMRMLVolumeNode, optional
        The PET node to be used. Default is the pet_node.
    
    Returns
    -------
    None
    """

    try:
        #Check to see if the inputs are of the correct type
        if not isinstance(segmentation_name, str):
            raise ValueError("Segmentation name must be a string")
        
        if not isinstance(threshold, str):
            raise ValueError("Threshold must be a string")
        
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise ValueError("segmentationNode must be a vtkMRMLSegmentationNode")
        
        if not isinstance(pet_node, slicer.vtkMRMLVolumeNode):
            raise ValueError("pet_node must be a vtkMRMLVolumeNode")
        
        #Create a blank segmentation to do the autothresholding
        make_blank_segment(segmentationNode,segmentation_name)

        # Get the segmentation editor widget
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        #Set the correct volumes to be used in the Segment Editor
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setSourceVolumeNode(pet_node)
        # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
        segmentEditorNode.SetOverwriteMode(2)
        # Get the segment ID
        segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentation_name)
        segmentEditorNode.SetSelectedSegmentID(segid_src)

        # Set the active effect to Threshold on the Segment Editor widget
        segmentEditorWidget.setActiveEffectByName("Threshold")
        effect = segmentEditorWidget.activeEffect()
        
        if threshold == 'OTSU' or threshold == 'otsu':
            effect.setParameter("AutoThresholdMethod","OTSU")
            effect.setParameter("AutoThresholdMode", "SET_LOWER_MAX")
            #Save the active effects
            effect.self().onAutoThreshold()
        else:
            raise Exception("Threshold not recognized")
            
        #Apply the effect
        effect.self().onApply()

    except Exception as e:
        logger.error(f"Error setting threshold: {e}")


def logical_operator_slicer(main_segmentation, secondary_segmentation, segmentationNode, volumeNode, operator='copy'):
    """Function to apply logical operations to two segmentations in slicer. The main segmentation is the segmentation that will be modified. The operator will select the type of transformation applied.
    
    Parameters
    ----------
    main_segmentation : str
        The name of the segmentation to be modified.
    secondary_segmentation : str
        The name of the segmentation to be used as a reference.
        operator : str, optional
        The operation to be applied. Default is 'copy'. Options are 'copy', 'union', 'intersect', 'subtract', 'invert', 'clear', 'fill'.
    segmentationNode : vtkMRMLSegmentationNode, optional
        The segmentation node to be used. Default is the segmentationNode.
    volumeNode : vtkMRMLVolumeNode, optional
        The volume node to be used. Default is the VolumeNode.
    
    Returns
    -------
    None
    """

    try:
        #Check to see if the inputs are of the correct type
        if not isinstance(main_segmentation, str):
            raise ValueError("Main segmentation must be a string")
        
        if not isinstance(secondary_segmentation, str):
            raise ValueError("Secondary segmentation must be a string")
        
        if not isinstance(operator, str):
            raise ValueError("Operator must be a string")
        
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise ValueError("segmentationNode must be a vtkMRMLSegmentationNode")
        
        if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
            raise ValueError("volumeNode must be a vtkMRMLVolumeNode")
          
        # Get the segmentation editor widget
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        #Set the correct volumes to be used in the Segment Editor
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
        segmentEditorWidget.setSourceVolumeNode(volumeNode)
        # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
        segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
        # Get the segment ID
        segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(main_segmentation)
        segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(secondary_segmentation)

        # Set the active effect to the logical operator on the Segment Editor widget
        segmentEditorNode.SetSelectedSegmentID(segid_src)
        segmentEditorWidget.setActiveEffectByName("Logical operators")
        effect = segmentEditorWidget.activeEffect()

        #Set the operation to be applied
        if operator.lower() == 'copy':
            effect.setParameter("Operation", "COPY")
        elif operator.lower() == 'union':
            effect.setParameter("Operation", "UNION")
        elif operator.lower() == 'intersect':
            effect.setParameter("Operation", "INTERSECT")
        elif operator.lower() == 'subtract':
            effect.setParameter("Operation", "SUBTRACT")
        elif operator.lower() == 'invert':
            effect.setParameter("Operation", "INVERT")
        elif operator.lower() == 'clear':
            effect.setParameter("Operation", "CLEAR")
        elif operator.lower() == 'fill':
            effect.setParameter("Operation", "FILL")
        else:
            raise ValueError("Operator not recognized")
        
        #Apply the effect 
        effect.setParameter("ModifierSegmentID", segid_tgt)
        effect.self().onApply()

    except Exception as e:
        logger.error(f"Error applying logical operator: {e}")


def zero_image_where_mask_is_present(image, mask):
    """
    Sets the values to zero in the specified image where a binary mask overlaps.

    Parameters
    ----------
    image : numpy.ndarray
        The original image as a NumPy array.
    mask : numpy.ndarray
        A binary mask with the same shape as the image. The mask should contain
        1s in the regions to be masked out (set to zero) and 0s elsewhere.

    Returns
    -------
    numpy.ndarray
        The modified image with values set to zero where the mask overlaps.

    Raises
    ------
    ValueError
        If the input image and mask do not have the same shape or are not NumPy arrays.
    """
    try:
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Both image and mask must be NumPy arrays.")
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape.")

        # Apply the mask: Set image pixels to zero where mask is 1
        modified_image = np.where(mask == 1, 0, image)

        return modified_image
    
    except ValueError as ve:
        logger.error(ve)


def islands_effect_segment_editor(segment_name, segmentationNode, volume_node, edit='KEEP_LARGEST_ISLAND', minimum_size=5):
    try:
        edit_options_size = ['KEEP_LARGEST_ISLAND', 'REMOVE_SMALL_ISLANDS', 'SPLIT_ISLANDS_TO_SEGMENTS']
        
        edit_options_others = ['KEEP_SELECTED_ISLAND', 'REMOVE_SELECTED_ISLAND', 'ADD_SELECTED_ISLAND']

        #Check to see if the inputs are of the correct type
        if not isinstance(segment_name, str):
            raise ValueError("Segmentation name must be a string")
        
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise ValueError("segmentationNode must be a vtkMRMLSegmentationNode")
        
        if not isinstance(volume_node, slicer.vtkMRMLVolumeNode):
            raise ValueError("pet_node must be a vtkMRMLVolumeNode")
        
        if not isinstance(edit, str) and edit not in edit_options_size + edit_options_others:
            raise ValueError("edit must be a string and one of 'KEEP_LARGEST_ISLAND', 'KEEP_SELECTED_ISLAND', 'REMOVE_SMALL_ISLANDS', 'REMOVE_SELECTED_ISLAND', 'ADD_SELECTED_ISLAND', 'SPLIT_ISLANDS_TO_SEGMENTS'")
        
        if not isinstance(minimum_size, int) or minimum_size < 0 or minimum_size > 1000:
            raise ValueError("minimum_size must be an integer between 0 and 1000")
        
        # Get the segmentation editor widget
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        #Set the correct volumes to be used in the Segment Editor
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setSourceVolumeNode(volume_node)

        # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
        segmentEditorNode.SetOverwriteMode(2)
        # Get the segment ID
        segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
        segmentEditorNode.SetSelectedSegmentID(segmentID)

        # Set the active effect to Islands on the segment editor widget
        segmentEditorWidget.setActiveEffectByName("Islands")
        effect = segmentEditorWidget.activeEffect()


        #set a minimum size for the effect
        if edit in edit_options_size:
            effect.setParameter("MinimumSize", minimum_size)

        # set the effect to be used
        effect.setParameter("Operation", edit)

        # Apply the effect
        effect.self().onApply()
    
    except ValueError as e:
        logger.error(e)


def pet_removal_for_autothreshold(pet_array, kidney_threshold, expanded_spleen_array, expanded_bladder_array):
    try:
        if not isinstance(pet_array, np.ndarray):
            raise ValueError("PET array must be a NumPy array.")
        if not isinstance(kidney_threshold, np.ndarray):
            raise ValueError("Kidney threshold array must be a NumPy array.")
        if not isinstance(expanded_spleen_array, np.ndarray):
            raise ValueError("Expanded spleen array must be a NumPy array.")
        if not isinstance(expanded_bladder_array, np.ndarray):
            raise ValueError("Expanded bladder array must be a NumPy array.")
        
        if not kidney_threshold.shape == pet_array.shape:
            raise ValueError("Kidney threshold and PET arrays must have the same shape.")
        if not expanded_spleen_array.shape == pet_array.shape:
            raise ValueError("Expanded spleen and PET arrays must have the same shape.")
        if not expanded_bladder_array.shape == pet_array.shape:
            raise ValueError("Expanded bladder and PET arrays must have the same shape.")
        
        pet_copy = np.copy(pet_array)

        kidney_spleen_bladder = np.bitwise_or(kidney_threshold.astype(int), expanded_spleen_array.astype(int))
        kidney_spleen_bladder = np.bitwise_or(kidney_spleen_bladder, expanded_bladder_array.astype(int))

        modified_pet = np.where(kidney_spleen_bladder == 1, 0, pet_copy)

        return modified_pet

    except ValueError as e:
        logger.error(e)


def make_pet_for_autothreshold_kidney_removal(pet_array, left_kidney_modified, right_kidney_modified):
    try:
        if not isinstance(pet_array, np.ndarray):
            raise ValueError("PET array must be a NumPy array.")
        if not isinstance(left_kidney_modified, np.ndarray):
            raise ValueError("Left kidney modified array must be a NumPy array.")
        if not isinstance(right_kidney_modified, np.ndarray):
            raise ValueError("Right kidney modified array must be a NumPy array.")
        
        if not left_kidney_modified.shape == pet_array.shape:
            raise ValueError("Left kidney and PET arrays must have the same shape.")
        if not right_kidney_modified.shape == pet_array.shape:
            raise ValueError("Right kidney and PET arrays must have the same shape.")
        
        pet_copy = np.copy(pet_array)

        right_and_left_combined = np.bitwise_or(left_kidney_modified, right_kidney_modified)
        modified_pet = np.where(right_and_left_combined != 1, 0, pet_copy)

        return modified_pet

    except ValueError as e:
        logger.error(e)


def right_kidney_expand_from_left(left_kidney_convex_hull, right_kidney_array, pet_array):
    try:
        if not isinstance(left_kidney_convex_hull, np.ndarray):
            raise ValueError("Kidney array must be a NumPy array.")
        if not isinstance(pet_array, np.ndarray):
            raise ValueError("PET array must be a NumPy array.")
        if not isinstance(right_kidney_array, np.ndarray):
            raise ValueError("PET array must be a NumPy array.")
        
        if not left_kidney_convex_hull.shape == pet_array.shape:
            raise ValueError("Left kidney convex hull and PET arrays must have the same shape.")
        if not right_kidney_array.shape == pet_array.shape:
            raise ValueError("Right kidney and PET arrays must have the same shape.")
        
        # duplicate the right kidney array
        right_kidney_copy = np.copy(right_kidney_array).astype(int)

        # get the bounds (top and bottom) slice of the left kidney
        top_slice = max(left_kidney_convex_hull.nonzero()[0])

        # get the largest slice in the right kidney array
        sums_axial = np.sum(right_kidney_copy, axis=(1,2))
        largest_slice_index = sums_axial.argmax()
        largest_axial_mask = right_kidney_copy[largest_slice_index]

        # get the convex hull of the largest slice
        right_kidney_convex_hull = ski.morphology.convex_hull_image(largest_axial_mask)

        for i in range( largest_slice_index, top_slice+1):
            right_kidney_copy[i] = np.bitwise_or(right_kidney_copy[i], right_kidney_convex_hull)

        return right_kidney_copy
        
    except ValueError as e:
        logger.error(e)


def grow_left_kidney(kidney_array, pet_array):

    try:
        if not isinstance(kidney_array, np.ndarray):
            raise ValueError("Kidney array must be a NumPy array.")
        if not isinstance(pet_array, np.ndarray):
            raise ValueError("PET array must be a NumPy array.")
        
        if not kidney_array.shape == pet_array.shape:
            raise ValueError("Kidney and PET arrays must have the same shape.")
        
        # duplicate the kidney array
        kidney_copy = np.copy(kidney_array).astype(int)
        pet_copy = np.copy(pet_array)

        logger.debug(kidney_copy.nonzero()[0])

        # get the top and bottom slice of the kidney
        top_slice = max(kidney_copy.nonzero()[0])
        bottom_slice = min(kidney_copy.nonzero()[0])

        print(top_slice, bottom_slice)

        # get the largest slice in the kidney array
        sums_axial = np.sum(kidney_copy, axis=(1,2))
        largest_slice_index = sums_axial.argmax()
        largest_axial_mask = kidney_copy[largest_slice_index]

        # get the convex hull of the largest slice
        kidney_convex_hull = ski.morphology.convex_hull_image(largest_axial_mask)

        uniform_kidney = np.zeros_like(kidney_copy)
        uniform_kidney[bottom_slice:top_slice+1] = kidney_convex_hull
        uniform_kidney = np.bitwise_or(uniform_kidney, kidney_copy)

        #get the standard 42% of he suv max used for thresholding
        max_pet_value = np.max(pet_array)
        percentile_42 = max_pet_value * 0.42

        logger.info(f"Max PET value: {max_pet_value}, 42%: {percentile_42}")

        continue_moving_up = True
        current_slice = top_slice
        while continue_moving_up:
            slice_in_kidney = np.multiply(pet_copy[current_slice], kidney_convex_hull)
            if logger.getEffectiveLevel() == logging.DEBUG:
                plt.imshow(slice_in_kidney, cmap='hot')
                logger.debug(f"Slice {current_slice}, max value: {np.max(slice_in_kidney)}")
            if np.max(slice_in_kidney) > percentile_42:
                uniform_kidney[current_slice] = kidney_convex_hull
                current_slice += 1
                logger.info(f"the hotspot is {np.max(slice_in_kidney)}")
            else:
                continue_moving_up = False
                logger.info(f"The maximum slice is found at {current_slice-1}")

        return uniform_kidney

    except ValueError as e:
        logger.error(e)


def zero_image_where_mask_is_present(image, mask):
    """
    Sets the values to zero in the specified image where a binary mask overlaps.

    Parameters
    ----------
    image : numpy.ndarray
        The original image as a NumPy array.
    mask : numpy.ndarray
        A binary mask with the same shape as the image. The mask should contain
        1s in the regions to be masked out (set to zero) and 0s elsewhere.

    Returns
    -------
    numpy.ndarray
        The modified image with values set to zero where the mask overlaps.

    Raises
    ------
    ValueError
        If the input image and mask do not have the same shape or are not NumPy arrays.
    """
    try:
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Both image and mask must be NumPy arrays.")
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape.")

        # Apply the mask: Set image pixels to zero where mask is 1
        modified_image = np.where(mask != 1, image, np.min(image))

        return modified_image
    
    except ValueError as ve:
        logger.error(ve)


def resampleScalarVolumeBrains(inputVolumeNode, referenceVolumeNode, NodeName, interpolator_type='NearestNeighbor'):
    """
    Resamples a scalar volume node based on a reference volume node using the BRAINSResample module.

    This function creates a new scalar volume node in the Slicer scene, resamples the input volume node
    to match the geometry (e.g., dimensions, voxel size, orientation) of the reference volume node, and
    assigns the specified node name to the newly created volume node if possible. If the specified node
    name is already in use or not provided, a default name is assigned by Slicer.

    Parameters
    ----------
    inputVolumeNode : vtkMRMLScalarVolumeNode
        The input volume node to be resampled.
    referenceVolumeNode : vtkMRMLScalarVolumeNode
        The reference volume node whose geometry will be used for resampling.
    NodeName : str
        The name to be assigned to the newly created, resampled volume node.
    interpolator_type : str, optional
        The interpolator type to be used for resampling. The default is 'NearestNeighbor'. Other options include 'Linear', 'ResampleInPlace', 'BSpline', and 'WindowedSinc'.
    

    Returns
    -------
    vtkMRMLScalarVolumeNode
        The newly created and resampled volume node.

    Raises
    ------
    ValueError
        If the resampling process fails, an error is raised with the CLI execution failure message.
    """
    # Set parameters
    try:
        if not isinstance(inputVolumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise ValueError("Input volume node must be a vtkMRMLScalarVolumeNode")
        if not isinstance(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise ValueError("Reference volume node must be a vtkMRMLScalarVolumeNode")
        if not isinstance(NodeName, str):
            raise ValueError("Node name must be a string")
        if not isinstance(interpolator_type, str) and interpolator_type not in ['NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline','WindowedSinc']:
            raise ValueError("Interpolator type must be a string and one of 'NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline', 'WindowedSinc'")

        parameters = {}
        parameters["inputVolume"] = inputVolumeNode
        try:
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', NodeName)
        except:
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        parameters["outputVolume"] = outputModelNode
        parameters["referenceVolume"] = referenceVolumeNode
        parameters["interpolatorType"] = interpolator_type

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
        return outputModelNode
    
    except Exception as e:
        logger.exception(f"An error occurred: {e}")


def quick_visualize(image_array, cmap='gray', indices=None):
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 3:
        raise ValueError("image_array must be a 3D numpy array.")

    # Automatically select indices if not provided
    if indices is None:
        shape = image_array.shape
        indices = {
            'axial': [shape[0] // 4, shape[0] // 2, 3 * shape[0] // 4],
            'coronal': [shape[1] // 4, shape[1] // 2, 3 * shape[1] // 4],
            'sagittal': [shape[2] // 4, shape[2] // 2, 3 * shape[2] // 4],
        }

    plt.figure(figsize=(10, 10))

    # Axial slices
    for i, idx in enumerate(indices['axial'], 1):
        plt.subplot(3, 3, i)
        plt.imshow(image_array[idx, :, :], cmap=cmap)
        plt.title(f"Axial slice {idx}")

    # Coronal slices
    for i, idx in enumerate(indices['coronal'], 4):
        plt.subplot(3, 3, i)
        plt.imshow(image_array[:, idx, :], cmap=cmap)
        plt.title(f"Coronal slice {idx}")

    # Sagittal slices
    for i, idx in enumerate(indices['sagittal'], 7):
        plt.subplot(3, 3, i)
        plt.imshow(image_array[:, :, idx], cmap=cmap)
        plt.title(f"Sagittal slice {idx}")

    plt.tight_layout()
    plt.show()


def dice_similarity(mask1, mask2):
    """
    Calculate the Dice Similarity Index between two binary masks.

    Parameters
    ----------------
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns
    ----------------
        float: Dice Similarity Index.

    Raises
    -------
        ValueError: If the masks do not have the same shape.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape")

    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)

    if sum_masks == 0:
        return 1.0  # Both masks are empty

    return 2.0 * intersection / sum_masks