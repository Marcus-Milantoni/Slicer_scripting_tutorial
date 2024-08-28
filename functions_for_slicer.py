# Created by Marcus Milantoni and Edward Wang. This script contains basic functions for image processing in Slicer.

import slicer, vtk
from DICOMLib import DICOMUtils
import numpy as np
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

    Raises
    ------
    TypeError
        If the dicomDataDir is not a string.
    ValueError
        If the dicomDataDir is not a valid directory.
    """
    if not isinstance(dicomDataDir, str):
        raise TypeError("The dicomDataDir parameter must be a string.")
    if not os.path.isdir(dicomDataDir):
        raise ValueError("The dicomDataDir parameter must be a valid directory.")

    try:
        logger.info(f"Loading DICOM data from directory: {dicomDataDir}")
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
    This function returns the full size segmentation array from a given segment ID. If the segmentationNode is None or the referenceVolumeNode is none, the first node by class is used. Warning, if the segment ID is not in the segmentation node, this will cause an error.
    
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

    Raises
    ------
    TypeError
        If the segmentID is not a string.
        If the segmentationNode is not a vtkMRMLSegmentationNode or None.
        If the referenceVolumeNode is not a vtkMRMLVolumeNode or None.
    """
    if not isinstance(segmentID, str):
        raise TypeError("The segmentID parameter must be a string.")
    if segmentationNode is not None and not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be of type vtkMRMLSegmentationNode or None.")
    if referenceVolumeNode is not None and not isinstance(referenceVolumeNode, slicer.vtkMRMLVolumeNode):
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
    This function creates a volume node from a numpy array. A reference node must be provided to create the volume node.

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

    Raises
    ------
    TypeError
        If the volumeArray is not a numpy array.
        If the referenceNode is not a vtkMRMLScalarVolumeNode.
        If the volumeName is not a string.
    """
    if not isinstance(volumeArray, np.ndarray):
        raise TypeError("The volumeArray parameter must be a numpy array.")
    if not isinstance(referenceNode, slicer.vtkMRMLScalarVolumeNode):
        raise TypeError("The referenceNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(volumeName, str):
        raise TypeError("The volumeName parameter must be a string.")
    
    try:
        logger.info(f"Creating a new volume node.")
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


def add_segmentation_array_to_node(segmentationArray, segmentationNode, segmentName, referenceVolumeNode, color=None):
    """
    This function adds a segmentation from a numpy array to a node in the slicer scene. If no color is provided, a random color will be generated.

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

    Raises
    ------
    TypeError
        If the segmentationArray is not a numpy array.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the segmentName is not a string.
        If the referenceVolumeNode is not a vtkMRMLScalarVolumeNode.
        If the color is not a tuple.
    """
    if color is None:
        color = tuple(np.random.rand(3))

    if not isinstance(segmentationArray, np.ndarray):
        raise TypeError("The segmentationArray parameter must be a numpy array.")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(segmentName, str):
        raise TypeError("The segmentName parameter must be a string.")
    if not isinstance(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode):
        raise TypeError("The referenceVolumeNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(color, tuple):
        raise TypeError("The color parameter must be a tuple.")
    
    try:
        logger.info(f"Adding the segmentation to the segmentation node.")
        tempVolumeNode = create_volume_node(segmentationArray, referenceVolumeNode, "TempNode")
        tempImageData = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(tempVolumeNode)
        slicer.mrmlScene.RemoveNode(tempVolumeNode)
        logger.debug(f"Adding the segmentation to the segmentation node.")
        segmentID = segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(tempImageData, segmentName, color)

        return segmentID
    
    except Exception:
        logger.exception("An error occurred in add_segmentation_array_to_node")
        raise


def check_intersection_binary_mask(array1, array2, numVoxelsThreshold=1):
    """
    Checks if two binary masks intersect.

    Parameters
    ----------
    array1 : numpy.ndarray
             The first binary mask.
    array2 : numpy.ndarray
             The second binary mask.
    numVoxelsThreshold : int, default: 1
                           The number of voxels that must be intersecting for the function to return True.
               
    Returns
    -------
    bool
        True if the masks intersect, False if they do not intersect.
    
    Raises
    ------
    TypeError
        If the array1 is not a numpy array.
        If the array2 is not a numpy array.
        If the numVoxelsThreshold is not an integer.
    ValueError
        If the array1 and array2 do not have the same shape.
        If the numVoxelsThreshold is not greater than 0 or less than the total number of voxels in the array.
    """
    if not isinstance(array1, np.ndarray):
        raise TypeError("The array1 parameter must be a numpy array.")
    if not isinstance(array2, np.ndarray):
        raise TypeError("The array2 parameter must be a numpy array.")
    if not isinstance(numVoxelsThreshold, int):
        raise TypeError("The numVoxelsThreshold parameter must be an integer.")
    if not array1.shape == array2.shape:
        raise ValueError("The array1 and array2 parameters must have the same shape.")
    if not numVoxelsThreshold > 0 and numVoxelsThreshold <= array1.size:
        raise ValueError("The numVoxelsThreshold parameter must be greater than 0 and less than the total number of voxels in the array.")

    try:
        intersection = np.logical_and(array1, array2)
        if np.sum(intersection) >= numVoxelsThreshold:
            return True
        else:
            return False
    
    except Exception:
        logger.exception("An error occurred in check_intersection_binary_mask")
        raise


def margin_editor_effect(segmentName, newName, segmentationNode, volumeNode, operation='Grow', MarginSize=10.0):
    """
    This function dilates or shrinks a segment in a segmentation node. This function automatically copies the function before applying the margin effect (makes a new segment).

    Parameters
    ----------
    segmentName : str
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

    Raises
    ------
    TypeError
        If the segmentName is not a string.
        If the newName is not a string.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the operation is not a string.
        If the MarginSize is not a float.
    ValueError
        If the operation is not 'Grow' or 'Shrink'.
        If the MarginSize is not greater than 0.
    """
    if not isinstance(segmentName, str):
        raise TypeError("The segmentName parameter must be a string.")
    if not isinstance(newName, str):
        raise TypeError("The newName parameter must be a string.")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
        raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(operation, str):
        raise TypeError("The operation parameter must be a string.")
    if not isinstance(MarginSize, float):
        raise TypeError("The MarginSize parameter must be a float.")
    
    try:
        logger.info(f"applying the margin effect to the segment {segmentName}")
        logger.debug(f"Copying segment {segmentName} to {newName}")
        copy_segmentation(segmentationNode, segmentName, newName)

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
            raise ValueError("Invalid operation. Operation must be 'Grow' or 'Shrink.'")
        
        if (isinstance(MarginSize, float) or isinstance(MarginSize, int)) and MarginSize > 0:
            effect.setParameter("MarginSizeMm", MarginSize)
        else:
            raise ValueError("Invalid MarginSize. MarginSize must be a positive number.")
        effect.self().onApply() 

        return newSegmentID 

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

    Raises
    ------
    TypeError
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the segmentName is not a string.
    """
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(segmentName, str):
        raise TypeError("The segmentName parameter must be a string.")
    
    try:
        logger.info(f"Creating a blank segment with name {segmentName}")
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
    
    Raises
    ------
    TypeError
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the segmentName is not a string.
        If the newSegmentName is not a string.
    """
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
    if not isinstance(segmentName, str):
        raise TypeError("The segmentName parameter must be a string.")
    if not isinstance(newSegmentName, str):
        raise TypeError("The newSegmentName parameter must be a string.")
    
    try:
        logger.info(f"Copying segment {segmentName} to {newSegmentName}")
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

    Raises
    ------
    TypeError
        If the mask1 is not a numpy array.
        If the mask2 is not a numpy array.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    """
    if not isinstance(mask1, np.ndarray):
        raise TypeError("The mask1 parameter must be a numpy.ndarray.")
    if not isinstance(mask2, np.ndarray):
        raise TypeError("The mask2 parameter must be a numpy.ndarray.")
    
    logger.info(f"Combining masks")
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
    
    Raises
    ------
    TypeError
        If the mask1 is not a numpy array.
        If the mask2 is not a numpy array.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    ValueError
        If the mask1 and mask2 do not have the same shape
    """
    if not isinstance(mask1, np.ndarray):
        raise TypeError("The mask1 parameter must be a numpy.ndarray.")
    if not isinstance(mask2, np.ndarray):
        raise TypeError("The mask2 parameter must be a numpy.ndarray.")
    if not mask1.shape == mask2.shape:
        raise ValueError("The mask1 and mask2 parameters must have the same shape.")
    
    try:
        logger.info(f"Combining masks")
        combined_mask = np.bitwise_and(mask1, mask2)

    except Exception:
        logger.exception("An error occurred in bitwise_and_from_array")
        raise
            
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


def remove_mask_superior_to_slice(maskArray, sliceNumber, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Crops the input maskArray to the region superior of the specified slice number. This function can also add the cropped mask to a segmentation node.
        
    Parameters
    ----------  
    maskArray : numpy.ndarray
                  The binary mask that is getting cropped.
    sliceNumber : int
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
    
    Raises
    ------
    TypeError
        If the maskArray is not a numpy array.
        If the sliceNumber is not an integer.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    IndexError
        If the sliceNumber is less than 0 or greater than or equal to the number of slices in the mask
    """
    if not isinstance(maskArray, np.ndarray):
        logger.error("The maskArray parameter must be a numpy.ndarray.")
        raise TypeError("The maskArray parameter must be a numpy.ndarray.")
    if not isinstance(sliceNumber, int):
        logger.error("The sliceNumber parameter must be an integer.")
        raise TypeError("The sliceNumber parameter must be an integer.")
    if sliceNumber < 0 or sliceNumber >= maskArray.shape[0]:
        logger.error("The sliceNumber parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")
        raise IndexError("The sliceNumber parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")

    try:
        logger.info(f"Cropping the mask to the region superior to slice {sliceNumber}")
        main_copy = np.copy(maskArray)
        main_copy[sliceNumber + 1:] = 0
    
    except Exception:
        logger.exception("An error occurred in remove_mask_superior_to_slice")
        raise

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


def remove_mask_inferior_to_slice(maskArray, sliceNumber, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Crops the input maskArray to the region inferior to the specified slice number. This function can also add the cropped mask to a segmentation node.
        
    Parameters
    ----------  
    maskArray : numpy.ndarray
                  The binary mask that is getting cropped.
    sliceNumber : int
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

    Raises
    ------
    TypeError
        If the maskArray is not a numpy array.
        If the sliceNumber is not an integer.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    IndexError
        If the sliceNumber is less than 0 or greater than or equal to the number of slices in the mask.
    """
    if not isinstance(maskArray, np.ndarray):
        raise TypeError("The maskArray parameter must be a numpy.ndarray.")
    if not isinstance(sliceNumber, int):
        raise TypeError("The sliceNumber parameter must be an integer.")
    if sliceNumber < 0 or sliceNumber >= maskArray.shape[0]:
        raise IndexError("The sliceNumber parameter must be less than the number of slices in the maskArray and greater than or equal to 0.")

    try:
        logger.info(f"Cropping the mask to the region inferior to slice {sliceNumber}")
        main_copy = np.copy(maskArray)
        main_copy[:sliceNumber] = 0

    except Exception:
        logger.exception("An error occurred in remove_mask_inferior_to_slice")
        raise

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
                The numpy array with the projected segmentation.

    Raises
    ------
    TypeError
        If the maskArray is not a numpy array.
        If the numberOfSlicesToCombine is not an integer.
        If the numberOfSlicesToProject is not an integer.
        If the projectInferior is not a boolean.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    ValueError
        If the numberOfSlicesToCombine is not greater than 0. 
    """
    if not isinstance(maskArray, np.ndarray):
        raise TypeError("The maskArray parameter must be a numpy.ndarray.")
    if not isinstance(numberOfSlicesToCombine, int):
        raise TypeError("The numberOfSlicesToCombine parameter must be an integer.")
    if not numberOfSlicesToCombine > 0:
        raise ValueError("The numberOfSlicesToCombine parameter must be greater than 0")
    if not isinstance(numberOfSlicesToProject, int):
        raise TypeError("The numberOfSlicesToProject parameter must be an integer.")
    
    slices_w_mask = []
    main_copy = np.copy(maskArray)
    for index, slice in enumerate(maskArray):
        if 1 in slice:
            slices_w_mask.append(index)

    try:
        logger.info(f"Projecting the segmentation vertically.")
        if projectInferior:
            logger.debug(f"Projecting inferiorly.")
            bottom_slice = min(slices_w_mask)

            logger.debug(f"Calculating the last slices.")
            last_slices = main_copy[bottom_slice : bottom_slice + numberOfSlicesToCombine]
            result = last_slices[0]
            all_slices_to_change = np.arange(bottom_slice - numberOfSlicesToProject, bottom_slice + numberOfSlicesToProject)

            logger.debug(f"Creating the array.")
            for slice in last_slices[1:]:
                result = np.bitwise_or(result, slice)        
            
            for index in all_slices_to_change:
                main_copy[index] = result

            logger.info(f"Finished creating the array.")

        if not projectInferior:
            logger.debug(f"Projecting superiorly.")
            top_slice = max(slices_w_mask)
            
            logger.debug(f"Calculating the first slices.")
            first_slices = main_copy[top_slice - numberOfSlicesToCombine + 1 : top_slice + 1]
            result = first_slices[0]
            all_slices_to_change = np.arange(top_slice - numberOfSlicesToCombine + 1, top_slice + numberOfSlicesToProject + 1)

            logger.debug(f"Creating the array.")
            for slice in first_slices[1:]:
                result = np.bitwise_or(result, slice)        

            for index in all_slices_to_change:
                main_copy[index] = result
            
            logger.info(f"Finished creating the array.")

    except Exception as e:
        logger.exception(e)
        raise

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


def create_binary_mask_between(binaryMaskArrayLeft, binaryMaskArrayRight, fromMedial=True, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Creates a binary mask between two binary masks. The function can create the binary mask from the medial or lateral direction. The function can also add the binary mask to a segmentation node.
    
    Parameters
    ----------
    binaryMaskArrayLeft : numpy.ndarray
                The binary mask array to the left.
    binaryMaskArrayRight : numpy.ndarray
                The binary mask array to the right.
    fromMedial : bool, default: True
                Specification to create the binary mask from the medial direction.
    addSegmentationToNode : bool, default: False
                Specification to add the binary mask to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to add the binary mask to.
    volumeNode : vtkMRMLScalarVolumeNode, default: None
                The volume node that the segmentation node is based on.
    segmentName : str, default: None
                The name of the segment to add the binary mask to.
                
    Returns
    -------
    numpy.ndarray
        The binary mask between the two binary masks.
    
    Raises
    ------
    TypeError
        If the binaryMaskArrayLeft is not a numpy array.
        If the binaryMaskArrayRight is not a numpy array.
        If the fromMedial is not a boolean.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    """
    if not isinstance(binaryMaskArrayLeft, np.ndarray):
        raise TypeError("The binaryMaskArrayLeft parameter must be a numpy.ndarray.")
    if not isinstance(binaryMaskArrayRight, np.ndarray):
        raise TypeError("The binaryMaskArrayRight parameter must be a numpy.ndarray.")
    if not isinstance(fromMedial, bool):
        raise TypeError("The fromMedial parameter must be a boolean.")

    try:
        logger.info(f"Creating a binary mask between the two binary masks.")
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

    except Exception as e:
        logger.exception(e)
        raise

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(segmentName, str):
            raise TypeError("The segmentName parameter must be a string.")

        logger.debug("Adding the segmentation to the segmentation node.")
        add_segmentation_array_to_node(segmentationNode, mask_w_ones, segmentName, volumeNode)
    
    return mask_w_ones


def crop_anterior_from_mask(binaryMaskToCrop, referenceMask, fromAnterior=True, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
    """
    Crops a binary mask from the anterior. This function uses the reference mask to determine the most anterior slice. The function can also crop the binary mask from the posterior.
    
    Parameters
    ----------
    binaryMaskToCrop : numpy.ndarray
                The binary mask to be cropped.
    referenceMask : numpy.ndarray
                The reference mask to be used for cropping.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    addSegmentationToNode : bool, default: False
                Specification to add the cropped binary mask to a segmentation node.
    segmentationNode : vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to.
    volumeNode : vtkMRMLScalarVolumeNode, default: None
                The volume node to add the cropped binary mask to.
    segmentName : str, default: None
                The name of the segment to add the cropped binary mask to.
                       
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.

    Raises
    ------
    TypeError
        If the binaryMaskToCrop is not a numpy array.
        If the referenceMask is not a numpy array.
        If the fromAnterior is not a boolean.
        If the segmentationNode is not a vtkMRMLSegmentationNode.
        If the volumeNode is not a vtkMRMLScalarVolumeNode.
        If the segmentName is not a string.
    ValueError
        If the binaryMaskToCrop and referenceMask do not have the same shape.
    """
    if not isinstance(binaryMaskToCrop, np.ndarray):
        raise TypeError("The binaryMaskToCrop parameter must be a numpy.ndarray.")
    if not isinstance(referenceMask, np.ndarray):
        raise TypeError("The referenceMask parameter must be a numpy.ndarray.")
    if not isinstance(fromAnterior, bool):
        raise TypeError("The fromAnterior parameter must be a boolean.")
    if not binaryMaskToCrop.shape == referenceMask.shape:
        raise ValueError("The binaryMaskToCrop and referenceMask parameters must have the same shape.")

    try:
        logger.info("Cropping the binary mask from the reference mask.")
        logger.debug("Making a copy of the masks.")
        binary_mask = np.copy(binaryMaskToCrop)
        referenceMask_copy = np.copy(referenceMask)

        if fromAnterior:
            logger.debug("Cropping the binary mask from the anterior.")
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

    except Exception as e:
        logger.exception(e)

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(segmentName, str):
            raise TypeError("The segmentName parameter must be a string.")

        logger.debug("Adding the segmentation to the segmentation node.")
        add_segmentation_array_to_node(segmentationNode, binary_mask, segmentName, volumeNode)
    
    return binary_mask


def remove_multiple_rois_from_mask(binary_mask_array, tupleOfMasksToRemove, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    Removes a region of interest from a binary mask.
    
    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    tupleOfMasksToRemove : tuple[numpy.ndarray]
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

    Raises
    ------
    TypeError
        If the binary_mask_array is not a numpy array.
        If the tupleOfMasksToRemove is not a tuple.
        If the tupleOfMasksToRemove contains an element that is not a numpy array.
        If the tupleOfMasksToRemove contains an element that does not have the same shape as the binary_mask_array.
        If the add_segmentation_to_node is not a boolean.
    ValueError
        If the tupleOfMasksToRemove contains an element that does not have the same shape as the binary_mask_array.
    """
    if not isinstance(binary_mask_array, np.ndarray):
        raise TypeError("The binary_mask_array parameter must be a numpy.ndarray.")
    if not isinstance(tupleOfMasksToRemove, tuple):
        raise TypeError("The tupleOfMasksToRemove parameter must be a tuple.")
    if not all(isinstance(mask, np.ndarray) for mask in tupleOfMasksToRemove):
        raise TypeError("The tupleOfMasksToRemove parameter must contain only numpy.ndarrays.")
    if not all(mask.shape == binary_mask_array.shape for mask in tupleOfMasksToRemove):
        raise ValueError("The tupleOfMasksToRemove parameter must contain numpy.ndarrays with the same shape as the binary_mask_array parameter.")
    if not isinstance(add_segmentation_to_node, bool):
        raise TypeError("The add_segmentation_to_node parameter must be a boolean.")
    
    try:
        binary_mask = np.copy(binary_mask_array)
        for mask in tupleOfMasksToRemove:
            binary_mask = binary_mask - mask

    except Exception:
        logger.exception("An error occurred in remove_multiple_rois_from_mask")
        raise

    if add_segmentation_to_node:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLVolumeNode.")
        if not isinstance(volume_name, str):
            raise TypeError("The volume_name parameter must be a string.")
    
    return binary_mask 


def save_image_from_DICOM_database(outputFolder, nodeTypesToSave = tuple(["vtkMRMLScalarVolumeNode"]), outputFileType=".nii"):
    """
    Save the images from the DICOM database to the output folder. The function saves the images as .nii files by default.

    Parameters
    ----------
    outputFolder : str
                The folder to save the images to.
    nodeTypesToSave : tuple, default: tuple(["vtkMRMLScalarVolumeNode"])
                The types of nodes to save.
    outputFileType : str, default: ".nii"
                The file type to save the images as. The default is ".nii". Other options include ".nrrd".

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the outputFolder is not a string.
        If the nodeTypesToSave is not a tuple.
        If the nodeTypesToSave contains an element that is not a string.
        If the outputFileType is not a string.
    ValueError
        If the outputFileType is not ".nii" or ".nrrd".
    """
    if not isinstance(outputFolder, str):
        raise TypeError("The outputFolder parameter must be a string.")
    if not isinstance(nodeTypesToSave, tuple):
        raise TypeError("The nodeTypesToSave parameter must be a tuple.")
    if not all(isinstance(nodeType, str) for nodeType in nodeTypesToSave):
        raise TypeError("The nodeTypesToSave parameter must contain only strings.")
    if not isinstance(outputFileType, str):
        raise TypeError("The outputFileType parameter must be a string.")
    if not outputFileType in [".nii", ".nrrd"]:
        raise ValueError("The outputFileType parameter must be either '.nii' or '.nrrd'.")
    
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    logger.info(f"Saving images from the DICOM database to {outputFolder}")
    patientUIDs = slicer.dicomDatabase.patients()
    for patientUID in patientUIDs:
        logger.debug(f'Loading patient with UID {patientUID}')
        loadedNodeIDs = DICOMUtils.loadPatientByUID(patientUID)
        for loadedNodeID in loadedNodeIDs:
            logger.debug(f'Loaded node with ID {loadedNodeID}')
            node = slicer.mrmlScene.GetNodeByID(loadedNodeID)
            if not node:
                continue
            if not any(node.IsA(nodeType) for nodeType in nodeTypesToSave):
                continue
            logger.debug(f'Found node of type {node.GetClassName()}')
            logger.debug(f'creating the filename')
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            seriesItem = shNode.GetItemByDataNode(node)
            studyItem = shNode.GetItemParent(seriesItem)
            patientItem = shNode.GetItemParent(studyItem)
            filename = shNode.GetItemAttribute(patientItem, 'DICOM.PatientID')
            filename += '_' + shNode.GetItemAttribute(seriesItem, 'DICOM.Modality')
            filename = slicer.app.ioManager().forceFileNameValidCharacters(filename) + outputFileType
            # Save node
            logger.debug(f'Write {node.GetName()} to {os.path.join(outputFolder, filename)}') 
            slicer.util.saveNode(node, os.path.join(outputFolder, filename))


def save_image_from_scalar_volume_node(outputFolder, scalarVolumeNode, outputFileType=".nii", additionalSaveInfo=None):
    """
    Save the image from the scalar volume node to the output folder. The function saves the image as a .nii file by default.

    Parameters
    ----------
    outputFolder : str
                The folder to save the image to.
    scalarVolumeNode : vtkMRMLScalarVolumeNode
                The scalar volume node to save.
    outputFileType : str, default: ".nii"
                The file type to save the image as. The default is ".nii". Other options include ".nrrd".

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the outputFolder is not a string.
        If the scalarVolumeNode is not a vtkMRMLScalarVolumeNode.
        If the outputFileType is not a string.
    """
    if not isinstance(outputFolder, str):
        raise TypeError("The outputFolder parameter must be a string.")
    if not isinstance(scalarVolumeNode, slicer.vtkMRMLScalarVolumeNode):
        raise TypeError("The scalarVolumeNode parameter must be a vtkMRMLScalarVolumeNode.")
    if not isinstance(outputFileType, str):
        raise TypeError("The outputFileType parameter must be a string.")
    if not outputFileType in [".nii", ".nrrd"]:
        raise ValueError("The outputFileType parameter must be either '.nii' or '.nrrd'.")
    
    if not os.path.exists(outputFolder):
        logger.debug(f"Creating the output folder {outputFolder}")
        os.makedirs(outputFolder)
    
    logger.info(f"Saving the image from the scalar volume node to {outputFolder}")
    logger.debug(f"Creating the default storage node")
    saveStorageNode = scalarVolumeNode.CreateDefaultStorageNode()
    if additionalSaveInfo is not None:
        logger.debug(f"Setting the file name with the additional save info")
        saveStorageNode.SetFileName(os.path.join(outputFolder, additionalSaveInfo + outputFileType))
    else:
        logger.debug(f"Setting the file name with the scalar volume node name")
        saveStorageNode.SetFileName(os.path.join(outputFolder, scalarVolumeNode.GetName() + outputFileType))
    logger.debug(f"Writing the data to the storage node")
    saveStorageNode.WriteData(scalarVolumeNode)


def save_rtstructs_as_nii(outputFolder, segmentationNode, segmentationsToSave):
    """
    Save the segmentation to the output folder as a .nii file.
    
    Parameters
    ----------
    output_folder : str
                The folder to save the .nii file to.
    segmentationsToSave : tuple[str]
                The segmentations to save.
                
    Returns
    -------
    None

    raises
    ------
    TypeError
        If the outputFolder is not a string.
        If the segmentationNode is not a slicer.vtkMRMLSegmentationNode
        If the segmentationsToSave is not a tuple.
        If the segmentationsToSave contains an element that is not a string.
    """

    if not isinstance(outputFolder, str):
        raise TypeError("The outputFolder parameter must be a string.")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode")
    if not isinstance(segmentationsToSave, tuple):
        raise TypeError("The nodeTypesToSave parameter must be a tuple.")
    if not all(isinstance(nodeType, str) for nodeType in segmentationsToSave):
        raise TypeError("The nodeTypesToSave parameter must contain only strings.")
    
    if not os.path.exists(outputFolder):
        logger.debug(f"Creating the output folder {outputFolder}")
        os.makedirs(outputFolder)

    # Create a vtkStringArray outside the loop
    segmentIds = vtk.vtkStringArray()

    for segment in segmentationsToSave:
        segmentation_id_to_save = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment)
        # Add each segment ID to the vtkStringArray
        segmentIds.InsertNextValue(segmentation_id_to_save)

    # Call the export function once, after the loop
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsBinaryLabelmapRepresentationToFiles(outputFolder, segmentationNode, segmentIds, ".nii", False)


def find_max_distance_in_2D(binaryMask):
    """
    Finds the maximum distance between two pixels in a 2D binary mask. If you want to find the max distance in a plane of a binary mask, 
    
    Parameters
    ----------
    binaryMask : numpy.ndarray
                The binary mask to find the maximum distance in. Must be a 2D array.
    
    Returns
    -------
    float
        The maximum distance between two pixels in the binary mask. Zero if no mask is found.

    Raises
    ------
    TypeError
        If the binaryMask is not a np.ndarray
    """
    if not isinstance(binaryMask, np.ndarray):
        raise TypeError("The parameter binaryMask must be a np.ndarray")
    
    try:
        # get the location of all the ones
        coords = np.where(binaryMask == 1)
        if not coords[0].size or not coords[1].size:
            return 0  # Return 0 if no pixels are found

        min_x, max_x = np.min(coords[0]), np.max(coords[0])
        min_y, max_y = np.min(coords[1]), np.max(coords[1])

        max_dist = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

        return max_dist
    
    except Exception:
        logger.exception("An error occurred in find_max_distance_in_2D")
        raise


def crop_inferior_to_slice(binaryMaskArray, sliceNumber, includeSlice=True):
    """
    Crops the given binary mask array to remove all slices inferior to the specified slice number.
    This function allows for selective cropping where the user can choose to include or exclude the slice
    at the specified slice number in the cropped output. The operation is performed in-place, modifying
    the input array.

    Parameters
    ----------
    binaryMaskArray : numpy.ndarray
        The binary mask to be cropped. It should be a 3D array where each slice along the first dimension
        represents a 2D binary mask.
    sliceNumber : int
        The slice number from which the cropping should start. Slices inferior to this number will be
        modified based on the value of `include_slice`.
    includeSlice : bool, optional
        A flag to determine whether the slice specified by `sliceNumber` should be included in the
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
    TypeError
        If binaryMaskArray is not a np.ndarray
        If sliceNumber is not an int
        If includeSlice is not a bool
    """
    if not isinstance(binaryMaskArray, np.ndarray):
        raise TypeError("The parameter binaryMaskArray must be a np.ndarray")
    if not isinstance(sliceNumber, int):
        raise TypeError("The parameter sliceNumber must be an int")
    if not isinstance(includeSlice, bool):
        raise TypeError("The parameter includeSlice must be a bool")
    
    # fix an exception here
    ### if not 
    
    try:
        binary_mask = np.copy(binaryMaskArray) #Make a copy of the mask array as we are working in place
        
        if includeSlice:
            for index, slice in enumerate(binary_mask):
                if index >= sliceNumber:
                    slice[slice == 1] = 0

        elif not includeSlice:
            for index, slice in enumerate(binary_mask):
                if index > sliceNumber:
                    slice[slice == 1] = 0

        return(binary_mask)

    except Exception:
        logger.exception("An error occurred in crop_inferior_to_slice")
        raise


def crop_superior_to_slice(main_mask, sliceNumber):
    """
    Crops the given binary mask array to keep only the region superior to the specified slice number.
    This function modifies the input array in-place, setting all slices inferior to the specified slice
    number to zero, effectively removing them from the binary mask.

    Parameters
    ----------
    main_mask : numpy.ndarray
        The binary mask to be cropped. It should be a 3D array where each slice along the first dimension
        represents a 2D binary mask.
    sliceNumber : int
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
        If `main_mask` is not a numpy.ndarray or `sliceNumber` is not an integer, an exception is raised
        with an appropriate error message.
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(sliceNumber, int):
            main_copy = np.copy(main_mask)
            slice_shape = main_mask.shape[1:]
            dtype_img = main_mask.dtype

            for index, slice in enumerate(main_copy):
                if index <= sliceNumber:
                    np.putmask(slice, np.ones(slice_shape, dtype=dtype_img), 0)
            
            return main_copy
        
        else:
            if not isinstance(main_mask, np.ndarray):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")
            if not isinstance(sliceNumber, int):
                raise Exception("The sliceNumber parameter must be an integer.")
    
    except Exception as e:
        logger.exception(e)


def crop_posterior_from_sliceNumber(binary_mask_array, sliceNumber, tuple_of_masks, fromAnterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    This function crops a binary mask posteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binary_mask_array.

    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    sliceNumber : int
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
        if isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int) and isinstance(tuple_of_masks, tuple) and isinstance(fromAnterior, bool) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
    
            if fromAnterior:
                most_anterior =  []

                for mask in tuple_of_masks:
                    if isinstance(mask, np.ndarray):
                        slice = mask[sliceNumber]
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
                        slice = mask[sliceNumber]
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
            if isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int) and isinstance(tuple_of_masks, tuple):
                raise Exception("The fromAnterior parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int):
                raise Exception("The tuple_of_masks parameter must be a tuple.")
            elif isinstance(tuple_of_masks, tuple):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_anterior_from_sliceNumber(binary_mask_array, sliceNumber, tuple_of_masks, fromAnterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    This function crops a binary mask anteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binary_mask_array.
    
    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    sliceNumber : int
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
        if isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int) and isinstance(tuple_of_masks, tuple) and isinstance(fromAnterior, bool) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
    
            if fromAnterior:
                most_anterior =  []

                for mask in tuple_of_masks:
                    if isinstance(mask, np.ndarray):
                        slice = mask[sliceNumber]
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
                        slice = mask[sliceNumber]
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
            if isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int) and isinstance(tuple_of_masks, tuple):
                raise Exception("The fromAnterior parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int):
                raise Exception("The tuple_of_masks parameter must be a tuple.")
            elif isinstance(tuple_of_masks, tuple):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_posterior_from_sliceNumber(binary_mask_array, sliceNumber, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    This function crops a binary mask posteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binary_mask_array.
    
    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask to be cropped.
    sliceNumber : int
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
        if isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
            binary_mask[:,:sliceNumber,:] = 0

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
            if isinstance(binary_mask_array, np.ndarray) and isinstance(sliceNumber, int):
                raise Exception("The add_segmentation_to_node parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray):
                raise Exception("The sliceNumber parameter must be an integer.")
            
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