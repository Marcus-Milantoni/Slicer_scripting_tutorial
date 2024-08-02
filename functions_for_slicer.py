# Created by Edward Wang and Marcus Milantoni

import slicer, vtk
from DICOMLib import DICOMUtils
import numpy as np
from numpy import random as rnd
import os
import logging
import matplotlib.pyplot as plt


logger = logging.getLogger(slicer.app.applicationLogic().GetClassName())
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_DICOM(dicomDataDir):
    """
    This function loads DICOM data into Slicer.

    Parameters
    ----------
    dicomDataDir : str
                The directory containing the DICOM data to load.
    
    Returns
    -------
    None
    """
    try:
        if isinstance(dicomDataDir, str) and os.path.isdir(dicomDataDir):
            loadedNodeIDs = []  # this list will contain the list of all loaded node IDs

            with DICOMUtils.TemporaryDICOMDatabase() as db:
                DICOMUtils.importDicom(dicomDataDir, db)
                patientUIDs = db.patients()
                for patientUID in patientUIDs:
                    loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
        else:
            raise Exception ("The dicomDataDir parameter must be a string.")

        return loadedNodeIDs

    except Exception as e:
        logger.exception(e)


def getFullSizeSegmentation(segmentId):
    """
    This function returns the full size segmentation from a segment ID.
    
    Parameters
    ----------
    segmentId : str
                The segment ID to get the full size segmentation from.
    
    Returns
    -------
    numpy.ndarray
        The full size segmentation.
    """
    segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
    referenceVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    segarr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, referenceVolumeNode)
    
    return segarr


def createVolumeNode(doseVolume, referenceNode, volumeName):
    """
    This function creates a volume node from a numpy array.

    Parameters
    ----------
    doseVolume : numpy.ndarray
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
    doseNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', volumeName)
    doseNode.CopyOrientation(referenceNode)
    doseNode.SetSpacing(referenceNode.GetSpacing())
    doseNode.CreateDefaultDisplayNodes()
    displayNode = doseNode.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRainbow')
    slicer.util.updateVolumeFromArray(doseNode, doseVolume)

    return doseNode


def addSegmentationToNodeFromNumpyArr(segmentationNode, numpyArr, name, referenceVolumeNode, color=rnd.rand(3)):
    """
    This function adds a segmentation to a node from a numpy array.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the segmentation to.
    numpyArr : numpy.ndarray
                The numpy array to add to the segmentation node.
    name : str
                The name of the segmentation to add.
    referenceVolumeNode : vtkMRMLScalarVolumeNode
                The reference volume node to add the segmentation to.
    color : tuple, default: rnd.rand(3)
                The color of the segmentation to add.
        
    Returns
    -------
    None
    """
    tempVolumeNode = createVolumeNode(numpyArr, referenceVolumeNode, "TempNode")
    tempImageData = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(tempVolumeNode)
    slicer.mrmlScene.RemoveNode(tempVolumeNode)
    segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(tempImageData, name, color)


def crop_anterior(binary_mask_to_be_cropped, reference_mask, from_anterior=True):
    """
    Crops a binary mask from the anterior.
    
    Parameters
    ----------
    binary_mask_to_be_cropped : numpy.ndarray
                The binary mask to be cropped.
    reference_mask : numpy.ndarray
                The reference mask to be used for cropping.
    from_anterior : bool, default: True
                Specification to crop the binary mask from the anterior.
                
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask_to_be_cropped, np.ndarray) and isinstance(reference_mask, np.ndarray) and (from_anterior == True or from_anterior == False):
            binary_mask = np.copy(binary_mask_to_be_cropped)
            reference_mask_copy = np.copy(reference_mask)

            if from_anterior:
                most_anterior =  []

                for slice in reference_mask_copy:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        if 1 in row:
                            most_anterior.append(index)
                            break

                binary_mask[:, min(most_anterior):,:] = 0


            elif from_anterior == False:
                most_posterior =  []
                
                for slice in reference_mask_copy:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        most_posterior_in_slice = 0
                        if 1 in row:
                            if index > most_posterior_in_slice:
                                most_posterior_in_slice = index
                        most_posterior.append(most_posterior_in_slice)

                binary_mask[:,: max(most_posterior),:] = 0

            return(binary_mask)
    
        else:
            if isinstance(binary_mask_to_be_cropped, np.ndarray) and isinstance(reference_mask, np.ndarray):
                raise Exception("The from_anterior parameter must be a boolean.")
            elif isinstance(binary_mask_to_be_cropped, np.ndarray):
                raise Exception("The reference_mask parameter must be a numpy.ndarray.")
            elif isinstance(reference_mask, np.ndarray):
                raise Exception("The binary_mask_to_be_cropped parameter must be a numpy.ndarray.")

    except Exception as e:
        logger.exception(e)


def crop_posterior(binary_mask_to_be_cropped, reference_tuple, from_anterior=True):
    """
    Crops a binary mask from the posterior.
    
    Parameters
    ----------
    binary_mask_to_be_cropped : numpy.ndarray
                The binary mask to be cropped.
    reference_tuple : Tuple[np.ndarray]
                The reference mask to be used for cropping.
    from_anterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask_to_be_cropped, np.ndarray) and isinstance(reference_tuple, tuple) and (from_anterior == True or from_anterior == False):
            binary_mask = np.copy(binary_mask_to_be_cropped)

            if from_anterior:
                most_anterior =  []

                for contour in reference_tuple:
                    if isinstance(contour, np.ndarray):
                        for slice in contour:
                            for index in range(np.shape(slice)[0]):
                                row = slice[index,:]
                                if 1 in row:
                                    most_anterior.append(index)
                                    break
                    
                    else:
                        raise Exception(f"The reference_tuple parameter must be a tuple of numpy.ndarrays.")
                binary_mask[:,: min(most_anterior),:] = 0


            elif from_anterior == False:
                most_posterior =  []
                
                for contour in reference_tuple:
                    for slice in contour:
                        if isinstance(slice, np.ndarray):
                            for index in range(np.shape(slice)[0]):
                                row = slice[index,:]
                                most_posterior_in_slice = 0
                                if 1 in row:
                                    if index > most_posterior_in_slice:
                                        most_posterior_in_slice = index
                                most_posterior.append(most_posterior_in_slice)
                            
                    else:
                        raise Exception(f"The reference_tuple parameter must be a tuple of numpy.ndarrays.")
                binary_mask[:, max(most_posterior):,:] = 0

            return(binary_mask)
    
        else:
            if isinstance(binary_mask_to_be_cropped, np.ndarray) and isinstance(reference_tuple, tuple):
                raise Exception("The from_anterior parameter must be a boolean.")
            elif isinstance(binary_mask_to_be_cropped, np.ndarray):
                raise Exception("The reference_tuple parameter must be a tuple.")
            elif isinstance(reference_tuple, tuple):
                raise Exception("The binary_mask_to_be_cropped parameter must be a numpy.ndarray.")

    except Exception as e:
        logger.exception(e)

def crop_inferior(main_mask, second_masks, from_top=True):
    """
    Keeps a binary mask in the region inferior to the secondary binary mask. If from_top=True, it will crop from the most superior slice of the second binary 
    mask (inclusive). If from_top=Faslse, it will crop from the most inferior slice of the second binary mask (not inclusive).

    Parameters
    ----------
    main_mask : numpy.ndarray
                  The binary mask that is getting cropped.
    second_masks : Tuple[numpy.ndarray]
                    The binary masks that are selecting the bottom slice.
    from_top : bool, default: True
               Specification to take crop from top or bottom of the second_volume.
               
    Returns
    -------
    numpy.ndarray
        The main volume cropped. 
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(second_masks, tuple) and (from_top == True or from_top == False):
            main_copy = np.copy(main_mask)
            slices_w_mask = []

            if from_top:
                for outer_index, second_mask in enumerate(second_masks):
                    if isinstance(second_mask, np.ndarray):
                        for inner_index, slice in enumerate(second_mask):
                            if 1 in slice:
                                slices_w_mask.append(inner_index)
                    else:
                        raise Exception(f"The second_masks parameter must be a tuple of numpy.ndarrays. The array at index {outer_index} is not a numpy.ndarray.")
                top_slice = max(slices_w_mask)
                for index, slice in enumerate(main_copy):
                    if index > top_slice:
                        slice[slice == 1] = 0

            elif not from_top:
                for outer_index, second_mask in enumerate(second_masks):
                    if isinstance(second_mask, np.ndarray):
                        for inner_index, slice in enumerate(second_mask):
                            if 1 in slice:
                                slices_w_mask.append(inner_index)
                    else:
                        raise Exception(f"The second_masks parameter must be a tuple of numpy.ndarrays. The array at index {outer_index} is not a numpy.ndarray.")
                bottom_slice = min(slices_w_mask)
                for index, slice in enumerate(main_copy):
                    if index >= bottom_slice:
                        slice[slice == 1] = 0

            return(main_copy)
        
        else:
            if isinstance(main_mask, np.ndarray) and isinstance(second_masks, tuple):
                raise Exception("The from_top parameter must be a boolean.")
            elif isinstance(main_mask, np.ndarray):
                raise Exception("The second_masks parameter must be a tuple.")
            elif isinstance(second_masks, tuple):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")

    except Exception as e:
        logger.exception(e)


def crop_inferior_lateral_bound(main_mask, second_masks, left_bound, right_bound, from_top=True):
    """
    Keeps a binary mask in the region inferior to the secondary binary mask. If from_top=True, it will crop from the most superior slice of the second binary 
    mask (inclusive). If from_top=Faslse, it will crop from the most inferior slice of the second binary mask (not inclusive).

    Parameters
    ----------
    main_mask : numpy.ndarray
                  The binary mask that is getting cropped.
    second_masks : Tuple[numpy.ndarray]
                    The binary masks that are selecting the bottom slice.
    left_bound : int
                The left bound of the region to crop.
    right_bound : int
                The right bound of the region to crop.
    from_top : bool, default: True
               Specification to take crop from top or bottom of the second_volume.
               
    Returns
    -------
    numpy.ndarray
        The main volume cropped. 
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(second_masks, tuple) and (from_top == True or from_top == False) and isinstance(left_bound, int) and isinstance(right_bound, int):
            main_copy = np.copy(main_mask)
            slices_w_mask = []

            if from_top:
                for outer_index, second_mask in enumerate(second_masks):
                    if isinstance(second_mask, np.ndarray):
                        for inner_index, slice in enumerate(second_mask):
                            if 1 in slice[:, left_bound:right_bound]:
                                slices_w_mask.append(inner_index)
                    else:
                        raise Exception(f"The second_masks parameter must be a tuple of numpy.ndarrays. The array at index {outer_index} is not a numpy.ndarray.")
                top_slice = max(slices_w_mask)
                for index, slice in enumerate(main_copy):
                    if index > top_slice:
                        slice[slice == 1] = 0

            elif not from_top:
                for outer_index, second_mask in enumerate(second_masks):
                    if isinstance(second_mask, np.ndarray):
                        for inner_index, slice in enumerate(second_mask):
                            if 1 in slice[:, left_bound:right_bound]:
                                slices_w_mask.append(inner_index)
                    else:
                        raise Exception(f"The second_masks parameter must be a tuple of numpy.ndarrays. The array at index {outer_index} is not a numpy.ndarray.")
                bottom_slice = min(slices_w_mask)
                for index, slice in enumerate(main_copy):
                    if index >= bottom_slice:
                        slice[slice == 1] = 0

            return(main_copy)
        
        else:
            if isinstance(main_mask, np.ndarray) and isinstance(second_masks, tuple):
                raise Exception("The from_top parameter must be a boolean.")
            elif isinstance(main_mask, np.ndarray):
                raise Exception("The second_masks parameter must be a tuple.")
            elif isinstance(second_masks, tuple):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")

    except Exception as e:
        logger.exception(e)


def crop_superior(main_mask, second_masks, from_top=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
    """
    Keeps a binary mask in the region superior to the secondary binary mask. If from_top=True, it will crop from the most superior slice of the second binary 
    mask (not inclusive). If from_top=Faslse, it will crop from the most inferior slice of the second binary mask(inclusive).

    Parameters
    ----------
    main_masks : numpy.ndarray
                  The binary mask that is getting cropped.
    second_masks : Tuple[numpy.ndarray]
                    The binary masks that are selecting the bottom slice.
        from_top : bool, default: True
               Specification to take crop from top or bottom of the second_volume.
               
    Returns
    -------
    numpy.ndarray
        The main volume cropped.
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(second_masks, tuple) and (from_top == True or from_top == False):
            main_copy = np.copy(main_mask)
            slices_w_mask = []

            if from_top:
                for outer_index, second_mask in enumerate(second_masks):
                    if isinstance(second_mask, np.ndarray):
                        for inner_index, slice in enumerate(second_mask):
                            if 1 in slice:
                                slices_w_mask.append(inner_index)
                    else:
                        raise Exception(f"The second_masks parameter must be a tuple of numpy.ndarrays. The array at index {outer_index} is not a numpy.ndarray.")
                top_slice = max(slices_w_mask)
                for index, slice in enumerate(main_copy):
                    if index-1 < top_slice:
                        slice[slice == 1] = 0

            if not from_top:
                for outer_index, second_mask in enumerate(second_masks):
                    if isinstance(second_mask, np.ndarray):
                        for inner_index, slice in enumerate(second_mask):
                            if 1 in slice:
                                slices_w_mask.append(inner_index)
                    else:
                        raise Exception(f"The second_masks parameter must be a tuple of numpy.ndarrays. The array at index {outer_index} is not a numpy.ndarray.")
                bottom_slice = min(slices_w_mask)
                for index, slice in enumerate(main_copy):
                    if index <= bottom_slice:
                        slice[slice == 1] = 0

            if add_segmentation_to_node:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode) and isinstance(volume_name, str):
                    addSegmentationToNodeFromNumpyArr(segmentationNode, main_copy, volume_name, volumeNode)
                else:
                    raise Exception("The segmentationNode parameter must be a vtkMRMLSegmentationNode, the volumeNode parameter must be a vtkMRMLScalarVolumeNode, and the volume_name parameter must be a string.")

            return(main_copy)
        
        else:
            if isinstance(main_mask, np.ndarray) and isinstance(second_masks, tuple):
                raise Exception("The from_top parameter must be a boolean.")
            elif isinstance(main_mask, np.ndarray):
                raise Exception("The second_masks parameter must be a tuple.")
            elif isinstance(second_masks, tuple):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")
    
    except Exception as e:
        logger.exception(e)


def check_intersection(mask_1, mask_2, num_voxels_threshold=1):
    """
    Checks if two binary masks intersect.

    Parameters
    ----------
    mask_1 : numpy.ndarray
             The first binary mask.
    mask_2 : numpy.ndarray
             The second binary mask.
    num_voxels_threshold : int, default: 1
                           The number of voxels that must be intersecting for the function to return True.
               
    Returns
    -------
    bool
        True if the masks intersect, False if they do not intersect.
    """
    intersection = np.logical_and(mask_1, mask_2)
    if np.sum(intersection) >= num_voxels_threshold:
        return(True)
    else:
        return(False)
    

def Margin_editor(input_Id, new_ID, segmentationNode, volumeNode, operation='Grow', MarginSize=10.0):
    """
    This function dilates or shrinks a segment in a segmentation node. This function automatically copys the function and crops the segment in the vertical direction.

    Parameters
    ----------
    inputId : str
                The name of the segment to dilate or shrink.
    new_ID : str
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
    None
    """
    copy_segmentation(segmentationNode, input_Id, new_ID)

    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(volumeNode)
    # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
    segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
    # Get the segment IDs
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(new_ID)
    segmentEditorNode.SetSelectedSegmentID(segid_src)


    segmentEditorWidget.setActiveEffectByName("Margin")
    effect = segmentEditorWidget.activeEffect()
    try:
        if operation == 'Grow' or operation == 'Shrink':
            effect.setParameter("Operation", operation)
        else:
            raise Exception ("Invalid operation. Operation must be 'Grow' or 'Shrink.'")
        if (isinstance(MarginSize, float) or isinstance(MarginSize, int)) and MarginSize > 0:
            effect.setParameter("MarginSizeMm", MarginSize)
        else:
            raise Exception ("Invalid MarginSize. MarginSize must be a positive number.")
        effect.self().onApply()
    except Exception as e:
        logger.exception(e)
        

def make_blank_segment(segmentationNode, segment_name):
    """
    This function creates a blank segment in a segmentation node.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the segment to.
    segment_name : str
                The name of the segment to add.

    Returns
    -------
    None
    """
    segmentationNode.GetSegmentation().AddEmptySegment(segment_name)


def copy_segmentation(segmentationNode, segment_name, new_segment_name):
    """
    This function copies a segment in a segmentation node.

    Parameters
    ----------
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to copy the segment in.
    segment_name : str
                The name of the segment to copy.
    new_segment_name : str
                The name of the new segment.

    Returns
    -------
    None
    """
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    segmentEditorWidget.setSegmentationNode(segmentationNode)
    volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
    segmentEditorWidget.setSourceVolumeNode(volumeNode)
    # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
    segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
    # Get the segment IDs
    segmentationNode.AddSegmentFromClosedSurfaceRepresentation(vtk.vtkPolyData(), new_segment_name)
    segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(new_segment_name)
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
    
    segmentEditorNode.SetSelectedSegmentID(segid_tgt)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation","COPY") # change the operation here
    effect.setParameter("ModifierSegmentID",segid_src)
    effect.self().onApply()


def combine_masks_from_array(mask_1, mask_2, add_segmentation_to_node=True, segmentationNode=None, volumeNode=None, segment_name=None):
    """
    Combines two binary masks.

    Parameters
    ----------
    mask_1 : numpy.ndarray
             The first binary mask.
    mask_2 : numpy.ndarray
             The second binary mask.
    add_segmentation_to_node : bool, default: True
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
        if isinstance(mask_1, np.ndarray) and isinstance(mask_2, np.ndarray):
            combined_mask = np.bitwise_or(np.copy(mask_1), np.copy(mask_2))
            if add_segmentation_to_node:
                addSegmentationToNodeFromNumpyArr(segmentationNode, combined_mask, segment_name, volumeNode)
            return(combined_mask)
        else:
            if isinstance(mask_1, np.ndarray):
                raise Exception("The mask_2 parameter must be a numpy.ndarray.")
            elif isinstance(mask_2, np.ndarray):
                raise Exception("The mask_1 parameter must be a numpy.ndarray.")
    except Exception as e:
        logger.exception(e)


def bitwise_and_from_array(mask_1, mask_2, add_segmentation_to_node=True, segmentationNode=None, volumeNode=None, segment_name=None):
    """
    Combines two binary masks.

    Parameters
    ----------
    mask_1 : numpy.ndarray
             The first binary mask.
    mask_2 : numpy.ndarray
             The second binary mask.
    add_segmentation_to_node : bool, default: True
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
    combined_mask = np.bitwise_and(np.copy(mask_1), np.copy(mask_2))

    if add_segmentation_to_node:
        addSegmentationToNodeFromNumpyArr(segmentationNode, combined_mask, segment_name, volumeNode)

    return(combined_mask)


def crop_inferior_to_slice(main_mask, slice_number):
    """
    Keeps a binary mask in the region inferior to the specified slice number.
    
    Parameters
    ----------  
    main_volume : numpy.ndarray
                  The binary mask that is getting cropped.
    slice_number : int
                   The slice number to crop to.
    
    Returns
    -------
    numpy.ndarray
        The main volume cropped.
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(slice_number, int):
            main_copy = np.copy(main_mask)
            slice_shape = main_mask.shape[1:]
            dtype_img = main_mask.dtype

            for index, slice in enumerate(main_copy):
                if index < slice_number:
                    np.putmask(slice, np.ones(slice_shape, dtype=dtype_img), 0)
            
            return(main_copy)
        
        else:
            if isinstance(main_mask, np.ndarray):
                raise Exception("The slice_number parameter must be an integer.")
            elif isinstance(slice_number, int):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")
    
    except Exception as e:
        logger.exception(e)


def crop_superior_to_slice(main_mask, slice_number):
    """
    Keeps a binary mask in the region superior to the specified slice number.
    
    Parameters
    ----------  
    main_volume : numpy.ndarray
                  The binary mask that is getting cropped.
    slice_number : int
                   The slice number to crop to.
    
    Returns
    -------
    numpy.ndarray
        The main volume cropped.
    """
    try:
        if isinstance(main_mask, np.ndarray) and isinstance(slice_number, int):
            main_copy = np.copy(main_mask)
            slice_shape = main_mask.shape[1:]
            dtype_img = main_mask.dtype

            for index, slice in enumerate(main_copy):
                if index > slice_number:
                    np.putmask(slice, np.ones(slice_shape, dtype=dtype_img), 0)
            
            return(main_copy)
        
        else:
            if isinstance(main_mask, np.ndarray):
                raise Exception("The slice_number parameter must be an integer.")
            elif isinstance(slice_number, int):
                raise Exception("The main_mask parameter must be a numpy.ndarray.")
    
    except Exception as e:
        logger.exception(e)


def dfs(i, j, slice):
    """
    Performs a depth-first search on a binary mask array.

    Parameters
    ----------
    i : int
        The row of the starting point.
    j : int
        The column of the starting point.
    slice : numpy.ndarray
            The binary mask array to perform the depth-first search on.
               
    Returns
    -------
    None
    """
    rows, cols = len(slice), len(slice[0])
    if i < 0 or i >= rows or j < 0 or j >= cols or slice[i][j] != 1:
        return
    slice[i][j] = 0
    dfs(i+1, j, slice)
    dfs(i-1, j, slice)
    dfs(i, j+1, slice)
    dfs(i, j-1, slice)


def first_slice_with_island(binary_mask_array, from_superior=True):
    """
    Finds the first slice with an island in a binary mask array.

    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask array to find the first slice with an island in.
    from_superior : bool, default: True
                Specification to find the first slice with an island from the superior direction.
               
    Returns
    -------
    int
        The first slice with an island.
    """
    try:
        mask_copy = np.copy(binary_mask_array)
        count = 0
        
        if not from_superior:
            for index, slice in enumerate(mask_copy):
                rows, cols = len(slice), len(slice[0])
                for i in range(rows):
                    for j in range(cols):
                        if slice[i][j] == 1:
                            dfs(i,j, slice)
                            count +=1
                if count >= 2:
                    return index
                
        elif from_superior:
            for index, slice in enumerate(mask_copy[::-1]):
                rows, cols = len(slice), len(slice[0])
                for i in range(rows):
                    for j in range(cols):
                        if slice[i][j] == 1:
                            dfs(i,j, slice)
                            count +=1
                if count >= 2:
                    return len(mask_copy) - index - 1
                
        else:
            raise Exception ("Invalid from_superior. from_superior must be True or False.")
    
    except Exception as e:
        logger.exception(e)


def project_segmentation_from_array(binary_mask_array, number_of_slices_to_combine, number_of_slices_to_project, segmentationNode, volumeNode,  project_inferior=True, project_superior=True, addSegmentationToNode=True, segment_name=None):
    """
    This function projects a binary mask array in the vertical direction.

    Parameters
    ----------
    binary_mask_array : numpy.ndarray
                The binary mask array to project.
    number_of_slices_to_combine : int
                The number of slices to combine before projection.
    number_of_slices_to_project : int
                The number of slices to project.
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the projected segment to.
    volumeNode : vtkMRMLScalarVolumeNode
                The volume node that the segmentation node is based on. 
    project_inferior : bool, default: True
                Specification to project the inferior direction.
    project_superior : bool, default: True
                Specification to project the superior direction.
    addSegmentationToNode : bool, default: True
                Specification to add the projected segment to a segmentation node.
    segment_name : str
                The name of the segment to add the projected mask to.

    Returns
    -------
    None
    """
    slices_w_mask = []
    main_copy = np.copy(binary_mask_array)
    for index, slice in enumerate(binary_mask_array):
        if 1 in slice:
            slices_w_mask.append(index)

    try:
        if project_inferior:
            bottom_slice = min(slices_w_mask)

            if isinstance(number_of_slices_to_combine, int) and number_of_slices_to_combine > 0:
                last_slices = main_copy[bottom_slice : bottom_slice + number_of_slices_to_combine]
                result = last_slices[0]
                all_slices_to_change = np.arange(bottom_slice - number_of_slices_to_project, bottom_slice + number_of_slices_to_combine)

                for slice in last_slices[1:]:
                    result = np.bitwise_or(result, slice)        
                
                for index in all_slices_to_change:
                    main_copy[index] = result
            
            else:
                raise Exception ("Invalid number_of_slices. number_of_slices must be a positive integer.")

        if project_superior:
            top_slice = max(slices_w_mask)

            if isinstance(number_of_slices_to_combine, int) and number_of_slices_to_combine > 0:
                first_slices = main_copy[top_slice - number_of_slices_to_combine + 1 : top_slice + 1]
                result = first_slices[0]
                all_slices_to_change = np.arange(top_slice - number_of_slices_to_combine + 1, top_slice + number_of_slices_to_project + 1)

                for slice in first_slices[1:]:
                    result = np.bitwise_or(result, slice)        

                for index in all_slices_to_change:
                    main_copy[index] = result
            
            else:
                raise Exception ("Invalid number_of_slices. number_of_slices must be a positive integer.")

        if addSegmentationToNode:
            addSegmentationToNodeFromNumpyArr(segmentationNode, main_copy, segment_name, volumeNode)
        
        return(main_copy)

    except Exception as e:
        logger.exception(e)


def create_binary_mask_between(binary_mask_array_left, binary_mask_array_right, from_medial=True):
    """
    Creates a binary mask between two binary masks.
    
    Parameters
    ----------
    binary_mask_array_left : numpy.ndarray
                The binary mask array to the left.
    binary_mask_array_right : numpy.ndarray
                The binary mask array to the right.
    from_medial : bool, default: True
                Specification to create the binary mask from the medial direction.
                
    Returns
    -------
    numpy.ndarray
        The binary mask between the two binary masks.
    """
    try:
        if isinstance(binary_mask_array_left, np.ndarray) and isinstance(binary_mask_array_right, np.ndarray) and (from_medial == True or from_medial == False):
            left_copy = np.copy(binary_mask_array_left)
            right_copy = np.copy(binary_mask_array_right)
            mask_w_ones = np.ones_like(left_copy)

            if from_medial:
                most_medial_left =  []
                most_medial_right = []
                
                for slice in right_copy:
                    for index in range(np.shape(slice)[1]):
                        column = slice[:,index]
                        most_medial_in_slice = 0
                        if 1 in column:
                            if index > most_medial_in_slice:
                                most_medial_in_slice = index
                        most_medial_right.append(most_medial_in_slice)
                            
                for slice in left_copy:
                    for index in range(np.shape(slice)[1]):
                        column = slice[:,index]
                        if 1 in column:
                            most_medial_in_slice = index
                            most_medial_left.append(most_medial_in_slice)
                            break
                
                mask_w_ones[:,:, min(most_medial_left):] = 0
                mask_w_ones[:,:, : max(most_medial_right) + 1] = 0


            elif from_medial == False:
                most_lateral_left =  []
                most_lateral_right = []
                
                for slice in right_copy:
                    for index in range(np.shape(slice)[1]):
                        column = slice[:,index]
                        if 1 in column:
                            most_lateral_right.append(index)
                            break
                            
                for slice in left_copy:
                    for index in range(np.shape(slice)[1]):
                        column = slice[:,index]
                        most_lateral_in_slice= 0
                        if 1 in column:
                            if index > most_lateral_in_slice:
                                most_lateral_in_slice = index
                        most_lateral_left.append(most_lateral_in_slice)

                mask_w_ones[:,:, max(most_medial_left) + 1:] = 0
                mask_w_ones[:,:, : max(most_medial_right)] = 0

            return(mask_w_ones)
    
        else:
            if isinstance(binary_mask_array_left, np.ndarray) and isinstance(binary_mask_array_right, np.ndarray):
                raise Exception("The from_medial parameter must be a boolean.")
            elif isinstance(binary_mask_array_left, np.ndarray):
                raise Exception("The binary_mask_array_right parameter must be a numpy.ndarray.")
            elif isinstance(binary_mask_array_right, np.ndarray):
                raise Exception("The binary_mask_array_left parameter must be a numpy.ndarray.")

    except Exception as e:
        logger.exception(e)


def crop_anterior(binary_mask_to_be_cropped, reference_mask, from_anterior=True):
    """
    Crops a binary mask from the anterior.
    
    Parameters
    ----------
    binary_mask_to_be_cropped : numpy.ndarray
                The binary mask to be cropped.
    reference_mask : numpy.ndarray
                The reference mask to be used for cropping.
    from_anterior : bool, default: True
                Specification to crop the binary mask from the anterior.
                
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    try:
        if isinstance(binary_mask_to_be_cropped, np.ndarray) and isinstance(reference_mask, np.ndarray) and (from_anterior == True or from_anterior == False):
            binary_mask = np.copy(binary_mask_to_be_cropped)
            reference_mask_copy = np.copy(reference_mask)

            if from_anterior:
                most_anterior =  []

                for slice in reference_mask_copy:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        if 1 in row:
                            most_anterior.append(index)
                            break

                binary_mask[:, min(most_anterior):,:] = 0


            elif from_anterior == False:
                most_posterior =  []
                
                for slice in reference_mask_copy:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        most_posterior_in_slice = 0
                        if 1 in row:
                            if index > most_posterior_in_slice:
                                most_posterior_in_slice = index
                        most_posterior.append(most_posterior_in_slice)

                binary_mask[:,: max(most_posterior),:] = 0

            return(binary_mask)
    
        else:
            if isinstance(binary_mask_to_be_cropped, np.ndarray) and isinstance(reference_mask, np.ndarray):
                raise Exception("The from_anterior parameter must be a boolean.")
            elif isinstance(binary_mask_to_be_cropped, np.ndarray):
                raise Exception("The reference_mask parameter must be a numpy.ndarray.")
            elif isinstance(reference_mask, np.ndarray):
                raise Exception("The binary_mask_to_be_cropped parameter must be a numpy.ndarray.")

    except Exception as e:
        logger.exception(e)


def remove_roi_from_mask(binary_mask_array, Tuple_of_masks_to_remove, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
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


def crop_posterior_from_slice_number(binary_mask_array, slice_number, tuple_of_masks, from_anterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
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
    from_anterior : bool, default: True
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
        if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(tuple_of_masks, tuple) and isinstance(from_anterior, bool) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
    
            if from_anterior:
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


            elif from_anterior == False:
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
                raise Exception("The from_anterior parameter must be a boolean.")
            elif isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int):
                raise Exception("The tuple_of_masks parameter must be a tuple.")
            elif isinstance(tuple_of_masks, tuple):
                raise Exception("The binary_mask_array parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_anterior_from_slice_number(binary_mask_array, slice_number, tuple_of_masks, from_anterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
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
    from_anterior : bool, default: True
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
        if isinstance(binary_mask_array, np.ndarray) and isinstance(slice_number, int) and isinstance(tuple_of_masks, tuple) and isinstance(from_anterior, bool) and isinstance(add_segmentation_to_node, bool):
            binary_mask = np.copy(binary_mask_array)
    
            if from_anterior:
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


            elif from_anterior == False:
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
                raise Exception("The from_anterior parameter must be a boolean.")
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



def crop_posterior_from_distance(binary_mask, reference_mask, number_of_pixels, from_anterior=True, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
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
    from_anterior : bool, default: True
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
        if isinstance(binary_mask, np.ndarray) and isinstance(reference_mask, np.ndarray) and isinstance(number_of_pixels, int) and (from_anterior == True or from_anterior == False):
            binary_mask = np.copy(binary_mask)


            if from_anterior:
                most_anterior =  []

                for slice in reference_mask:
                    for index in range(np.shape(slice)[0]):
                        row = slice[index,:]
                        if 1 in row:
                            most_anterior.append(index)
                            break

                binary_mask[:, : min(most_anterior) - number_of_pixels,:] = 0


            elif from_anterior == False:
                most_posterior =  []

                for slice in reference_mask:
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
            if isinstance(binary_mask, np.ndarray) and isinstance(reference_mask, np.ndarray) and isinstance(number_of_pixels, int):
                raise Exception("The from_anterior parameter must be a boolean.")
            elif isinstance(binary_mask, np.ndarray) and isinstance(reference_mask, np.ndarray):
                raise Exception("The number_of_pixels parameter must be an integer.")
            elif isinstance(reference_mask, np.ndarray) and isinstance(number_of_pixels, int):
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


def create_binary_mask_between_slices(reference_mask_array, left_bound, right_bound, segmentationNode=None, volumeNode=None, volume_name=None, add_segmentation_to_node=False):
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
            mask_w_ones = np.ones_like(reference_mask_array)

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