# Created by Marcus Milantoni and Edward Wang from Western University/ Verspeeten Family Cancer Centre. This script contains basic functions for image processing in Slicer.

import slicer, vtk
from DICOMLib import DICOMUtils
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import ScreenCapture


# Setup the logger (can customize)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class medicalImage:
    preset_colormaps = ['CT_BONE', 'CT_AIR', 'CT_BRAIN', 'CT_ABDOMEN', 'CT_LUNG', 'PET', 'DTI']
    pet_colormaps = ['PET-Heat','PET-Rainbow2']

    def __init__(self, name, volumeNode):
        self.name = name
        self.volumeNode = volumeNode
        self.NumPyArray = slicer.util.arrayFromVolume(self.volumeNode)
        self.shape = self.NumPyArray.shape
        self.maxValue = np.max(self.NumPyArray)
        self.minValue = np.min(self.NumPyArray)
        self.spacing = self.volumeNode.GetSpacing()
        self.origin = self.volumeNode.GetOrigin()
        self.ID = self.volumeNode.GetID()
        self.getName = self.volumeNode.GetName()


    def description(self):
        return f"Name: {self.name}, Shape: {self.shape}"


    def updateSlicerView(self):
        slicer.util.updateVolumeFromArray(self.volumeNode, self.NumPyArray)


    def makeCopy(self, newName):
        if not isinstance(newName, str):
            raise TypeError("The newName parameter must be a string.")
        
        try:
            logger.info(f"Making a copy of the volume node with name {newName}")
            copiedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', newName)
            copiedNode.Copy(self.volumeNode)
            return copiedNode

        except Exception:
            logger.exception("An error occurred in makeCopy")
            raise


    def setAsForeground(self, backgroundNode, opacity=0.75):
        if not isinstance(backgroundNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The backgroundNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(opacity, (int, float)):
            raise TypeError("The opacity parameter must be an integer or a float.")
        if not 0 <= opacity <= 1:
            raise ValueError("The opacity parameter must be between 0 and 1.")

        try:
            logger.info(f"Setting the volume node as the foreground image with opacity {opacity}")
            slicer.util.setSliceViewerLayers(background=backgroundNode, foreground=self.volumeNode, foregroundOpacity=opacity)

        except Exception:
            logger.exception("An error occurred in setAsForeground")
            raise


    def setCMAP(self, cmap):
        if not isinstance(cmap, str):
            raise TypeError("The cmap parameter must be a string.")
        if not cmap in self.pet_colormaps or cmap in self.preset_colormaps:
            raise ValueError("The cmap parameter must be a valid colormap.")

        try:
            logger.info(f"Setting the colormap to {cmap}")

            if cmap in self.preset_colormaps:
                slicer.modules.volumes.logic().ApplyVolumeDisplayPreset(self.volumeNode.GetVolumeDisplayNode(), cmap)

            else:
                ColorNode = slicer.mrmlScene.GetFirstNodeByName(cmap)
                self.volumeNode.GetVolumeDisplayNode().SetAndObserveColorNodeID(ColorNode.GetID())
                self.volume_node.GetVolumeDisplayNode().AutoWindowLevelOn()

        except Exception:
            logger.exception("An error occurred in setCMAP")
            raise


    def editName(self, newName):
        if not isinstance(newName, str):
            raise TypeError("The newName parameter must be a string.")

        try:
            logger.info(f"Editing the name of the volume node to {newName}")
            self.volumeNode.SetName(newName)
            self.name = newName

        except Exception:
            logger.exception("An error occurred in editName")
            raise


    def checkSegmentShapeMatch(self, segmentationArray):
        if not isinstance(segmentationArray, np.ndarray):
            raise TypeError("The segmentationArray parameter must be a numpy array.")

        try:
            logger.info(f"Checking if the shape of the segmentation array matches the volume node")
            if segmentationArray.shape == self.shape:
                return True
            else:
                return False

        except Exception:
            logger.exception("An error occurred in checkSegmentShapeMatch")
            raise


    def cropVolumeFromSegment(self, segmentationArray, updateSlicerView=True):
        if not isinstance(segmentationArray, np.ndarray):
            raise TypeError("The segmentationArray parameter must be a numpy array.")
        if not segmentationArray.shape == self.shape:
            raise ValueError("The segmentationArray parameter must have the same shape as the volume node.")

        try:
            logger.info(f"Cropping the volume node from the segmentation array")
            self.NumPyArray = self.NumPyArray * segmentationArray
            
            if updateSlicerView:
                self.updateSlicerView()

        except Exception:
            logger.exception("An error occurred in cropVolumeFromSegment")
            raise

    
    def resampleScalarVolumeBRAINS(self, referenceVolumeNode, nodeName, interpolatorType='NearestNeighbor'):
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
        interpolatorType : str, optional
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
        if not isinstance(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise ValueError("Reference volume node must be a vtkMRMLScalarVolumeNode")
        if not isinstance(nodeName, str):
            raise ValueError("Node name must be a string")
        if not isinstance(interpolatorType, str) or interpolatorType not in ['NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline','WindowedSinc']:
            raise ValueError("Interpolator type must be a string and one of 'NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline', 'WindowedSinc'")

        parameters = {}
        parameters["inputVolume"] = self.volumeNode
        try:
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', nodeName)
        except:
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        parameters["outputVolume"] = outputModelNode
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
        return outputModelNode


    def quickVisualize(self, cmap='gray', indicies=None):
        """
        Visualizes axial, coronal, and sagittal slices of a 3D image array.
        
        Parameters
        ----------
        cmap : str, default: 'gray'
            The colormap to use for the visualization.
        indices : dict, default: None
            The indices of the slices to visualize. If None, slices at 1/4, 1/2, and 3/4 of the image dimensions are used.
        
        Raises
        ------
        TypeError
            If imageArray is not a numpy ndarray.
            If cmap is not a string.
            If indices is not a dictionary.
        ValueError
            If imageArray is not 3D.
            If cmap is not a valid matplotlib colormap.
            If indices does not contain 'axial', 'coronal', and 'sagittal' keys with lists of 3 integers each.
        """
        if not isinstance(cmap, str):
            raise TypeError("cmap must be a string.")
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


#################################################################################
# Basic functions for import and creation                                       #
#################################################################################

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


def get_ct_and_pet_volume_nodes():
    """
    This function gets the volume nodes of the CT and PET images.

    Returns
    -------
    tuple[slicer.vtkMRMLVolumeNode]
        The volume nodes of the CT and PET images.
    """
    ct_node_found = False
    pet_node_found = False
    for Volume_Node in slicer.util.getNodesByClass("vtkMRMLVolumeNode"):
        if ('ct ' in Volume_Node.GetName().lower() or 'ct_' in Volume_Node.GetName().lower())and not ct_node_found:
            ct_node_found = True
            ct_node = Volume_Node
        elif ('suvbw' in Volume_Node.GetName().lower() or 'standardized_uptake_value_body_weight' in Volume_Node.GetName().lower() or 'pet ' in Volume_Node.GetName().lower()) and not pet_node_found:
            pet_node_found = True
            pet_node = Volume_Node
        if ct_node_found and pet_node_found:
            break
    
    if ct_node_found and pet_node_found:
        return(ct_node, pet_node)
    else:
        raise Exception("CT and PET nodes not found.")


def quick_visualize(imageArray, cmap='gray', indices=None):
    """
    Visualizes axial, coronal, and sagittal slices of a 3D image array.
    
    Parameters
    ----------
    imageArray : np.ndarray
        The 3D image array to visualize.
    cmap : str, default: 'gray'
        The colormap to use for the visualization.
    indices : dict, default: None
        The indices of the slices to visualize. If None, slices at 1/4, 1/2, and 3/4 of the image dimensions are used.
    
    Raises
    ------
    TypeError
        If imageArray is not a numpy ndarray.
        If cmap is not a string.
        If indices is not a dictionary.
    ValueError
        If imageArray is not 3D.
        If cmap is not a valid matplotlib colormap.
        If indices does not contain 'axial', 'coronal', and 'sagittal' keys with lists of 3 integers each.
    """
    if not isinstance(imageArray, np.ndarray):
        raise TypeError("imageArray must be a numpy ndarray.")
    if imageArray.ndim != 3:
        raise ValueError("imageArray must be 3D.")
    if not isinstance(cmap, str):
        raise TypeError("cmap must be a string.")
    if indices is not None:
        if not isinstance(indices, dict):
            raise TypeError("indices must be a dictionary.")
        if not all(key in indices for key in ['axial', 'coronal', 'sagittal']):
            raise ValueError("indices must contain 'axial', 'coronal', and 'sagittal' keys.")
        if not all(isinstance(indices[key], list) and len(indices[key]) == 3 for key in ['axial', 'coronal', 'sagittal']):
            raise ValueError("Each key in indices must have a list of 3 integers.")

    # Automatically select indices if not provided
    if indices is None:
        shape = imageArray.shape
        indices = {
            'axial': [shape[0] // 4, shape[0] // 2, 3 * shape[0] // 4],
            'coronal': [shape[1] // 4, shape[1] // 2, 3 * shape[1] // 4],
            'sagittal': [shape[2] // 4, shape[2] // 2, 3 * shape[2] // 4],
        }

    plt.figure(figsize=(10, 10))

    # Axial slices
    for i, idx in enumerate(indices['axial'], 1):
        plt.subplot(3, 3, i)
        plt.imshow(imageArray[idx, :, :], cmap=cmap)
        plt.title(f"Axial slice {idx}")

    # Coronal slices
    for i, idx in enumerate(indices['coronal'], 4):
        plt.subplot(3, 3, i)
        plt.imshow(imageArray[:, idx, :], cmap=cmap)
        plt.title(f"Coronal slice {idx}")

    # Sagittal slices
    for i, idx in enumerate(indices['sagittal'], 7):
        plt.subplot(3, 3, i)
        plt.imshow(imageArray[:, :, idx], cmap=cmap)
        plt.title(f"Sagittal slice {idx}")

    plt.tight_layout()
    plt.show()


def dice_similarity(mask1, mask2):
    """
    Calculate the Dice Similarity Index between two binary masks.

    Parameters
    ----------
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns
    -------
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


def sweep_screen_capture(backgroundImageNode, savePath, saveName, tupleOfSegmentationNodesToShow=None, view='axial', frameRate=None, startSweepOffset=None, endSweepOffset=None, foregroundImageNode=None, foregroundOpacity=None, numberOfImages= None, ):
    """
    This function captures a sweep of images from a volume node and saves them as a video to a specified location.

    Parameters:
    ------------
    backgroundImageNode: vtkMRMLScalarVolumeNode
        The volume node to capture the images from.
    savePath: str
        The path to save the images to.
    saveName: str
        The name to save the images as.
    tupleOfSegmentationNodesToShow: tuple, optional
        A tuple of vtkMRMLSegmentationNodes to show in the video. Default is None.
    view: str, optional
        The view to capture the images from. Default is 'axial'.
    frameRate: int or float, optional
        The frame rate of the video. Default is None.
    startSweepOffset: int or float, optional
        The offset to start the sweep from. Default is None.
    endSweepOffset: int or float, optional
        The offset to end the sweep at. Default is None.
    foregroundImageNode: vtkMRMLScalarVolumeNode, optional
        The volume node to overlay on the background image. Default is None.
    foregroundOpacity: int or float, optional
        The opacity of the foreground image. Default is None.
    numberOfImages: int, optional
        The number of images to capture. Default is None. 
    
    Returns:
    ---------
    None
    """
    if not isinstance(backgroundImageNode, slicer.vtkMRMLScalarVolumeNode):
        raise TypeError("backgroundImageNode must be a vtkMRMLScalarVolumeNode")
    if not isinstance(savePath, str):
        raise TypeError("savePath must be a string")
    if not isinstance(saveName, str):
        raise TypeError("saveName must be a string")
    if not isinstance(tupleOfSegmentationNodesToShow, tuple) or not tupleOfSegmentationNodesToShow:
        raise TypeError("tupleOfSegmentationNodesToShow must be a tuple or None")
    if not isinstance(view, str):
        raise TypeError("view must be a string")
    if frameRate is not None and not isinstance(frameRate, (int, float)):
        raise TypeError("frameRate must be an integer or a float or None.")
    if startSweepOffset is not None and not isinstance(startSweepOffset, (int, float)):
        raise TypeError("startSweepOffset must be an integer or a float or None.")
    if endSweepOffset is not None and not isinstance(endSweepOffset, (int, float)):
        raise TypeError("endSweepOffset must be an integer or a float or None.")
    if foregroundImageNode is not None and not isinstance(foregroundImageNode, slicer.vtkMRMLScalarVolumeNode):
        raise TypeError("ForegroundImageNode must be a vtkMRMLScalarVolumeNode or None")
    if numberOfImages is not None and not isinstance(numberOfImages, int):
        raise TypeError("numberOfImages must be an integer")
    if foregroundOpacity is not None and not isinstance(foregroundOpacity, (int, float)):
        raise TypeError("foregroundOpacity must be an integer or a float or None.")

    if foregroundOpacity is not None and not 0 <= foregroundOpacity <= 1:
        raise ValueError("foregroundOpacity must be between 0 and 1")
    if not view.lower() in ['axial', 'sagittal', 'coronal']:
        raise ValueError("view must be either 'axial', 'sagittal' or 'coronal'")
    if not all(isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) for segmentationNode in tupleOfSegmentationNodesToShow):
        raise ValueError("All elements of tupleOfSegmentationNodesToShow must be vtkMRMLSegmentationNodes")
    if frameRate is not None and not 0 <= frameRate <= 60:
        raise ValueError("frameRate must be between 0 and 60 frames per second")
    
    # Create the save path if it does not exist
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # Set the index for the desired view
    logger.debug(f"Setting the view to {view}")
    if view.lower() == 'axial':
        index = 2
        sliceNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
    elif view.lower() == 'sagittal':
        index = 0
        sliceNode = slicer.util.getNode("vtkMRMLSliceNodeYellow")
    elif view.lower() == 'coronal':
        index = 1
        sliceNode = slicer.util.getNode("vtkMRMLSliceNodeGreen")

    # set the start and end sweep offsets if none are provided
    if not startSweepOffset:
        logger.debug("No start sweep offset provided, setting it to the start of the volume")
        if index == 2:
            startSweepOffset = round(backgroundImageNode.GetOrigin()[index], 2)
        else:
            startSweepOffset = round(backgroundImageNode.GetOrigin()[index] - backgroundImageNode.GetSpacing()[index] * (backgroundImageNode.GetImageData().GetDimensions()[index]-1), 2)
    
    if not endSweepOffset:
        logger.debug("No end sweep offset provided, setting it to the end of the volume")
        if index == 2:
            endSweepOffset = round(backgroundImageNode.GetOrigin()[index] + backgroundImageNode.GetSpacing()[index] * (backgroundImageNode.GetImageData().GetDimensions()[index]-1), 2)
        else:
            round(backgroundImageNode.GetOrigin()[index], 2)

    if not numberOfImages:
        logger.debug("No number of images provided, setting it to the number of slices in the volume")
        numberOfImages = backgroundImageNode.GetImageData().GetDimensions()[index] - 1 # Set the number of images to the number of slices in the volume

    if not frameRate:
        logger.debug("No frame rate provided, setting it to 6 frames per second")
        frameRate = 4 # Set the frame rate to 6 frames per second

    # Set the foreground opacity to 50% if none is provided and there is a foreground image
    if foregroundImageNode and not foregroundOpacity:
        logger.debug("Foreground image provided but no opacity, setting opacity to 50%")
        foregroundOpacity = 0.5

    # Set the display to what is desired for the video
    logger.debug(f"Setting the display for the {view} view")
    slicer.util.setSliceViewerLayers(background=backgroundImageNode, foreground=foregroundImageNode, foregroundOpacity=foregroundOpacity)
    for currentSegmentationNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"): # Hide all segmentations
        currentSegmentationNode.GetDisplayNode().SetVisibility(False)
    if tupleOfSegmentationNodesToShow:
        for currentSegmentationNode in tupleOfSegmentationNodesToShow: # Show the desired segmentations
            currentSegmentationNode.GetDisplayNode().SetVisibility(True)

    # Capture the individual images
    logger.debug(f"Capturing {numberOfImages} images from {startSweepOffset} to {endSweepOffset} in the {view} view")
    ScreenCapture.ScreenCaptureLogic().captureSliceSweep(sliceNode, startSweepOffset, endSweepOffset, numberOfImages, savePath, f"{saveName}_%05d.png")

    # create the video freom the images
    logger.debug(f"Creating video from images at {savePath}/{saveName}.mp4")
    ScreenCapture.ScreenCaptureLogic().createVideo(frameRate, "-codec libx264 -preset slower -pix_fmt yuv420p", savePath, f"{saveName}_%05d.png", f"{saveName}.mp4")

    # Delete the temporairly saved images after the video is created
    for imageIndex in range(numberOfImages):
       logger.debug(f"Deleting {savePath}/{saveName}_{imageIndex:05d}.png")
       os.remove(os.path.join(savePath, f"{saveName}_{imageIndex:05d}.png"))



#################################################################################
# Checks and finds                                                              #
#################################################################################

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


def get_most_superior_slice(tupleOfMasks):
    """
    Finds the most superior slice in a tuple of binary masks.
    
    Parameters
    ----------
    tupleOfMasks : tuple[numpy.ndarray]
                The tuple of binary masks to find the most superior slice in.
    
    Returns
    -------
    int
        The most superior slice.
    """
    if not isinstance(tupleOfMasks, tuple):
        raise TypeError("The tupleOfMasks parameter must be a tuple.")

    most_superior = 0
    for mask in tupleOfMasks:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"The tupleOfMasks parameter must contain only numpy.ndarrays. The index {index} is not a numpy array.")
        for index, slice in enumerate(mask):
            if 1 in slice and index > most_superior:
                most_superior = index

    return most_superior



#################################################################################
# Effect utilization                                                            #
#################################################################################

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
                The distance in milimeters to dilate or shrink the segment by.

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


def logical_operator_slicer(primarySegment, secondarySegment, segmentationNode, volumeNode, operator='copy'):
    """
    Function to apply logical operations to two segmentations in slicer. The main segmentation is the segmentation that will be modified. The operator will select the type of transformation applied.
    
    Parameters
    ----------
    primarySegment : str
        The name of the segmentation to be modified.
    secondarySegment : str
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
    if not isinstance(primarySegment, str):
        raise TypeError("Main segmentation must be a string")
    if not isinstance(secondarySegment, str):
        raise TypeError("Secondary segmentation must be a string")
    if not isinstance(operator, str):
        raise TypeError("Operator must be a string")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("segmentationNode must be a vtkMRMLSegmentationNode")
    if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
        raise TypeError("volumeNode must be a vtkMRMLVolumeNode")

    # Get the segmentation editor widget
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    #Set the correct volumes to be used in the Segment Editor
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(volumeNode)
    # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
    segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
    # Get the segment ID
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(primarySegment)
    segid_tgt = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(secondarySegment)

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


def segmentation_by_auto_threshold(segmentName, segmentationNode, pet_node, threshold='OTSU'):
    """
    Function to apply the autosegmentation threshold to a new segmentation in slicer defined by the segmentation name.
    
    Parameters
    ----------
    segmentName : str
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
    if not isinstance(segmentName, str):
        raise TypeError("Segmentation name must be a string")
    if not isinstance(threshold, str):
        raise TypeError("Threshold must be a string")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("segmentationNode must be a vtkMRMLSegmentationNode")
    if not isinstance(pet_node, slicer.vtkMRMLVolumeNode):
        raise TypeError("pet_node must be a vtkMRMLVolumeNode")
    
    #Create a blank segmentation to do the autothresholding
    make_blank_segment(segmentationNode,segmentName)

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
    segid_src = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    segmentEditorNode.SetSelectedSegmentID(segid_src)

    # Set the active effect to Threshold on the Segment Editor widget
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    
    if threshold.lower() == 'otsu':
        effect.setParameter("AutoThresholdMethod","OTSU")
        effect.setParameter("AutoThresholdMode", "SET_LOWER_MAX")
        #Save the active effects
        effect.self().onAutoThreshold()
    else:
        raise ValueError("Threshold not recognized")
        
    #Apply the effect
    effect.self().onApply()


def resampleScalarVolumeBrains(inputVolumeNode, referenceVolumeNode, NodeName, interpolatorType='NearestNeighbor'):
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
    interpolatorType : str, optional
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
    if not isinstance(inputVolumeNode, slicer.vtkMRMLScalarVolumeNode):
        raise ValueError("Input volume node must be a vtkMRMLScalarVolumeNode")
    if not isinstance(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode):
        raise ValueError("Reference volume node must be a vtkMRMLScalarVolumeNode")
    if not isinstance(NodeName, str):
        raise ValueError("Node name must be a string")
    if not isinstance(interpolatorType, str) or interpolatorType not in ['NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline','WindowedSinc']:
        raise ValueError("Interpolator type must be a string and one of 'NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline', 'WindowedSinc'")

    parameters = {}
    parameters["inputVolume"] = inputVolumeNode
    try:
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', NodeName)
    except:
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    parameters["outputVolume"] = outputModelNode
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
    return outputModelNode


def islands_effect_segment_editor(segmentName, segmentationNode, volumeNode, edit='KEEP_LARGEST_ISLAND', minimumSize=5):
    """
    Applies the 'Islands' effect in the Segment Editor of 3D Slicer to a given segment.
    
    Parameters
    ----------
    segmentName : str
        The name of the segment to apply the effect to.
    segmentationNode : slicer.vtkMRMLSegmentationNode
        The segmentation node containing the segment.
    volumeNode : slicer.vtkMRMLVolumeNode
        The volume node associated with the segmentation.
    edit : str, default: 'KEEP_LARGEST_ISLAND'
        The type of edit to perform. Must be one of 'KEEP_LARGEST_ISLAND', 'KEEP_SELECTED_ISLAND', 'REMOVE_SMALL_ISLANDS', 'REMOVE_SELECTED_ISLAND', 'ADD_SELECTED_ISLAND', 'SPLIT_ISLANDS_TO_SEGMENTS'.
    minimumSize : int, default: 5
        The minimum size of islands to keep when removing small islands. Only applicable if edit is 'KEEP_LARGEST_ISLAND', 'REMOVE_SMALL_ISLANDS', or 'SPLIT_ISLANDS_TO_SEGMENTS'.
    
    Raises
    ------
    TypeError
        If segmentName is not a string.
        If segmentationNode is not a vtkMRMLSegmentationNode.
        If volumeNode is not a vtkMRMLVolumeNode.
        If edit is not a string.
        If minimumSize is not an integer.
    ValueError
        If edit is not one of the valid options.
        If minimumSize is not between 0 and 1000.
    """
    edit_options_size = ['KEEP_LARGEST_ISLAND', 'REMOVE_SMALL_ISLANDS', 'SPLIT_ISLANDS_TO_SEGMENTS']
    edit_options_others = ['KEEP_SELECTED_ISLAND', 'REMOVE_SELECTED_ISLAND', 'ADD_SELECTED_ISLAND']

    if not isinstance(segmentName, str):
        raise TypeError("Segmentation name must be a string.")
    if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
        raise TypeError("segmentationNode must be a vtkMRMLSegmentationNode.")
    if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
        raise TypeError("volumeNode must be a vtkMRMLVolumeNode.")
    if not isinstance(edit, str) or edit not in edit_options_size + edit_options_others:
        raise ValueError("edit must be a string and one of 'KEEP_LARGEST_ISLAND', 'KEEP_SELECTED_ISLAND', 'REMOVE_SMALL_ISLANDS', 'REMOVE_SELECTED_ISLAND', 'ADD_SELECTED_ISLAND', 'SPLIT_ISLANDS_TO_SEGMENTS'.")
    if not isinstance(minimumSize, int) or minimumSize < 0 or minimumSize > 1000:
        raise ValueError("minimumSize must be an integer between 0 and 1000.")

    try:
        logger.info("Applying Islands effect to segment.")

        # Get the segmentation editor widget
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        # Set the correct volumes to be used in the Segment Editor
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setSourceVolumeNode(volumeNode)

        # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
        segmentEditorNode.SetOverwriteMode(2)
        # Get the segment ID
        segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        segmentEditorNode.SetSelectedSegmentID(segmentID)

        # Set the active effect to Islands on the segment editor widget
        segmentEditorWidget.setActiveEffectByName("Islands")
        effect = segmentEditorWidget.activeEffect()

        # Set a minimum size for the effect
        if edit in edit_options_size:
            effect.setParameter("MinimumSize", minimumSize)

        # Set the effect to be used
        effect.setParameter("Operation", edit)

        # Apply the effect
        effect.self().onApply()
    except Exception as e:
        logger.exception(e)



#################################################################################
# Array manipulation                                                            #
#################################################################################
        
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
            raise TypeError("The segmentName parameter must be a string.")

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
    segmentName : str
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
            raise TypeError("The segmentName parameter must be a string.")

        logger.debug(f"Adding the combined mask to the segmentation node {segmentationNode}")     
        add_segmentation_array_to_node(segmentationNode, combined_mask, segmentName, volumeNode)
    
    return combined_mask


def bitwise_or_from_array(mask1, mask2, addSegmentationToNode=False, segmentationNode=None, volumeNode=None, segmentName=None):
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
    segmentationNode : vtkMRMLSegmentationNode
                The segmentation node to add the combined mask to.
    volumeNode : vtkMRMLScalarVolumeNode
                The volume node that the segmentation node is based on. 
    segmentName : str
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
        combined_mask = np.bitwise_or(mask1, mask2)

    except Exception:
        logger.exception("An error occurred in bitwise_or_from_array")
        raise
            
    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLScalarVolumeNode.")
        if not isinstance(segmentName, str):
            raise TypeError("The segmentName parameter must be a string.")

        logger.debug(f"Adding the combined mask to the segmentation node {segmentationNode}")
        add_segmentation_array_to_node(segmentationNode, combined_mask, segmentName, volumeNode)

    return combined_mask


def remove_multiple_rois_from_mask(binaryMaskArray, tupleOfMasksToRemove, segmentationNode=None, volumeNode=None, volumeName=None, addSegmentationToNode=False):
    """
    Removes a region of interest from a binary mask.
    
    Parameters
    ----------
    binaryMaskArray : numpy.ndarray
                The binary mask to be cropped.
    tupleOfMasksToRemove : tuple[numpy.ndarray]
                The masks to be removed from the binary mask.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeName : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    addSegmentationToNode : bool, default: False
                Specification to add the cropped binary mask to a node. 
    
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.

    Raises
    ------
    TypeError
        If the binaryMaskArray is not a numpy array.
        If the tupleOfMasksToRemove is not a tuple.
        If the tupleOfMasksToRemove contains an element that is not a numpy array.
        If the tupleOfMasksToRemove contains an element that does not have the same shape as the binaryMaskArray.
        If the addSegmentationToNode is not a boolean.
    ValueError
        If the tupleOfMasksToRemove contains an element that does not have the same shape as the binaryMaskArray.
    """
    if not isinstance(binaryMaskArray, np.ndarray):
        raise TypeError("The binaryMaskArray parameter must be a numpy.ndarray.")
    if not isinstance(tupleOfMasksToRemove, tuple):
        raise TypeError("The tupleOfMasksToRemove parameter must be a tuple.")
    if not all(isinstance(mask, np.ndarray) for mask in tupleOfMasksToRemove):
        raise TypeError("The tupleOfMasksToRemove parameter must contain only numpy.ndarrays.")
    if not all(mask.shape == binaryMaskArray.shape for mask in tupleOfMasksToRemove):
        raise ValueError("The tupleOfMasksToRemove parameter must contain numpy.ndarrays with the same shape as the binaryMaskArray parameter.")
    if not isinstance(addSegmentationToNode, bool):
        raise TypeError("The addSegmentationToNode parameter must be a boolean.")
    
    try:
        binary_mask = np.copy(binaryMaskArray)
        for mask in tupleOfMasksToRemove:
            binary_mask = binary_mask - mask

    except Exception:
        logger.exception("An error occurred in remove_multiple_rois_from_mask")
        raise

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
            raise TypeError("The volumeNode parameter must be a vtkMRMLVolumeNode.")
        if not isinstance(volumeName, str):
            raise TypeError("The volumeName parameter must be a string.")
    
    return binary_mask 


def zero_image_where_mask_is_present(imageArray, binaryMaskArray):
    """
    Sets the values to zero in the specified image where a binary mask overlaps.

    Parameters
    ----------
    imageArray : numpy.ndarray
        The original image as a NumPy array.
    binaryMaskArray : numpy.ndarray
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
    if not isinstance(imageArray, np.ndarray) or not isinstance(binaryMaskArray, np.ndarray):
        raise ValueError("Both imageArray and binaryMaskArray must be NumPy arrays.")
    if imageArray.shape != binaryMaskArray.shape:
        raise ValueError("imageArray and binaryMaskArray must have the same shape.")

    # Apply the binaryMaskArray: Set imageArray pixels to zero where binaryMaskArray is 1
    modified_image = np.where(binaryMaskArray == 1, 0, imageArray)

    return modified_image



#################################################################################
# Crop functions                                                                #
#################################################################################

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
            raise TypeError("The segmentName parameter must be a string.")

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
            raise TypeError("The segmentName parameter must be a string.")

        logger.debug(f"Adding the cropped mask to the segmentation node {segmentationNode}")     
        add_segmentation_array_to_node(segmentationNode, main_copy, segmentName, volumeNode)
    
    return main_copy


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
        The cropped binary mask. The returned array is the same instance as the input `binaryMaskArray`
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


def crop_superior_to_slice(binaryMaskArray, sliceNumber):
    """
    Crops the given binary mask array to keep only the region superior to the specified slice number.
    This function returns a new array, with all slices inferior to the specified slice number set to zero,
    effectively removing them from the binary mask.

    Parameters
    ----------
    binaryMaskArray : numpy.ndarray
        The binary mask to be cropped. It should be a 3D array where each slice along the first dimension
        represents a 2D binary mask.
    sliceNumber : int
        The slice number above which the binary mask should be kept. All slices at and below this number
        will be set to 0.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask. The returned array is a new instance with modifications applied.

    Raises
    ------
    TypeError
        If `binaryMaskArray` is not a numpy.ndarray or `sliceNumber` is not an integer, a TypeError is raised
        with an appropriate error message.
    """
    if not isinstance(binaryMaskArray, np.ndarray):
        raise TypeError("The binaryMaskArray parameter must be a numpy.ndarray.")
    if not isinstance(sliceNumber, int):
        raise TypeError("The sliceNumber parameter must be an integer.")

    main_copy = np.copy(binaryMaskArray)
    slice_shape = binaryMaskArray.shape[1:]
    dtype_img = binaryMaskArray.dtype

    for index, slice in enumerate(main_copy):
        if index <= sliceNumber:
            np.putmask(slice, np.ones(slice_shape, dtype=dtype_img), 0)
    
    return main_copy


def crop_posterior_from_sliceNumber(binaryMaskArray, sliceNumber, tupleOfMasks, fromAnterior=True, segmentationNode=None, volumeNode=None, volumeName=None, addSegmentationToNode=False):
    """
    This function crops a binary mask posteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binaryMaskArray.

    Parameters
    ----------
    binaryMaskArray : numpy.ndarray
                The binary mask to be cropped.
    sliceNumber : int
                The slice number to crop from.
    tupleOfMasks : tuple[numpy.ndarray]
                The masks to be used for cropping.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeName : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    addSegmentationToNode : bool, default: False
                Specification to add the cropped binary mask to a node.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """

    if not isinstance(binaryMaskArray, np.ndarray):
        raise TypeError("The binaryMaskArray parameter must be a numpy.ndarray.")
    if not isinstance(sliceNumber, int):
        raise TypeError("The sliceNumber parameter must be an integer.")
    if not isinstance(tupleOfMasks, tuple):
        raise TypeError("The tupleOfMasks parameter must be a tuple.")
    if not isinstance(fromAnterior, bool):
        raise TypeError("The fromAnterior parameter must be a boolean.")
    if not isinstance(addSegmentationToNode, bool):
        raise TypeError("The addSegmentationToNode parameter must be a boolean.")

    binary_mask = np.copy(binaryMaskArray)

    if fromAnterior:
        most_anterior =  []

        for mask in tupleOfMasks:
            if isinstance(mask, np.ndarray):
                slice = mask[sliceNumber]
                for index in range(np.shape(slice)[0]):
                    row = slice[index,:]
                    if 1 in row:
                        most_anterior.append(index)
                        break

        binary_mask[:, : min(most_anterior),:] = 0


    elif fromAnterior == False:
        most_posterior =  []

        for mask in tupleOfMasks:
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

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
            raise TypeError("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
        if not isinstance(volumeName, str):
            raise TypeError("The volumeName parameter must be a string.")

        add_segmentation_array_to_node(segmentationNode, binary_mask, volumeName, volumeNode)

    return binary_mask


def crop_anterior_from_sliceNumber(binaryMaskArray, sliceNumber, tupleOfMasks, fromAnterior=True, segmentationNode=None, volumeNode=None, volumeName=None, addSegmentationToNode=False, debug=False):
    """
    This function crops a binary mask anteriorly from the most anterior or posterior row in a mask called tuple of masks. The mask to be cropped is called binaryMaskArray.

    Parameters
    ----------
    binaryMaskArray : numpy.ndarray
                The binary mask to be cropped.
    sliceNumber : int
                The slice number to crop from.
    tupleOfMasks : tuple[numpy.ndarray]
                The masks to be used for cropping.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeName : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node
    addSegmentationToNode : bool, default: False
                Specification to add the cropped binary mask to a node.
    debug : bool, default: False
                If set to True, debug information will be logged.

    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """

    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        if isinstance(binaryMaskArray, np.ndarray) and isinstance(sliceNumber, int) and isinstance(tupleOfMasks, tuple) and isinstance(fromAnterior, bool) and isinstance(addSegmentationToNode, bool):
            binary_mask = np.copy(binaryMaskArray)
    
            if fromAnterior:
                most_anterior =  []

                for mask in tupleOfMasks:
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

                for mask in tupleOfMasks:
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
                    
            if addSegmentationToNode:
                if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode) and isinstance(volumeNode, slicer.vtkMRMLVolumeNode) and isinstance(volumeName, str):
                    add_segmentation_array_to_node(segmentationNode, binary_mask, volumeName, volumeNode)
                else:
                    if isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
                        raise Exception("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
                    elif isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
                        raise Exception("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
                    elif isinstance(volumeName, str):
                        raise Exception("The volumeName parameter must be a string.")

        
            return(binary_mask)

        else:
            if isinstance(binaryMaskArray, np.ndarray) and isinstance(sliceNumber, int) and isinstance(tupleOfMasks, tuple):
                raise Exception("The fromAnterior parameter must be a boolean.")
            elif isinstance(binaryMaskArray, np.ndarray) and isinstance(sliceNumber, int):
                raise Exception("The tupleOfMasks parameter must be a tuple.")
            elif isinstance(tupleOfMasks, tuple):
                raise Exception("The binaryMaskArray parameter must be a numpy.ndarray.")
            
    except Exception as e:
        logger.exception(e)


def crop_posterior_from_distance(binaryMaskArray, referenceMask, numberOfPixels, fromAnterior=True, segmentationNode=None, volumeNode=None, volumeName=None, addSegmentationToNode=False):
    """
    Crops a binary mask from the anterior direction from a specified distance.
    
    Parameters
    ----------
    binaryMaskArray : numpy.ndarray
                The binary mask to be cropped.
    referewnce_mask : numpy.ndarray
                The reference mask to be used for cropping.
    numberOfPixels : int
                The number of pixels to crop from.
    fromAnterior : bool, default: True
                Specification to crop the binary mask from the anterior.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    volumeName : str, default: None
                The name of the volume node to add the cropped binary mask to. Only needed if adding the segmentation to a node.
    addSegmentationToNode : bool, default: False
                Specification to add the cropped binary mask to a node.
                
    Returns
    -------
    numpy.ndarray
        The cropped binary mask.
    """
    if not isinstance(binaryMaskArray, np.ndarray):
        raise TypeError("The binaryMaskArray parameter must be a numpy.ndarray.")
    if not isinstance(referenceMask, np.ndarray):
        raise TypeError("The referenceMask parameter must be a numpy.ndarray.")
    if not isinstance(numberOfPixels, int):
        raise TypeError("The numberOfPixels parameter must be an integer.")
    if not isinstance(fromAnterior, bool):
        raise TypeError("The fromAnterior parameter must be a boolean.")

    binaryMaskArray = np.copy(binaryMaskArray)

    if fromAnterior:
        most_anterior =  []

        for slice in referenceMask:
            for index in range(np.shape(slice)[0]):
                row = slice[index,:]
                if 1 in row:
                    most_anterior.append(index)
                    break

        binaryMaskArray[:, : min(most_anterior) - numberOfPixels,:] = 0

    else:
        most_posterior =  []

        for slice in referenceMask:
            for index in range(np.shape(slice)[0]):
                row = slice[index,:]
                most_posterior_in_slice = 0
                if 1 in row:
                    if index > most_posterior_in_slice:
                        most_posterior_in_slice = index
                most_posterior.append(most_posterior_in_slice)

        binaryMaskArray[:,: max(most_posterior) - numberOfPixels,:] = 0

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
            raise TypeError("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
        if not isinstance(volumeName, str):
            raise TypeError("The volumeName parameter must be a string.")
        add_segmentation_array_to_node(segmentationNode, binaryMaskArray, volumeName, volumeNode)

    return binaryMaskArray



#################################################################################
# Creation functions                                                            #
#################################################################################

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
            raise TypeError("The segmentName parameter must be a string.")
        
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


def create_binary_mask_between_slices(referenceMaskArray, leftBound, rightBound, segmentationNode=None, volumeNode=None, volumeName=None, addSegmentationToNode=False):
    """
    Creates a binary mask between two slice numbers.
    
    Parameters
    ----------
    referenceMaskArray : numpy.ndarray
                The reference mask to create the binary mask from.
    leftBound : int
                The left bound of the binary mask.
    rightBound : int
                The right bound of the binary mask.
    segmentationNode : slicer.vtkMRMLSegmentationNode, default: None
                The segmentation node to add the binary mask to. Only needed if adding the segmentation to a node.
    volumeNode : slicer.vtkMRMLVolumeNode, default: None
                The volume node to add the binary mask to. Only needed if adding the segmentation to a node.
    volumeName : str, default: None
                The name of the volume node to add the binary mask to. Only needed if adding the segmentation to a node.
    addSegmentationToNode : bool, default: False
                Specification to add the binary mask to a node.
                
    Returns
    -------
    numpy.ndarray
        The binary mask between the two binary masks.
    """
    if not isinstance(referenceMaskArray, np.ndarray):
        raise TypeError("The referenceMaskArray parameter must be a numpy.ndarray.")
    if not isinstance(leftBound, int):
        raise TypeError("The leftBound parameter must be an integer.")
    if not isinstance(rightBound, int):
        raise TypeError("The rightBound parameter must be an integer.")

    mask_w_ones = np.ones_like(referenceMaskArray)

    mask_w_ones[:,:, : rightBound] = 0
    mask_w_ones[:,:, leftBound :] = 0

    if addSegmentationToNode:
        if not isinstance(segmentationNode, slicer.vtkMRMLSegmentationNode):
            raise TypeError("The segmentationNode parameter must be a slicer.vtkMRMLSegmentationNode.")
        if not isinstance(volumeNode, slicer.vtkMRMLVolumeNode):
            raise TypeError("The volumeNode parameter must be a slicer.vtkMRMLVolumeNode.")
        if not isinstance(volumeName, str):
            raise TypeError("The volumeName parameter must be a string.")
        add_segmentation_array_to_node(segmentationNode, mask_w_ones, volumeName, volumeNode)

    return mask_w_ones



#################################################################################
# Save functions                                                                #
#################################################################################

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
    additionalSaveInfo : str, default: None
                Additional information to add to the file name.

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


def save_segmentations_to_files(TupleOfSegmentNamesToSave, SegmentationNode, ReferenceVolumeNode, SaveDirectory, FileType, ExtraSaveInfo=None):
    """
    This function saves the specified segmentations to files. The function works on one Segmentation Node at a time.

    Parameters
    ----------
    TupleOfSegmentNamesToSave : tuple
                The tuple of segment names to save.
    SegmentationNode : vtkMRMLSegmentationNode
                The segmentation node to save the segmentations from.
    ReferenceVolumeNode : vtkMRMLScalarVolumeNode
                The reference volume node to save the segmentations to.
    SaveDirectory : str
                The directory to save the segmentations to.
    FileType : str
                The file type to save the segmentations as. The options are "nrrd", "nii.gz", and "nii".
    ExtraSaveInfo : str, default: None
                Additional information to add to the file name.

    Returns
    -------
    bool
        True if the segmentations were saved successfully.
    """

    if not isinstance(TupleOfSegmentNamesToSave, tuple):
        raise ValueError("TupleOfSegmentNamesToSave must be a tuple.")
    if not isinstance(SegmentationNode, slicer.vtkMRMLSegmentationNode):
        raise ValueError("SegmentationNode must be a vtkMRMLSegmentationNode.")
    if not isinstance(ReferenceVolumeNode, slicer.vtkMRMLScalarVolumeNode):
        raise ValueError("ReferenceVolumeNode must be a vtkMRMLScalarVolumeNode.")
    if FileType not in ["nrrd", "nii.gz", "nii"]:
        raise ValueError("FileType must be 'nrrd', 'nii.gz', or 'nii'.")
    if not os.path.isdir(SaveDirectory):
        raise ValueError("SaveDirectory must be a valid directory.")
    if ExtraSaveInfo is not None:
        if not isinstance(ExtraSaveInfo, str):
            raise ValueError("ExtraSaveInfo must be a string.")

    try:
        logger.debug("Checking all of the available segments in the segmentation node.")
        available_segments = tuple(SegmentationNode.GetSegmentation().GetNthSegment(i).GetName() for i in range(SegmentationNode.GetSegmentation().GetNumberOfSegments()))    

        # Check to see if the segment names are in the provided segmentation nodes
        if len(TupleOfSegmentNamesToSave) == 1:
            if not TupleOfSegmentNamesToSave[0] in available_segments:
                raise ValueError(f"The segment name {TupleOfSegmentNamesToSave[0]} is not in the SegmentationNode.")
            else:
                segment_IDs_to_save = [SegmentationNode.GetSegmentation().GetSegmentIdBySegmentName(TupleOfSegmentNamesToSave[0]), ]
                logger.debug(f"Segment ID to save: {segment_IDs_to_save}")

        else:
            segment_IDs_to_save = []
            for segment_name in TupleOfSegmentNamesToSave:
                if not segment_name in available_segments:
                    raise ValueError(f"The segment name {segment_name} is not in the SegmentationNode.")
                else:
                    segment_IDs_to_save.append(SegmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment_name))
                    logger.debug(f"Segment IDs to save: {segment_IDs_to_save}")

        #Create the label map volume node
        logger.debug("Creating the label map volume node.")
        label_map_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{SegmentationNode.GetName()}_label_map_node")
    
        # Export the segments to the label map volume node
        logger.debug("Exporting the segments to the label map volume node.")
        slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(SegmentationNode, segment_IDs_to_save, label_map_volume_node, ReferenceVolumeNode)

        # Save the label map volume node to a file
        logger.debug("Saving the label map volume node to a file.")
        if ExtraSaveInfo is None:
            slicer.util.saveNode(label_map_volume_node, os.path.join(SaveDirectory, f"{label_map_volume_node.GetName().split('_label_map_node')[0]}.{FileType}"))
        else:
            slicer.util.saveNode(label_map_volume_node, os.path.join(SaveDirectory, f"{ExtraSaveInfo}_{label_map_volume_node.GetName().split('_label_map_node')[0]}.{FileType}"))

        # Remove the label map volume node
        logger.debug("Removing the label map volume node.")
        slicer.mrmlScene.RemoveNode(label_map_volume_node)

        return True

    except Exception:
        logger.exception("An error occured in save_segmentations_to_files")
        raise

