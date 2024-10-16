import slicer
import numpy as np
from segmentation_node import SegmentationNode
from utils import log_and_raise, check_type
import logging
import os
from utils import TempNodeManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Segment:
    """
    Class to handle the segments of a segmentation node in 3D Slicer.
    
    Attributes
    ----------
    segmentationNode : SegmentationNode
        The segmentation node to which the segment belongs.
    segmentObject : slicer.vtkMRMLSegment
        The segment object in Slicer.
    name : str
        The name of the segment.
    NumPyArray : np.ndarray
        The NumPy array associated with the segment.
    hasArray : bool
        A boolean indicating if the segment has a NumPy array associated with it.
    associatedVolume : slicer.vtkMRMLScalarVolumeNode
        The volume node associated with the segment.
        
    Methods
    -------
    description() -> str
        Get the description of the segment.
    get_name() -> str
        Get the name of the segment.
    get_id() -> str
        Get the ID of the segment.
    edit_name(newName: str) -> None
        Edit the name of the segment.
    delete() -> None
        Delete the segment.
    delete_array() -> None
        Delete the NumPy array associated with the segment.
    has_array() -> bool
        Check if the segment has a NumPy array associated with it.
    get_color() -> tuple
        Get the color of the segment.
    set_color(color: tuple) -> None
        Set the color of the segment.
    set_associated_volume(volumeNode: slicer.vtkMRMLScalarVolumeNode) -> None
        Set the associated volume node of the segment.
    get_array_from_slicer(referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, updateClass: bool = True) -> np.ndarray
        Get the NumPy array associated with the segment from Slicer.
    set_array(array: np.ndarray, associatedVolume: slicer.vtkMRMLScalarVolumeNode, updateSlicer: bool = False) -> None
        Set the NumPy array associated with the segment.
    update_slicer() -> None
        Update the segment in Slicer with the NumPy array.
    copy(newName: str) -> Segment
        Copy the segment.
    dice_similarity(segmentationArray: np.ndarray) -> float
        Calculate the Dice similarity coefficient between the segment and a segmentation array.
    margin_editor_effect(volumeNode: slicer.vtkMRMLScalarVolumeNode, operation: str = 'Grow', MarginSize: float = 10.0) -> None
        Apply the margin effect to the segment in Slicer.
    logical_operator_slicer(secondarySegment: slicer.vtkMRMLSegment, volumeNode: slicer.vtkMRMLScalarVolumeNode, operator='copy') -> None
        Apply logical operations to two segmentations in Slicer.
    segmentation_by_auto_threshold(volumeNode: slicer.vtkMRMLScalarVolumeNode, threshold='OTSU') -> None
        Apply the autosegmentation threshold to a new segmentation in Slicer.
    islands_effect_segment_editor(volumeNode: slicer.vtkMRMLScalarVolumeNode, edit: str = 'KEEP_LARGEST_ISLAND', minimumSize: int = 5) -> None
        Apply the 'Islands' effect in the Segment Editor of 3D Slicer to a given segment.
    combine_masks_from_arrays(NumPyArray: np.ndarray, update_slicer: bool = True) -> None
        Combine the segment mask with another mask.
    subtract_masks_from_arrays(NumPyArray: np.ndarray, update_slicer: bool = True) -> None
        Subtract another mask from the segment mask.
    intersect_masks_from_arrays(NumPyArray: np.ndarray, update_slicer: bool = True) -> None
        Intersect the segment mask with another mask.
    remove_mask_superior_to_slice(sliceNumber: int, update_slicer: bool = True) -> None
        Remove the mask superior to a given slice.
    remove_mask_inferior_to_slice(sliceNumber: int, update_slicer: bool = True) -> None
        Remove the mask inferior to a given slice.
    remove_mask_anterior_to_slice(sliceNumber: int, update_slicer: bool = True) -> None
        Remove the mask anterior to a given slice.
    remove_mask_posterior_to_slice(sliceNumber: int, update_slicer: bool = True) -> None
        Remove the mask posterior to a given slice.
    save_segment_to_file(SaveDirectory: str, ReferenceVolumeNode: slicer.vtkMRMLScalarVolumeNode) -> None
        Save the segment to a file in the specified directory.
        """
    
    def __init__(self, SegmentationNode: SegmentationNode, segmentObject, segmentName: str = None):
        self.segmentationNode = SegmentationNode
        self.segmentObject = segmentObject
        if segmentName == None:
            self.name = self.segmentObject.GetName()
        else:
            self.name = segmentName
            self.segmentObject.SetName(segmentName)
        self.NumPyArray = None
        self.hasArray = False
        self.associatedVolume = None
        

    def description(self) -> str:
        """
        Get the description of the segment.
        """
        return f"Segment {self.name} with ID {self.segmentID} in segmentation {self.segmentationNode.name}"
    

    def get_name(self) -> str:
        """
        Get the name of the segment.
        """
        return self.name
    

    def get_id(self) -> str:
        """
        Get the ID of the segment.
        """
        return self.segmentObject.GetSegmentID()
    

    def edit_name(self, newName: str) -> None:
        """
        Edit the name of the segment.
        """
        self.segmentObject.SetName(newName)
        self.name = newName


    def delete(self) -> None:
        """
        Delete the segment.
        """
        self.segmentationNode.remove_segment(self)


    def delete_array(self) -> None:
        """
        Delete the NumPy array associated with the segment.
        """
        self.NumPyArray = None
        self.hasArray = False
        self.associatedVolume = None


    def has_array(self) -> bool:
        """
        Check if the segment has a NumPy array associated with it.
        """
        return self.hasArray
    

    def get_color(self) -> tuple:
        """
        Get the color of the segment.
        """
        return self.segmentObject.GetColor()
    

    def set_color(self, color: tuple) -> None:
        """
        Set the color of the segment.
        
        Parameters
        ----------
        color : tuple
            The color to set the segment to.
        """
        check_type(color, tuple, 'color')
        self.segmentObject.SetColor(color[0], color[1], color[2])


    def set_associated_volume(self, volumeNode: slicer.vtkMRMLScalarVolumeNode) -> None:
        """
        Set the associated volume node of the segment.

        Parameters
        ----------
        volumeNode : slicer.vtkMRMLScalarVolumeNode
            The volume node to associate with the segment.
        """
        check_type(volumeNode, slicer.vtkMRMLScalarVolumeNode, 'volumeNode')
        self.associatedVolume = volumeNode


    def get_array_from_slicer(self, referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, updateClass: bool = True) -> np.ndarray:
        """
        Get the NumPy array associated with the segment from Slicer.

        Parameters
        ----------
        referenceVolumeNode : slicer.vtkMRMLScalarVolumeNode
            The volume node to use as a reference.
        updateClass : bool, optional
            A boolean indicating if the class should be updated. Default is True.

        Returns
        -------
        np.ndarray
            The NumPy array associated with the segment.
        """
        try:
            check_type(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode, 'referenceVolumeNode')
            segment_array = slicer.util.arrayFromSegmentBinaryLabelmap(self.segmentObject, self.segmentID, referenceVolumeNode)
            if updateClass:
                self.set_array(segment_array, False)
                self.associatedVolume = referenceVolumeNode
            return segment_array
        except Exception as e:
            log_and_raise(logger, "An error occurred in getArrayFromSlicer", type(e))


    def set_array(self, array: np.ndarray, associatedVolume: slicer.vtkMRMLScalarVolumeNode, updateSlicer: bool = False) -> None:
        """
        Set the NumPy array associated with the segment.

        Parameters
        ----------
        array : np.ndarray
            The NumPy array to set.
        associatedVolume : slicer.vtkMRMLScalarVolumeNode
            The volume node associated with the segment.
        updateSlicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is False.
        """
        try:
            self.NumPyArray = array
            self.hasArray = True
            self.associatedVolume = associatedVolume
            if updateSlicer:
                self.update_slicer()  
        except Exception as e:
            log_and_raise(logger, "An error occurred in setArray", type(e))


    def update_slicer(self) -> None:
        """
        Update the segment in Slicer with the NumPy array.
        """
        try:
            if not self.hasArray:
                logger.warning("No array found. Cannot update Slicer.")
            if not self.associatedVolume:
                logger.warning("No associated volume found. Cannot update Slicer.")
            slicer.util.updateSegmentBinaryLabelmapFromArray(self.NumPyArray, self.segmentObject, self.get_id(), self.associatedVolume)
        except Exception as e:
            log_and_raise(logger, "An error occurred in updateSlicer", type(e))


    def copy(self, newName: str) -> Segment:
        """
        Copy the segment.
        
        Parameters
        ----------
        newName : str
            The name of the new segment.
            
        Returns
        -------
        Segment
            The new segment.
        """
        return self.segmentationNode.copy_segment(self.name, newName)
        
    
    def dice_similarity(self, segmentationArray: np.ndarray) -> float:
        """
        Calculate the Dice similarity coefficient between the segment and a segmentation array.

        Parameters
        ----------
        segmentationArray : np.ndarray
            The segmentation array to compare to the segment.

        Returns
        -------
        float
            The Dice similarity coefficient.
        """
        try:
            check_type(segmentationArray, np.ndarray, 'segmentationArray')
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if self.NumPyArray.shape != segmentationArray.shape:
                raise ValueError("The shape of the segmentation array does not match the shape of the segment.")
            logger.info(f"Calculating the Dice similarity coefficient for segment {self.name}")
            intersection = np.sum(self.NumPyArray * segmentationArray)
            sum_masks = np.sum(self.NumPyArray) + np.sum(segmentationArray)
            if sum_masks == 0:
                raise ValueError("The sum of the masks is 0. both masks are empty.")
            else:
                return 2 * intersection / sum_masks
        except Exception as e:
            log_and_raise(logger, "An error occurred in diceSimilarity", type(e))


    def margin_editor_effect(self, volumeNode: slicer.vtkMRMLScalarVolumeNode, operation: str = 'Grow', MarginSize: float = 10.0) -> None:
        """
        Apply the margin effect to the segment in Slicer.

        Parameters
        ----------
        volumeNode : slicer.vtkMRMLScalarVolumeNode
            The volume node associated with the segmentation.
        operation : str, optional
            The operation to apply. Default is 'Grow'.
        MarginSize : float, optional
            The size of the margin. Default is 10.0.
        """
        check_type(volumeNode, slicer.vtkMRMLScalarVolumeNode, 'volumeNode')
        check_type(operation, str, 'operation')
        check_type(MarginSize, float, 'MarginSize')
        try:
            # Setup the segment editor widget
            logger.debug(f"Setting up segment editor widget")
            segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            segmentEditorWidget.setSegmentationNode(self.segmentationNode)
            segmentEditorWidget.setSourceVolumeNode(volumeNode)
            segmentEditorNode.SetOverwriteMode(2)  # i.e. "allow overlap" in UI
            segmentEditorNode.SetSelectedSegmentID(self.get_id())
            # Set up the margin effect
            logger.debug(f"Setting up margin effect")
            segmentEditorWidget.setActiveEffectByName("Margin")
            effect = segmentEditorWidget.activeEffect()
            logger.debug(f"Applying margin effect")
            # Set up the opperation
            if operation in ['Grow', 'Shrink']:
                effect.setParameter("Operation", operation)
            else:
                raise ValueError("The operation parameter must be 'Grow' or 'Shrink'.")
            if (isinstance(MarginSize, float) or isinstance(MarginSize, int)) and MarginSize > 0:
                effect.setParameter("MarginSizeMm", MarginSize)
            else:
                raise ValueError("Invalid MarginSize. MarginSize must be a positive number.")
            effect.self().onApply()
        except Exception as e:
            log_and_raise(logger, "An error occurred in marginEditorEffect", type(e))

    
    def logical_operator_slicer(self, secondarySegment: slicer.vtkMRMLSegment, volumeNode: slicer.vtkMRMLScalarVolumeNode, operator='copy') -> None:
        """
        Function to apply logical operations to two segmentations in slicer. The main segmentation is the segmentation that will be modified. The operator will select the type of transformation applied.
        
        Parameters
        ----------
        secondarySegment : slicer.vtkMRMLSegment
            The name of the segmentation to be used as a reference.
        operator : str, optional
            The operation to be applied. Default is 'copy'. Options are 'copy', 'union', 'intersect', 'subtract', 'invert', 'clear', 'fill'.
        volumeNode : slicer.vtkMRMLScalarVolumeNode, optional
            The volume node to be used. Default is the VolumeNode.
        
        Returns
        -------
        None
        """
        try:
            check_type(secondarySegment, slicer.vtkMRMLSegment, 'secondarySegment')
            check_type(operator, str, 'operator')
            # Get the segmentation editor widget
            segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            #Set the correct volumes to be used in the Segment Editor
            segmentEditorWidget.setSegmentationNode(self.segmentationNode)
            segmentEditorWidget.setSourceVolumeNode(volumeNode)
            # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
            segmentEditorNode.SetOverwriteMode(2) # i.e. "allow overlap" in UI
            # Get the segment ID
            segid_src = self.segmentObject
            segid_tgt = secondarySegment
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
            log_and_raise(logger, "An error occurred in logicalOperatorSlicer", type(e))


    def segmentation_by_auto_threshold(self, volumeNode: slicer.vtkMRMLScalarVolumeNode, threshold='OTSU') -> None:
        """
        Function to apply the autosegmentation threshold to a new segmentation in slicer defined by the segmentation name. This will overwrite the current segmentation.
        
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
        try:
            check_type(volumeNode, slicer.vtkMRMLScalarVolumeNode, 'volumeNode')
            check_type(threshold, str, 'threshold')
            # Get the segmentation editor widget
            segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            #Set the correct volumes to be used in the Segment Editor
            segmentEditorWidget.setSegmentationNode(self.segmentationNode)
            segmentEditorWidget.setSourceVolumeNode(volumeNode)
            # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
            segmentEditorNode.SetOverwriteMode(2)
            # Get the segment ID
            segid_src = self.segmentObject
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
        except Exception as e:
            log_and_raise(logger, "An error occurred in segmentationByAutoThreshold", type(e))


    def islands_effect_segment_editor(self, volumeNode: slicer.vtkMRMLScalarVolumeNode, edit: str = 'KEEP_LARGEST_ISLAND', minimumSize: int = 5):
        """
        Applies the 'Islands' effect in the Segment Editor of 3D Slicer to a given segment.
        
        Parameters
        ----------
        volumeNode : slicer.vtkMRMLScalarVolumeNode
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
        try:
            check_type(volumeNode, slicer.vtkMRMLScalarVolumeNode, 'volumeNode')
            check_type(edit, str, 'edit')
            check_type(minimumSize, int, 'minimumSize')
            if not isinstance(edit, str) or edit not in edit_options_size + edit_options_others:
                raise ValueError("edit must be a string and one of 'KEEP_LARGEST_ISLAND', 'KEEP_SELECTED_ISLAND', 'REMOVE_SMALL_ISLANDS', 'REMOVE_SELECTED_ISLAND', 'ADD_SELECTED_ISLAND', 'SPLIT_ISLANDS_TO_SEGMENTS'.")
            if not isinstance(minimumSize, int) or minimumSize < 0 or minimumSize > 1000:
                raise ValueError("minimumSize must be an integer between 0 and 1000.")
            logger.info("Applying Islands effect to segment.")
            # Get the segmentation editor widget
            segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            # Set the correct volumes to be used in the Segment Editor
            segmentEditorWidget.setSegmentationNode(self.segmentationNode)
            segmentEditorWidget.setSourceVolumeNode(volumeNode)
            # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
            segmentEditorNode.SetOverwriteMode(2)
            # Get the segment ID
            segmentEditorNode.SetSelectedSegmentID(self.get_id())
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
            log_and_raise(logger, "An error occurred in islandsEffectSegmentEditor", type(e))   


    def combine_masks_from_arrays(self, NumPyArray: np.ndarray, update_slicer: bool = True) -> None:
        """
        Combine the segment mask with another mask.

        Parameters
        ----------
        NumPyArray : np.ndarray
            The NumPy array to combine with the segment mask.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(NumPyArray, np.ndarray, 'NumPyArray')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if self.NumPyArray.shape != NumPyArray.shape:
                raise ValueError("The shape of the input array does not match the shape of the segment.")
            combined_array = np.logical_or(self.NumPyArray, NumPyArray)
            self.set_array(combined_array, self.associatedVolume, update_slicer)
        except Exception as e:
            log_and_raise(logger, "An error occurred in combineMasksFromArrays", type(e))

    
    def subtract_masks_from_arrays(self, NumPyArray: np.ndarray, update_slicer: bool = True) -> None:
        """
        Subtract another mask from the segment mask.

        Parameters
        ----------
        NumPyArray : np.ndarray
            The NumPy array to subtract from the segment mask.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(NumPyArray, np.ndarray, 'NumPyArray')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if self.NumPyArray.shape != NumPyArray.shape:
                raise ValueError("The shape of the input array does not match the shape of the segment.")
            subtracted_array = np.logical_and(self.NumPyArray, np.logical_not(NumPyArray))
            self.set_array(subtracted_array, self.associatedVolume, update_slicer)
        except Exception as e:
            log_and_raise(logger, "An error occurred in subtractMasksFromArrays", type(e))

    
    def intersect_masks_from_arrays(self, NumPyArray: np.ndarray, update_slicer: bool = True) -> None:
        """
        Intersect the segment mask with another mask.

        Parameters
        ----------
        NumPyArray : np.ndarray
            The NumPy array to intersect with the segment mask.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(NumPyArray, np.ndarray, 'NumPyArray')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if self.NumPyArray.shape != NumPyArray.shape:
                raise ValueError("The shape of the input array does not match the shape of the segment.")
            intersected_array = np.logical_and(self.NumPyArray, NumPyArray)
            self.set_array(intersected_array, self.associatedVolume, update_slicer)
        except Exception as e:
            log_and_raise(logger, "An error occurred in intersectMasksFromArrays", type(e))


    def remove_mask_superior_to_slice(self, sliceNumber: int, update_slicer: bool = True) -> None:
        """
        Remove the mask superior to a given slice.

        Parameters
        ----------
        sliceNumber : int
            The slice number.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(sliceNumber, int, 'sliceNumber')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if sliceNumber < 0 or sliceNumber >= self.NumPyArray.shape[0]:
                raise ValueError("The sliceNumber is out of range.")
            removed_array = np.copy(self.NumPyArray)
            removed_array[sliceNumber + 1:, :, :] = False
            self.set_array(removed_array, self.associatedVolume, update_slicer)
            del removed_array
        except Exception as e:
            log_and_raise(logger, "An error occurred in removeMaskSuperiorToSlice", type(e))


    def remove_mask_inferior_to_slice(self, sliceNumber: int, update_slicer: bool = True) -> None:
        """
        Remove the mask inferior to a given slice.

        Parameters
        ----------
        sliceNumber : int
            The slice number.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(sliceNumber, int, 'sliceNumber')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if sliceNumber < 0 or sliceNumber >= self.NumPyArray.shape[0]:
                raise ValueError("The sliceNumber is out of range.")
            removed_array = np.copy(self.NumPyArray)
            removed_array[:sliceNumber, :, :] = False
            self.set_array(removed_array, self.associatedVolume, update_slicer)
            del removed_array
        except Exception as e:
            log_and_raise(logger, "An error occurred in removeMaskInferiorToSlice", type(e))


    def remove_mask_anterior_to_slice(self, sliceNumber: int, update_slicer: bool = True) -> None:
        """
        Remove the mask anterior to a given slice.
        
        Parameters
        ----------
        sliceNumber : int
            The slice number.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(sliceNumber, int, 'sliceNumber')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if sliceNumber < 0 or sliceNumber >= self.NumPyArray.shape[1]:
                raise ValueError("The sliceNumber is out of range.")
            removed_array = np.copy(self.NumPyArray)
            removed_array[:, sliceNumber + 1:, :] = False
            self.set_array(removed_array, self.associatedVolume, update_slicer)
            del removed_array
        except Exception as e:
            log_and_raise(logger, "An error occurred in removeMaskAnteriorToSlice", type(e))

    
    def remove_mask_posterior_to_slice(self, sliceNumber: int, update_slicer: bool = True) -> bool:
        """
        Remove the mask posterior to a given slice.

        Parameters
        ----------
        sliceNumber : int
            The slice number.
        update_slicer : bool, optional
            A boolean indicating if Slicer should be updated. Default is True.
        """
        check_type(sliceNumber, int, 'sliceNumber')
        check_type(update_slicer, bool, 'update_slicer')
        try:
            if not self.hasArray:
                if self.associatedVolume == None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                self.get_array_from_slicer(self.associatedVolume)
            if sliceNumber < 0 or sliceNumber >= self.NumPyArray.shape[1]:
                raise ValueError("The sliceNumber is out of range.")
            removed_array = np.copy(self.NumPyArray)
            removed_array[:, :sliceNumber, :] = False
            self.set_array(removed_array, self.associatedVolume, update_slicer)
            del removed_array
        except Exception as e:
            log_and_raise(logger, "An error occurred in removeMaskPosteriorToSlice", type(e))

    
    def save_segment_to_files(self, SaveDirectory: str, referenceVolumeNode=None, FileType: str='nii', ExtraSaveInfo=None) -> None:
        """
        Save the segment to a file in the specified directory.

        Parameters
        ----------
        SaveDirectory : str
            The directory to save the segment to.
        referenceVolumeNode : slicer.vtkMRMLScalarVolumeNode, optional
            The volume node to use as a reference. Default is None.
        FileType : str, optional
        ExtraSaveInfo : str, optional
            Extra information to add to the file name. Default is None.
        """
        check_type(SaveDirectory, str, 'SaveDir')
        check_type(FileType, str, 'FileType')
        if FileType not in ["nrrd", "nii.gz", "nii"]:
            raise ValueError("FileType must be 'nrrd', 'nii.gz', or 'nii'.")
        if not os.path.isdir(SaveDirectory):
            raise ValueError("SaveDirectory must be a valid directory.")
        if ExtraSaveInfo is not None:
            check_type(ExtraSaveInfo, str, 'ExtraSaveInfo')
        try:
            if referenceVolumeNode is None:
                if self.associatedVolume is None:
                    raise ValueError("No associated volume found. Please set an associated volumeNode.")
                referenceVolumeNode = self.associatedVolume
            segmentIdToSave = [self.get_id(), ]
            logger.debug(f"Segment ID to save: {segmentIdToSave}")
            
            # Use TempNodeManager to manage the temporary label map volume node
            with TempNodeManager("vtkMRMLLabelMapVolumeNode", f"{SegmentationNode.GetName()}_label_map_node") as label_map_volume_node:
                # Export the segments to the label map volume node
                logger.debug("Exporting the segments to the label map volume node.")
                slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(SegmentationNode, segmentIdToSave, label_map_volume_node, referenceVolumeNode)
                
                # Save the label map volume node to a file
                logger.debug("Saving the label map volume node to a file.")
                if ExtraSaveInfo is None:
                    slicer.util.saveNode(label_map_volume_node, os.path.join(SaveDirectory, f"{label_map_volume_node.GetName().split('_label_map_node')[0]}.{FileType}"))
                else:
                    slicer.util.saveNode(label_map_volume_node, os.path.join(SaveDirectory, f"{ExtraSaveInfo}_{label_map_volume_node.GetName().split('_label_map_node')[0]}.{FileType}"))
            
            return True
        except Exception as e:
            log_and_raise(logger, "An error occurred in saveSegmentToFiles", type(e))

