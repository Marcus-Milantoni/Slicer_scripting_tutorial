import slicer, vtk
import numpy as np
import os
import logging
from segment import Segment
from utils import check_type, log_and_raise, TempNodeManager


# Setup the logger (can customize)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SegmentationNode:
    """
    Class to handle the segmentation nodes in 3D Slicer.
    
    Attributes
    ----------
    segmentationNode : slicer.vtkMRMLSegmentationNode
        The segmentation node in Slicer.
    name : str
        The name of the segmentation node.
    segments : list
        The list of segments in the segmentation node.
    representation : str
        The representation of the segmentation node.
    
    Methods
    -------
    description() -> str
        Get the description of the segmentation node.
    get_name() -> str
        Get the name of the segmentation node.
    set_binary_labelmaprepresentation() -> None
        Set the representation of the segmentation node to binary labelmap.
    set_closed_surface_representation() -> None
        Set the representation of the segmentation node to closed surface.
    get_segments() -> tuple
        Get the segments of the segmentation node.
    get_segment_names() -> tuple
        Get the names of the segments of the segmentation node.
    get_number_of_segments() -> int
        Get the number of segments in the segmentation node.
    get_segment_by_segment_name(segmentName: str) -> Segment
        Get a segment by its name.
    edit_name(newName : str) -> None
        Edit the name of the segmentation node.
    _add_segment(segmentObject: slicer.vtkMRMLSegment, segmentName: str = None) -> None
        Add a segment to the segmentation node.
    remove_segment(segment: Segment) -> None
        Remove a segment from the segmentation node.    
    remove_segment_by_name(segmentName: str) -> None
        Remove a segment by its name.
    clear_segments() -> None
        Clear all segments from the segmentation node.
    add_blank_segmentation(segmentName: str) -> Segment
        Add a blank segment to the segmentation node.
    add_segment_from_array(segmentName: str, segmentArray: np.ndarray, referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, color: tuple = None) -> Segment
        Add a segment to the segmentation node from a numpy array.
    copy_segmentation(segmentName, newSegmentName)
        This function copies a segment in a segmentation node.
    save_segmentations_by_name(segmentNamesToSave: list, ReferenceVolumeNode: slicer.vtkMRMLVolumeNode, SaveDirectory: str, FileType: str = 'nii', ExtraSaveInfo: str = None)
        This function saves the specified segmentations to files.
    save_all_segmentations(ReferenceVolumeNode: slicer.vtkMRMLVolumeNode, SaveDirectory: str, FileType: str = 'nii', ExtraSaveInfo: str = None)
        This function saves all segmentations in the Segmentation Node to files.
    """

    def __init__(self, SegmentationNodeObject: slicer.vtkMRMLSegmentationNode, name: str = None):
        self.segmentationNode = SegmentationNodeObject
        self._segmentation = self.segmentationNode.GetSegmentation()
        self.nodeID = self.segmentationNode.GetID()
        self.segments = [] # List to store Segment Instances
        self.representation = None
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
        """ Get the description of the segmentation node. """
        return f"Name: {self.name}, Number of segments: {len(self.segments)}, Representation: {self.representation}"


    def get_name(self) -> str:
        """ Get the name of the segmentation node. """
        return self.name


    def set_binary_labelmaprepresentation(self) -> None:
        """ Set the representation of the segmentation node to binary labelmap."""
        self.segmentationNode.CreateBinaryLabelmapRepresentation()
        self.segmentationNode.SetSourceRepresentationToBinaryLabelmap()
        self.representation = 'binary labelmap'


    def set_closed_surface_representation(self) -> None:
        """ Set the representation of the segmentation node to closed surface."""
        self.segmentationNode.CreateClosedSurfaceRepresentation()
        self.segmentationNode.SetSourceRepresentationToClosedSurface()
        self.representation = 'closed surface'

    def get_segments(self) -> tuple:
        """ Get the segments of the segmentation node. """
        return tuple(self.segments) 


    def get_segment_names(self) -> tuple:
        """ Get the names of the segments of the segmentation node."""
        segment_names = [segment.get_name() for segment in self.segments]
        return tuple(segment_names)


    def get_number_of_segments(self) -> int:
        """ Get the number of segments in the segmentation node."""
        return len(self.segments)


    def get_segment_by_segment_name(self, segmentName: str) -> Segment:
        """ 
        Get a segment by its name.
        
        Parameters
        ----------
        segmentName : str
            The name of the segment to get.
        """
        for segment in self.segments:
            if segment.getName() == segmentName:
                return segment
        logger.warning(f"No segment with the name {segmentName} was found.")


    def edit_name(self, newName : str) -> None:
        """
        Edit the name of the segmentation node.
        
        Parameters
        ----------
        newName : str
            The new name of the segmentation node.
        """
        self.name = newName
        self.segmentationNode.SetName(newName)


    def _add_segment(self, segmentObject: slicer.vtkMRMLSegment, segmentName: str = None) -> None:
        """
        Add a segment to the segmentation node.
        
        Parameters
        ----------
        segmentObject : slicer.vtkMRMLSegment
            The segment object to add.
        segmentName : str, optional
            The name of the segment. Default is None.
            """
        check_type(segmentObject, slicer.vtkMRMLSegment, 'segmentObject')
        try:
            if segmentName == None:
                segment_object = Segment(self, segmentObject) 
                self.segments.append(segment_object)
                return segment_object
            else:
                segment_object = Segment(self, segmentObject, segmentName)
                self.segments.append(segment_object)
                return segment_object
        except Exception as e:
            log_and_raise(logger, "An error occurred in _add_segment", type(e))


    def remove_segment(self, segment: Segment) -> None:
        """
        Remove a segment from the segmentation node.
        
        Parameters
        ----------
        segment : Segment
            The segment to remove.
        """
        check_type(segment, Segment, 'segment')
        self.segments.remove(segment)
        self._segmentation.RemoveSegment(segment.getSegmentID())
        del segment


    def remove_segment_by_name(self, segmentName: str) -> None:
        """
        Remove a segment by its name.

        Parameters
        ----------
        segmentName : str
            The name of the segment to remove.
        """
        for segment in self.segments:
            if segment.getName() == segmentName:
                self.remove_segment(segment)
                return
        logger.warning(f"No segment with the name {segmentName} was found.")


    def clear_segments(self) -> None:
        """Clear all segments from the segmentation node."""
        for segment in self.segments:
            self._segmentation.RemoveSegment(segment.getSegmentID())
        self.segments.clear()


    def add_blank_segmentation(self, segmentName: str) -> Segment:
        """
        Add a blank segment to the segmentation node.
        
        Parameters
        ----------
        segmentName : str
            The name of the segment to add.
        """
        check_type(segmentName, str, 'segmentName')
        try:
            logger.info(f"Adding a blank segment with name {segmentName}")
            segmentID = self._segmentation.GetSegmentIdBySegmentName(segmentName)
            if segmentID == None:            
                segmentID = self._segmentation.AddEmptySegment(segmentName)
                segmentObject = self._segmentation.GetNthSegment(segmentID)
                return self._add_segment(segmentObject, segmentName)
            else:
                logger.warning(f"A segment with the name {segmentName} already exists.")
        except Exception as e:
            log_and_raise(logger, "An error occurred in addBlankSegmentation", type(e))


    def add_segment_from_array(self, segmentName: str, segmentArray: np.ndarray, referenceVolumeNode: slicer.vtkMRMLScalarVolumeNode, color: tuple = None) -> Segment:
        """
        Add a segment to the segmentation node from a numpy array.
        
        Parameters
        ----------
        segmentName : str
            The name of the segment to add.
        segmentArray : np.ndarray
            The numpy array to add as a segment.
        referenceVolumeNode : slicer.vtkMRMLScalarVolumeNode
            The reference volume node to add the segment to.
        color : tuple, optional
            The color of the segment. Default is None.
        """
        check_type(segmentName, str, 'segmentName')
        check_type(segmentArray, np.ndarray, 'segmentArray')
        check_type(referenceVolumeNode, slicer.vtkMRMLScalarVolumeNode, 'referenceVolumeNode')
        check_type(color, (tuple, type(None)), 'color')
        if color is not None:
            if len(color) != 3 or not all(isinstance(color[i], (int, float)) for i in range(3)):
                raise TypeError("The color parameter must be a tuple of three integers or floats.")
        else:
            color = tuple(np.random.rand(3))
        if self.representation != 'binary labelmap':
            self.set_binary_labelmaprepresentation()
        try:
            with TempNodeManager('vtkMRMLScalarVolumeNode', segmentName) as tempVolumeNode:
                tempVolumeNode.CopyOrientation(referenceVolumeNode)
                tempVolumeNode.SetSpacing(referenceVolumeNode.GetSpacing())
                slicer.util.updateVolumeFromArray(tempVolumeNode, segmentArray)
                tempImageData = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(tempVolumeNode)
                segmentID = self.segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(tempImageData, segmentName, color)
                segmentObject = self._segmentation.GetSegmentBySegmentID(segmentID)
                return self._add_segment(segmentObject, segmentName)
        except Exception as e:
            log_and_raise(logger, "An error occurred in addSegmentFromArray", type(e))


    def copy_segmentation(self, segmentName, newSegmentName):
        """
        This function copies a segment in a segmentation node.

        Parameters
        ----------
        segmentName : str
                    The name of the segment to copy.
        newSegmentName : str
                    The name of the new segment.

        Returns
        -------
        Segment
            The new segment object.
        
        Raises
        ------
        TypeError
            If the segmentationNode is not a vtkMRMLSegmentationNode.
            If the segmentName is not a string.
            If the newSegmentName is not a string.
        """
        check_type(segmentName, str, 'segmentName')
        check_type(newSegmentName, str, 'newSegmentName') 
        try:
            logger.info(f"Copying segment {segmentName} to {newSegmentName}")
            logger.debug(f"Setting up segment editor widget")
            segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            segmentEditorWidget.setSegmentationNode(self.segmentationNode)
            volumeNode = slicer.util.getNodesByClass("vtkMRMLVolumeNode")[0]
            segmentEditorWidget.setSourceVolumeNode(volumeNode)
            # Set overwrite mode: 0/1/2 -> overwrite all/visible/none
            segmentEditorNode.SetOverwriteMode(2)  # i.e. "allow overlap" in UI
            # Get the segment IDs
            logger.debug(f"getting the segment IDs")
            self.segmentationNode.AddSegmentFromClosedSurfaceRepresentation(vtk.vtkPolyData(), newSegmentName)
            targetSegmentID = self.segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(newSegmentName)
            modifierSegmentID = self.segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
            logger.debug(f"Setting the parameters for the logical operators effect")
            segmentEditorNode.SetSelectedSegmentID(targetSegmentID)
            segmentEditorWidget.setActiveEffectByName("Logical operators")
            effect = segmentEditorWidget.activeEffect()
            effect.setParameter("Operation", "COPY")  # change the operation here
            effect.setParameter("ModifierSegmentID", modifierSegmentID)
            effect.self().onApply()
            segmentObject = self._segmentation.GetSegmentBySegmentID(targetSegmentID)
            return self._add_segment(segmentObject, newSegmentName)
        except Exception:
            logger.exception("An error occurred in copy_segmentation")
            raise


    def save_segmentations_by_name(self, segmentNamesToSave: list, ReferenceVolumeNode: slicer.vtkMRMLVolumeNode, SaveDirectory: str, FileType: str = 'nii', ExtraSaveInfo: str = None):
        """
        This function saves the specified segmentations to files. The function works on one Segmentation Node at a time.

        Parameters
        ----------
        TupleOfSegmentNamesToSave : tuple
                    The tuple of segment names to save.
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
        try:
            check_type(segmentNamesToSave, list, 'segmentNamesToSave')
            check_type(ReferenceVolumeNode, slicer.vtkMRMLScalarVolumeNode, 'ReferenceVolumeNode')
            check_type(SaveDirectory, str, 'SaveDirectory')
            check_type(FileType, str, 'FileType')
            if FileType not in ['nrrd', 'nii.gz', 'nii']:
                raise ValueError("FileType must be either 'nrrd', 'nii.gz', or 'nii'.")
            if ExtraSaveInfo is not None:
                check_type(ExtraSaveInfo, str, 'ExtraSaveInfo')

            available_segments = tuple(self.segmentationNode.GetSegmentation().GetNthSegment(i).GetName() for i in range(self.segmentationNode.GetSegmentation().GetNumberOfSegments()))    

            # Check to see if the segment names are in the provided segmentation nodes
            segment_IDs_to_save = []
            for segment_name in list(segmentNamesToSave):
                if not segment_name in available_segments:
                    raise ValueError(f"The segment name {segment_name} is not in the SegmentationNode.")
                else:
                    segment_IDs_to_save.append(self.segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment_name))
                    logger.debug(f"Segment IDs to save: {segment_IDs_to_save}")

            #Create the label map volume node
            logger.debug("Creating the label map volume node.")
            label_map_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{self.segmentationNode.GetName()}_label_map_node")
        
            # Export the segments to the label map volume node
            logger.debug("Exporting the segments to the label map volume node.")
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(self.segmentationNode, segment_IDs_to_save, label_map_volume_node, ReferenceVolumeNode)

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


    def save_all_segmentations(self, ReferenceVolumeNode: slicer.vtkMRMLVolumeNode, SaveDirectory: str, FileType: str = 'nii', ExtraSaveInfo: str = None):
        """
        This function saves all segmentations in the Segmentation Node to files. This function uses the save_segmentations_by_name method.

        Parameters
        ----------
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
        segment_names =list(self.get_segment_names())
        return self.save_segmentations_by_name(segment_names, ReferenceVolumeNode, SaveDirectory, FileType, ExtraSaveInfo)

