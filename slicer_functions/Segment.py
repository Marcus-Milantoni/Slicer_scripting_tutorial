import numpy as np
import SegmentationNode


class Segment:
    def __init__(self, SegmentationNode: SegmentationNode, segmentObject, segmentName: str = None):
        self.segmentationNode = SegmentationNode
        self.segmentObject = segmentObject
        if segmentName == None:
            self.name = self.segmentObject.GetName()
        else:
            self.name = segmentName
            self.segmentObject.SetName(segmentName)
        self.segmentID = self.segmentObject.GetSegmentID()
        self.NumPyArray = None
        self.hasArray = False
        

    def description(self) -> str:
        return f"Segment {self.name} with ID {self.segmentID} in segmentation {self.segmentationNode.name}"
    

    def get_name(self) -> str:
        return self.name
    

    def get_id(self) -> str:
        return self.segmentID
    

    def edit_name(self, newName: str):
        self.segmentObject.SetName(newName)
        self.name = newName


    def delete(self):
        self.segmentationNode.remove_segment(self)


    def has_array(self) -> bool:
        return self.hasArray
    

    def get_array(self) -> np.ndarray:
        if self.hasArray:
            return self.NumPyArray
        else:
            return None


    def update_slicer(self):
        if self.hasArray:
            


    def set_array(self, array: np.ndarray, update):
        self.NumPyArray = array
        self.hasArray = True
    
