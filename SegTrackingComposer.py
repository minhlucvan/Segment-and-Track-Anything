import json
import numpy as np

class SegTrackingComposer:
    def __init__(self, seg_tracker):
        """
        Initialize the SegTrackingComposer with a SegTracker instance.
        """
        self.seg_tracker = seg_tracker
        self.tracking_data = []

    def calculate_bounding_box(self, mask, obj_id):
        """
        Calculate the bounding box for a given object ID in a segmentation mask.
        Parameters:
            mask: numpy array, the segmentation mask.
            obj_id: int, the ID of the object to find the bounding box for.
        Returns:
            A tuple representing the bounding box in the format (x_min, y_min, x_max, y_max).
        """
        # Find the coordinates of pixels that belong to the object
        coords = np.where(mask == obj_id)
        if len(coords[0]) == 0:  # If the object is not found in the mask
            return None
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        return (x_min, y_min, x_max, y_max)

    def compile_tracking_data(self, frame_number=0):
        """
        Compile tracking data from the SegTracker instance.
        """
        # Assuming the latest segmentation mask is stored in self.seg_tracker.origin_merged_mask
        mask = self.seg_tracker.origin_merged_mask
        for obj_id in np.unique(mask):
            if obj_id == 0:  # Skip the background
                continue
            bounding_box = self.calculate_bounding_box(mask, obj_id)
            if bounding_box:
                obj_data = {
                    "id": obj_id,
                    "bounding_box": bounding_box,
                    "frame_number": frame_number
                }
                self.tracking_data.append(obj_data)

    def export_to_json(self, file_path="tracking_data.json"):
        """
        Export compiled tracking data to a JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(self.tracking_data, json_file, indent=4)
