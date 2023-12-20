import degirum as dg
import numpy as np
import supervision as sv
import cv2

class SlicedModel:
    def __init__(self, model , num_rows, num_cols, overlap_percent, frame_shape):
        self._model = model
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._overlap_percent = overlap_percent
        self._frame_shape = frame_shape
        self.slice_offsets = self.compute_slice_offsets()
           
    def compute_slice_offsets(self):
        if self._frame_shape is None:
            return None
                 
        patch_height = int((self._frame_shape[0])/(self._num_rows - self._overlap_percent*(self._num_rows-1)))
        patch_width = int((self._frame_shape[1])/(self._num_cols - self._overlap_percent*(self._num_cols-1)))
        print ("ibside compute slice offsets")
        return (patch_height, patch_width)
   
    def get_slice(self, frame, row, col):
        slice_height, slice_width = self.slice_offsets
        # Calculate patch start and end coordinates
        start_row = int(row * slice_height - row * self._overlap_percent * slice_height)
        end_row = int(start_row + slice_height)
        if row == (self._num_rows-1):
            end_row = self._frame_shape[0]

        start_col = int(col * slice_width - col * self._overlap_percent * slice_width)
        end_col = int(start_col + slice_width)
        if col == (self._num_cols-1):
            end_col = self._frame_shape[1]
        return (frame[start_row:end_row, start_col:end_col, :],(start_row,start_col))
   
    def slice_video_source(self, frame_generator):
        for frame in frame_generator:
            if self._frame_shape is None:
                self._frame_shape = frame.shape
                self.slice_offsets = self.compute_slice_offsets()

            for row in range(self._num_rows):
                for col in range(self._num_cols):
                    yield self.get_slice(frame, row, col)
                   
    def sv_detections_from_degirum(self, detections_dg):
        if not detections_dg.results:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([res["bbox"] for res in detections_dg.results], dtype=np.float32),
            confidence=np.array([res["score"] for res in detections_dg.results]),
            class_id=np.array([res["category_id"] for res in detections_dg.results])
        )


    def slice_recombine(self, framebuf, frame_slice, offset):
        framebuf[offset[0]:offset[0]+frame_slice.shape[0],offset[1]:offset[1]+frame_slice.shape[1],:] = frame_slice

    def slice_recombine_single_image(self, imagebuf, image_slice, offset):
        imagebuf[offset[0]:offset[0]+image_slice.shape[0],offset[1]:offset[1]+image_slice.shape[1],:] = image_slice
    


    def slice_image(self, image):
        if self._frame_shape is None:
            self._frame_shape = image.shape
            self.slice_offsets = self.compute_slice_offsets()

        for row in range(self._num_rows):
            for col in range(self._num_cols):
                print (row,col)
                yield self.get_slice(image, row, col)

    def predict(self,input_image):
        detection_list = []
        box_annotator = sv.BoxAnnotator()
        single_input_image = cv2.imread(input_image)
        image_height, image_width, _ = single_input_image.shape
        imagebuf = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for detection in self._model.predict_batch(self.slice_image(single_input_image)):
            offset = detection.info
            detection_list.append(self.sv_detections_from_degirum(detection))  # Add slice detections to list
            detection_list[-1].xyxy += (offset[1], offset[0], offset[1], offset[0])  # Offset slice detections to frame coords 
            
        detections = sv.Detections.merge(detections_list=detection_list).with_nms(threshold=0.5)
        single_input_image = box_annotator.annotate(scene=single_input_image, detections=detections)
        if detection_list:
            print ("detction list not empty")
            cv2.imwrite("output_image_traffic.jpg",single_input_image)
            detection_list.clear()

    
if __name__ == "__main__":
    zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com/degirum/public", "dg_VXc9Zfpkki4KgtjTvBdRY57QcCH6rbS3fqtw1")
    model_name = "yolo_v5s_coco--512x512_quant_n2x_orca_1"
    model = zoo.load_model(model_name)
    model.output_conf_threshold = 0.3
    model.output_nms_threshold = 0.7
    model.image_backend = "opencv"
    num_rows = 3
    num_cols = 2
    overlap_percent = 0.2
    # input_image_path = "small-vehicles1.jpeg"
    input_image_path = "traffic-jam-in-the-road.jpg"
    sliced_model = SlicedModel(model, num_rows, num_cols, overlap_percent, frame_shape = None)
    sliced_model.predict(input_image_path)
