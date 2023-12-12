import degirum as dg
import numpy as np
import supervision as sv

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
        return (frame[start_row:end_row, start_col:end_col, :],(start_row, start_col, end_row, end_col))
   
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
    

    def predict_batch(self, source_video_path):
        detection_list = []
        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        video_frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

        framebuf = np.zeros((video_info.height, video_info.width, 3), dtype=np.uint8)
        target_video_path = "slice_test2_traffic_cars.mp4"

        with sv.VideoSink(target_path = target_video_path, video_info=video_info) as sink:
            for detection in self._model.predict_batch(self.slice_video_source(video_frame_generator)):
                # print (detection,offset)
                offset = detection.info
                self.slice_recombine(framebuf, detection.image, offset)
                detection_list.append(self.sv_detections_from_degirum(detection))  # Add slice detections to list
                # print (detection_list)
                detection_list[-1].xyxy += (offset[1], offset[0], offset[1], offset[0])  # Offset slice detections to frame coords
                # print (detection,offset)
                if offset[3] == video_info.width:
                    detections = sv.Detections.merge(detections_list=detection_list).with_nms(threshold=0.5)
                    # print (detections)
                    detections = tracker.update_with_detections(detections)
                    labels = [f"#{tracker_id} {model.label_dictionary[class_id]}" for _, _, _, class_id, tracker_id in detections]
                    annotated_frame = box_annotator.annotate(scene=framebuf, detections=detections, labels=labels)
                    if detection_list:
                        sink.write_frame(frame=annotated_frame)
                        detection_list.clear()
    
if __name__ == "__main__":
    zoo = dg.connect(dg.CLOUD, "https://cs.degirum.com/degirum/public", "dg_VXc9Zfpkki4KgtjTvBdRY57QcCH6rbS3fqtw1")
    model_name = "yolo_v5s_coco--512x512_quant_n2x_orca_1"
    model = zoo.load_model(model_name)
    model.output_conf_threshold = 0.3
    model.output_nms_threshold = 0.7
    model.image_backend = "opencv"
    num_rows = 4
    num_cols = 4
    overlap_percent = 0.2
    source_video_path = "traffic_cars.mp4"
    # input_image_path = "small-vehicles1.jpeg"
    print ("--------------------model loaded---------------------")
    sliced_model = SlicedModel(model, num_rows, num_cols, overlap_percent, frame_shape = None)
    sliced_model.predict_batch(source_video_path)
    # sliced_model.predict(input_image_path)
