from typing import Any
import pickle
import cv2
import numpy as np
import utils
from utils.bbox_utils import measure_distance, measure_xy_distance
import os
import gc

class CameraMovementEstimator():
    def __init__(self, frame):
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:900, 1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.min_movement = 5

    def get_camera_movement(self, frames, read_from_stub=True, stub_path=None, batch_size=100):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print("Error loading the stub file. The file may be corrupted or empty.")
                # Proceed to calculate camera movement if stub loading fails

        num_frames = len(frames)
        camera_movement = [[0, 0] for _ in range(num_frames)]

        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch_frames = frames[start:end]

            old_gray = cv2.cvtColor(batch_frames[0], cv2.COLOR_RGB2GRAY)
            old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

            if old_features is not None:
                old_features = np.float32(old_features)

            for frame_num in range(1, len(batch_frames)):
                frame_gray = cv2.cvtColor(batch_frames[frame_num], cv2.COLOR_BGR2GRAY)
                if old_features is not None:
                    new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

                    max_distance = 0
                    camera_movement_x, camera_movement_y = 0, 0

                    for i, (new, old) in enumerate(zip(new_features, old_features)):
                        new_features_point = new.ravel()
                        old_features_point = old.ravel()

                        distance = measure_distance(new_features_point, old_features_point)
                        if distance > max_distance:
                            max_distance = distance
                            camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
                    if max_distance > self.min_movement:
                        camera_movement[start + frame_num] = [camera_movement_x, camera_movement_y]
                        old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                        if old_features is not None:
                            old_features = np.float32(old_features)
                    old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        return camera_movement
    



    def draw_camera_movement(self, frames, camera_movement_per_frame, batch_size=100, fps=60):
        output_video_path = 'output_video_with_movement.mp4'
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            for frame_num in range(batch_start, batch_end):
                frame = batch_frames[frame_num - batch_start].copy()
                overlay = frame.copy()

                cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)

                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                x_movement, y_movement = camera_movement_per_frame[frame_num]
                frame = cv2.putText(frame, f"Camera Movement X: {x_movement: .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                frame = cv2.putText(frame, f"Camera Movement Y: {y_movement: .2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                out.write(frame)
            
            # Release memory for the current batch
            del batch_frames
            cv2.destroyAllWindows()
            gc.collect()

            # Debugging: Print progress
            print(f"Processed batch: {batch_start} to {batch_end} of {len(frames)}")

        out.release()
        return output_video_path


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # def draw_camera_movement(self, frames, camera_movement_per_frame, batch_size=100):
    #     output_video_path = 'output_video_with_movement.mp4'
    #     height, width, layers = frames[0].shape
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

    #     for batch_start in range(0, len(frames), batch_size):
    #         batch_end = min(batch_start + batch_size, len(frames))
    #         batch_frames = frames[batch_start:batch_end]

    #         for frame_num in range(batch_start, batch_end):
    #             frame = batch_frames[frame_num - batch_start].copy()
    #             overlay = frame.copy()

    #             cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)

    #             alpha = 0.6
    #             cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    #             x_movement, y_movement = camera_movement_per_frame[frame_num]
    #             frame = cv2.putText(frame, f"Camera Movement X: {x_movement: .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    #             frame = cv2.putText(frame, f"Camera Movement Y: {y_movement: .2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    #             out.write(frame)
                
    #         # Release memory for the current batch
    #         del batch_frames
    #         cv2.destroyAllWindows()
    #         gc.collect()

    #         # Debugging: Print progress
    #         print(f"Processed batch: {batch_start} to {batch_end} of {len(frames)}")

    #     out.release()
    #     return output_video_path















#############


######CAMERA MOVEMENT ESTIMATOR



# from typing import Any
# import pickle
# import cv2
# import numpy as np
# import utils
# from utils.bbox_utils import measure_distance, measure_xy_distance
# import os

# class CameraMovementEstimator():
#     def __init__(self, frame):
#         first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         mask_features = np.zeros_like(first_frame_grayscale)
#         mask_features[:, 0:20] = 1
#         mask_features[:900, 1050] = 1

#         self.features = dict(
#             maxCorners=100,
#             qualityLevel=0.3,
#             minDistance=3,
#             blockSize=7,
#             mask=mask_features
#         )
#         self.lk_params = dict(
#             winSize=(15, 15),
#             maxLevel=2,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#         )

#         self.min_movement = 5

#     def get_camera_movement(self, frames, read_from_stub=True, stub_path=None):

#         if read_from_stub and stub_path is not None and os.path.exists(stub_path):
#             try:
#                 with open(stub_path, 'rb') as f:
#                     return pickle.load(f)
#             except (EOFError, pickle.UnpicklingError):
#                 print("Error loading the stub file. The file may be corrupted or empty.")
#                 # Proceed to calculate camera movement if stub loading fails

#         camera_movement = [[0, 0] for _ in range(len(frames))]
#         old_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
#         old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

#         if old_features is not None:
#             old_features = np.float32(old_features)

#         for frame_num in range(1, len(frames)):
#             frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
#             if old_features is not None:
#                 new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

#                 max_distance = 0
#                 camera_movement_x, camera_movement_y = 0, 0

#                 for i, (new, old) in enumerate(zip(new_features, old_features)):
#                     new_features_point = new.ravel()
#                     old_features_point = old.ravel()

#                     distance = measure_distance(new_features_point, old_features_point)
#                     if distance > max_distance:
#                         max_distance = distance
#                         camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
#                 if max_distance > self.min_movement:
#                     camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
#                     old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
#                     if old_features is not None:
#                         old_features = np.float32(old_features)
#                 old_gray = frame_gray.copy()

#         if stub_path is not None:
#             with open(stub_path, 'wb') as f:
#                 pickle.dump(camera_movement, f)
#         return camera_movement

#     def draw_camera_movement(self, frames, camera_movement_per_frame):
#         output_frames = []
#         for frame_num, frame in enumerate(frames):
#             frame = frame.copy()
#             overlay = frame.copy()

#             cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)

#             alpha = 0.6
#             cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

#             x_movement, y_movement = camera_movement_per_frame[frame_num]
#             frame = cv2.putText(frame, f"Camera Movement X: {x_movement: .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
#             frame = cv2.putText(frame, f"Camera Movement Y: {y_movement: .2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

#             output_frames.append(frame)
#         return output_frames
    
