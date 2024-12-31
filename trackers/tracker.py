

    
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import utils
# sys.path.append('../')
from utils import bbox_utils
import pandas as pd

from utils.bbox_utils import get_bbox_width, get_center_of_bbox

class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns= ['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        num_frames = len(frames)
        print(f"Total number of frames: {num_frames}")

        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            batch = frames[i: batch_end]
            print(f"Processing batch from frame {i} to frame {batch_end - 1} with batch size {len(batch)}")
            detections_batch = self.model.predict(batch, conf=0.1)
            detections.extend(detections_batch)

            if not detections_batch:
                print(f"No detections in batch from frame {i} to frame {batch_end - 1}")
            else:
                print(f"Detections in batch from frame {i} to frame {batch_end - 1}: {detections_batch}")

        return detections
    
    def get_object_tracks(self, frames, read_from_stub=True, stub_path=None):

        print(f"get_object_tracks called with {len(frames)} frames.")
        
        # Load tracks from a stub file if specified and exists
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)  # Deserialize the tracks from the file
            print("Tracks loaded from stub.")
            return tracks

        # Detection and tracking process
        detections = self.detect_frames(frames)
        print(f"Detected {len(detections)} batches of detections.")
        
        tracks = {
            "Player": [],
            "Ref": [],
            "Ball": [],
            "Hoop": []
        }

        for frame_num, detection in enumerate(detections):
            if detection is None or len(detection) == 0:
                print(f"No detections in frame {frame_num}")
                tracks["Ball"].append({})
                tracks["Hoop"].append({})
                tracks["Player"].append({})
                tracks["Ref"].append({})
                continue

            cls_names = detection.names
            cls_names_inver = {v: k for k, v in cls_names.items()}
            
            # Convert to supervision detection
            detection_supervision = sv.Detections.from_ultralytics(detection)
            if detection_supervision is None or len(detection_supervision) == 0:
                print(f"No valid detections in frame {frame_num}")
                tracks["Ball"].append({})
                tracks["Hoop"].append({})
                tracks["Player"].append({})
                tracks["Ref"].append({})
                continue

            print(f"Frame {frame_num} detection supervision: {detection_supervision}")

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            if detection_with_tracks is None or len(detection_with_tracks) == 0:
                print(f"No tracks found in frame {frame_num}")
                tracks["Ball"].append({})
                tracks["Hoop"].append({})
                tracks["Player"].append({})
                tracks["Ref"].append({})
                continue

            print(f"Frame {frame_num} detection with tracks: {detection_with_tracks}")

            # Initialize empty dictionaries for each object type in the current frame
            tracks["Ball"].append({})
            tracks["Hoop"].append({})
            tracks["Player"].append({})
            tracks["Ref"].append({})

            for frame_detection in detection_with_tracks:
                if len(frame_detection) < 5:
                    print(f"Unexpected frame detection format in frame {frame_num}: {frame_detection}")
                    continue
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if frame_detection[4] is not None and len(frame_detection[4]) > 0:
                    track_id = frame_detection[4]
                else:
                    print(f"No valid tracker_id for frame {frame_num}, detection: {frame_detection}")
                    continue
                
                if cls_id == cls_names_inver['Player']:
                    tracks["Player"][frame_num][track_id] = {"bbox": bbox}
                    print(f"Added Player track for frame {frame_num}, track ID {track_id}: {bbox}")
                elif cls_id == cls_names_inver['Ref']:
                    tracks["Ref"][frame_num][track_id] = {"bbox": bbox}
                    print(f"Added Ref track for frame {frame_num}, track ID {track_id}: {bbox}")

            for frame_detection in detection_supervision:
                if len(frame_detection) < 4:
                    print(f"Unexpected detection supervision format in frame {frame_num}: {frame_detection}")
                    continue
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                # Ensure track_id is initialized
                if frame_detection[4] is not None and len(frame_detection[4]) > 0:
                    track_id = frame_detection[4]
                else:
                    track_id = -1

                if cls_id == cls_names_inver['Ball']:
                    tracks["Ball"][frame_num][1] = {"bbox": bbox}
                elif cls_id == cls_names_inver['Hoop']:
                    tracks["Hoop"][frame_num][1] = {"bbox": bbox}

        # Save tracks to a stub file if specified
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)  # Serialize the tracks to the file
            print("Tracks saved to stub.")

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center,_ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center = (x_center, y2), axes= (int(width),int(.35*width)), angle = 0.0, startAngle= -45, endAngle=235, color = color, thickness= 4, lineType=cv2.LINE_4)



        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center - rectangle_width//2
        y1_rect = (y2-rectangle_height//2) + 15
        y2_rect = (y2+rectangle_height//2) + 15

        x1_text = x1_rect + 10

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)) , color,cv2.FILLED )
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect+15)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0),2)
        return frame
    

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_pts = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])

        cv2.drawContours(frame, [triangle_pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_pts], 0 , (0,0,0), 2)
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        cop = frame.copy()
        cv2.rectangle(cop, (0,0), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(cop, alpha, frame, 1-alpha, 0)


        team_ball_control_frame = team_ball_control[:frame_num+1]

        team_1_num_frames = team_ball_control_frame[team_ball_control_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_frame[team_ball_control_frame == 2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control:{team_1*100: .2f}% ", (1400,900), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3) 
        cv2.putText(frame, f"Team 2 Ball Control:{team_2*100: .2f}% ", (1400,950), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3) 

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["Player"][frame_num]
            ref_dict = tracks["Ref"][frame_num]
            ball_dict = tracks["Ball"][frame_num]
            hoop_dict = tracks["Hoop"][frame_num]

            for track_id, player in player_dict.items():
                if 'team_color' in player:
                    team_color = player['team_color']
                else:
                    team_color = (0, 0, 255) 
                
                frame = self.draw_ellipse(frame, player['bbox'],team_color, track_id)
                
                if 'has_ball' in player:
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))


            for track_id, ref in ref_dict.items():
                frame = self.draw_ellipse(frame, ref['bbox'],(0,255,255), track_id)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
            
            for track_id, hoop in hoop_dict.items():
                frame = self.draw_triangle(frame, hoop["bbox"], (0,0,0))
            


            #Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)           
        
        return output_video_frames
    























