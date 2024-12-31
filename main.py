# import utils
from utils import read_video, save_video
from utils.bbox_utils import get_bbox_width, get_center_of_bbox, measure_distance, measure_xy_distance
import trackers
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movemement_estimator import CameraMovementEstimator

def main():
    video_frames, fps = read_video('input_videos/bb_3.mp4')

    if not video_frames:
        print("Error: No frames read from video.")
        return

    print(f"Read {len(video_frames)} frames from video.")

    # Initialize Tracker
    tracker = Tracker('/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/model/best-2.pt')
    print("Tracker initialized.")

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/stubs/track_stubs.pkl')
    print("Tracking completed.")



    #camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/stubs/movement_stubs.pkl')

    print("Finished retrieving camera movement.")



    #draw cam movement

    tracks['Ball'] = tracker.interpolate_ball_positions(tracks['Ball'])

    team_assigner = TeamAssigner()

    # Set team colors directly (example colors in RGB)
    team_assigner.set_team_colors([210, 210, 210], [170, 93, 0])  # White and Blue

    # Process the frames
    for frame_num, player_track in enumerate(tracks['Player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['Player'][frame_num][player_id]['team'] = team 
            tracks['Player'][frame_num][player_id]['team_color'] = team_assigner.team_colors_rgb[team]




    player_assigner = PlayerBallAssigner()

    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['Player']):
        ball_bbox = tracks['Ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_player(player_track, ball_bbox)
        

        if assigned_player != -1:
            tracks['Player'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['Player'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)


    # Draw annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    print("Initial annotations drawn")

    #output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame, batch_size=50)
    save_video(output_video_frames, '/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/output_videos/output_video.mp4', fps)




    final_output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame, batch_size=100, fps=fps)

    # Save the final video with camera movement annotations
    final_output_video_path = '/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/output_videos/final_output_video.mp4'
    #save_video(final_output_video_frames, final_output_video_path, fps)

    print(f"Final video saved to: {final_output_video_path}")
    # print("Video saved.")





if __name__ == "__main__":
    main()







# def main():
#     video_frames, fps = read_video('input_videos/bb_2.mp4')


#     #Initialize Tracker
#     tracker = Tracker('/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/model/best-2.pt')
#     tracks = tracker.get_object_tracks(video_frames, read_from_stub= True, stub_path= '/Users/sanjayravindran/Documents/SoccerCV Project/basketballcv/stubs/track_stubs.pkl')

#     if not video_frames:
#         print("Error: No frames read from video.")
#         return
#     save_video(video_frames, 'output_videos/output_video.mp4', fps)


# if __name__ == "__main__":
#     main()




    # team_assigner.assign_team_color(video_frames[0], 
    #                             tracks['Player'][0])
    # team_assigner.finalize_team_colors()




    # for frame_num, player_track in enumerate(tracks['Player']):
    #     for player_id, track in player_track.items():
    #         team = team_assigner.get_player_team(video_frames[frame_num],   
    #                                                 track['bbox'],
    #                                                 player_id)
    #         tracks['Player'][frame_num][player_id]['team'] = team 
    #         tracks['Player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    # print(tracks['Player'][frame_num][player_id]['team'])


    # for frame_num, player_track in enumerate(tracks['Player']):
    #     for player_id, track in player_track.items():
    #         team_assigner.assign_team_color(video_frames[frame_num], track['bbox'])
    #         team = team_assigner.get_player_team(video_fram80es[frame_num],   
    #                                                 track['bbox'],
    #                                                 player_id)
    #         tracks['Player'][frame_num][player_id]['team'] = team 
    #         tracks['Player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # output_video_frames = tracker.draw_annotations(video_frames, tracks)




    # for frame_num, frame in enumerate(video_frames):

