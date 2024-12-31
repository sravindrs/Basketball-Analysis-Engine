import sys
# sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_dist = 60


    def assign_ball_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)
        min_dist = 9999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']


            x1, y1, x2, y2 = int(player_bbox[0]), int(player_bbox[1]), int(player_bbox[2]), int(player_bbox[3])
            bbox_height = y2 - y1

            # Define the region for the middle third of the image height
            quarter_height = bbox_height // 8
            region_y1 = y1 + 5 * quarter_height
            region_y2 = y1 + 5 * quarter_height


            # Extract the region of interest

            distance_left = measure_distance((x1,region_y1),ball_pos)
            distance_right = measure_distance((x2,region_y2),ball_pos)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_dist:
                if distance < min_dist:
                    min_dist = distance
                    assigned_player = player_id
        return assigned_player




