



# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# class TeamAssigner:
#     def __init__(self, delta_h=10, delta_s=50, delta_v=50, n_init=10):
#         self.delta_h = delta_h
#         self.delta_s = delta_s
#         self.delta_v = delta_v
#         self.n_init = n_init
#         self.player_colors = []
#         self.team_colors = {}
#         self.player_team_dict = {}

#     def img_to_hsv(self, frame, bbox):
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#         top_half_image = image[0:int(image.shape[0] / 2), :]
#         image_hsv = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)
#         return image_hsv

#     def exclude_brown(self, image_hsv):
#         lower_brown = np.array([10, 20, 100], dtype=np.uint8)
#         upper_brown = np.array([30, 255, 200], dtype=np.uint8)
#         mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
#         image_hsv[mask_brown > 0] = 0  # Set brown pixels to black (or any color not present in the image)
#         return image_hsv

#     def find_dominant_color(self, image_hsv):
#         image_hsv = self.exclude_brown(image_hsv)
#         pixels = image_hsv.reshape(-1, 3)
#         pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black pixels (used for excluding brown)
#         kmeans = KMeans(n_clusters=2, n_init=self.n_init)
#         kmeans.fit(pixels)
#         dominant_colors = kmeans.cluster_centers_.astype(int)

#         mask1 = cv2.inRange(image_hsv, self.create_hsv_range(dominant_colors[0])[0], self.create_hsv_range(dominant_colors[0])[1])
#         mask2 = cv2.inRange(image_hsv, self.create_hsv_range(dominant_colors[1])[0], self.create_hsv_range(dominant_colors[1])[1])

#         if np.count_nonzero(mask1) > np.count_nonzero(mask2):
#             return dominant_colors[0]
#         else:
#             return dominant_colors[1]

#     def create_hsv_range(self, center_color):
#         lower_bound = np.array([
#             max(center_color[0] - self.delta_h, 0),
#             max(center_color[1] - self.delta_s, 0),
#             max(center_color[2] - self.delta_v, 0)
#         ], dtype=np.uint8)
#         upper_bound = np.array([
#             min(center_color[0] + self.delta_h, 179),
#             min(center_color[1] + self.delta_s, 255),
#             min(center_color[2] + self.delta_v, 255)
#         ], dtype=np.uint8)
#         return lower_bound, upper_bound

#     def assign_team_color(self, frame, player_detections,):
#         for _, player_detection in player_detections.items():
#             bbox = player_detection["bbox"]
#             image_hsv = self.img_to_hsv(frame, bbox)
#             dominant_color = self.find_dominant_color(image_hsv)
#             self.player_colors.append(dominant_color)

            
            
            
            
#             # if player_detection[player_id] not in self.player_colors:
#             #     self.player_colors[player_detection[player_id]] = []
#             # self.player_colors[player_detection[player_id]].append(dominant_color)

#     def finalize_team_colors(self):
#         all_colors = []
#         for colors in self.player_colors:
#             all_colors.append(colors)

#         kmeans = KMeans(n_clusters=2, n_init=self.n_init)
#         kmeans.fit(all_colors)
#         self.team_colors[1] = kmeans.cluster_centers_[0]
#         self.team_colors[2] = kmeans.cluster_centers_[1]

#         print(f"Final Team Colors: {self.team_colors}")

#     def get_player_team(self, frame, player_bbox, player_id):
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]
        
#         new_img = self.img_to_hsv(frame, player_bbox)
#         new_img_no_brown = self.exclude_brown(new_img)
#         player_dom_color =  self.find_dominant_color(new_img_no_brown)

#         player_color = np.mean(player_dom_color, axis=0)
#         distance_to_team1 = np.linalg.norm(player_color - self.team_colors[1])
#         distance_to_team2 = np.linalg.norm(player_color - self.team_colors[2])

#         if distance_to_team1 < distance_to_team2:
#             team = 1
#         else:
#             team = 2

#         self.player_team_dict[player_id] = team
#         return team
        

    





############

# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# class TeamAssigner:
#     def __init__(self, delta_h=10, delta_s=50, delta_v=50, n_init=10):
#         self.delta_h = delta_h
#         self.delta_s = delta_s
#         self.delta_v = delta_v
#         self.n_init = n_init
#         self.player_colors = []
#         self.team_colors = {}
#         self.player_team_dict = {}

#     def img_to_hsv(self, frame, bbox):
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#         top_half_image = image[int(image.shape[0] / 2):, :]

#         image_hsv = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)
#         return image_hsv

#     def exclude_brown(self, image_hsv):
#         lower_brown = np.array([10, 20, 100], dtype=np.uint8)
#         upper_brown = np.array([30, 255, 200], dtype=np.uint8)
#         mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
#         image_hsv[mask_brown > 0] = 0  # Set brown pixels to black (or any color not present in the image)
#         return image_hsv

#     def find_dominant_color(self, image_hsv):
#         image_hsv = self.exclude_brown(image_hsv)
#         pixels = image_hsv.reshape(-1, 3)
#         pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black pixels (used for excluding brown)
#         kmeans = KMeans(n_clusters=2, n_init=self.n_init)
#         kmeans.fit(pixels)
#         dominant_colors = kmeans.cluster_centers_.astype(int)

#         mask1 = cv2.inRange(image_hsv, self.create_hsv_range(dominant_colors[0])[0], self.create_hsv_range(dominant_colors[0])[1])
#         mask2 = cv2.inRange(image_hsv, self.create_hsv_range(dominant_colors[1])[0], self.create_hsv_range(dominant_colors[1])[1])

#         if np.count_nonzero(mask1) > np.count_nonzero(mask2):
#             return dominant_colors[0]
#         else:
#             return dominant_colors[1]

#     def create_hsv_range(self, center_color):
#         lower_bound = np.array([
#             max(center_color[0] - self.delta_h, 0),
#             max(center_color[1] - self.delta_s, 0),
#             max(center_color[2] - self.delta_v, 0)
#         ], dtype=np.uint8)
#         upper_bound = np.array([
#             min(center_color[0] + self.delta_h, 179),
#             min(center_color[1] + self.delta_s, 255),
#             min(center_color[2] + self.delta_v, 255)
#         ], dtype=np.uint8)
#         return lower_bound, upper_bound

#     def assign_team_color(self, frame, player_detections):
#         for _, player_detection in player_detections.items():
#             bbox = player_detection["bbox"]
#             image_hsv = self.img_to_hsv(frame, bbox)
#             dominant_color = self.find_dominant_color(image_hsv)
#             self.player_colors.append(dominant_color)

#     def finalize_team_colors(self):
#         all_colors = []
#         for colors in self.player_colors:
#             all_colors.append(colors)

#         kmeans = KMeans(n_clusters=2, n_init=self.n_init)
#         kmeans.fit(all_colors)
#         team_colors_hsv = kmeans.cluster_centers_.astype(int)

#         # Convert team colors from HSV to RGB
#         hsv_reshaped = np.uint8(team_colors_hsv).reshape(1, 2, 3)
#         team_colors_rgb = cv2.cvtColor(hsv_reshaped, cv2.COLOR_HSV2RGB).reshape(2, 3)

#         self.team_colors[1] = team_colors_rgb[0].tolist()
#         self.team_colors[2] = team_colors_rgb[1].tolist()

#         print(f"Final Team Colors (HSV): {team_colors_hsv}")
#         print(f"Final Team Colors (RGB): {team_colors_rgb}")

#     def get_player_team(self, frame, player_bbox, player_id):
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]
        
#         new_img = self.img_to_hsv(frame, player_bbox)
#         new_img_no_brown = self.exclude_brown(new_img)
#         player_dom_color = self.find_dominant_color(new_img_no_brown)

#         player_color = np.mean(player_dom_color, axis=0)
#         distance_to_team1 = np.linalg.norm(player_color - self.team_colors[1])
#         distance_to_team2 = np.linalg.norm(player_color - self.team_colors[2])

#         if distance_to_team1 < distance_to_team2:
#             team = 1
#         else:
#             team = 2

#         self.player_team_dict[player_id] = team
#         return team










# #####################   3


# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# class TeamAssigner:
#     def __init__(self, delta_h=10, delta_s=50, delta_v=50, n_init=10):
#         self.delta_h = delta_h
#         self.delta_s = delta_s
#         self.delta_v = delta_v
#         self.n_init = n_init
#         self.player_colors = []
#         self.team_colors = {}
#         self.player_team_dict = {}

#     def img_to_hsv(self, frame, bbox):
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#         top_half_image = image[0:int(image.shape[0] / 2), :]
#         image_hsv = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)
#         return image_hsv

#     def exclude_colors(self, image_hsv, colors_to_exclude):
#         for lower, upper in colors_to_exclude:
#             mask = cv2.inRange(image_hsv, lower, upper)
#             image_hsv[mask > 0] = 0  # Set excluded color pixels to black
#         return image_hsv

#     def find_dominant_color(self, image_hsv):
#         colors_to_exclude = [
#             (np.array([10, 20, 100], dtype=np.uint8), np.array([30, 255, 200], dtype=np.uint8)),  # Brown
#             # Add more colors to exclude if needed
#         ]
#         image_hsv = self.exclude_colors(image_hsv, colors_to_exclude)
#         pixels = image_hsv.reshape(-1, 3)
#         pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black pixels (used for excluding unwanted colors)
#         kmeans = KMeans(n_clusters=3, n_init=self.n_init)  # Increase number of clusters
#         kmeans.fit(pixels)
#         dominant_colors = kmeans.cluster_centers_.astype(int)

#         # Use the two most prominent colors for team classification
#         counts = np.bincount(kmeans.labels_)
#         idx1, idx2 = np.argsort(counts)[-2:]

#         return dominant_colors[idx1], dominant_colors[idx2]

#     def create_hsv_range(self, center_color):
#         lower_bound = np.array([
#             max(center_color[0] - self.delta_h, 0),
#             max(center_color[1] - self.delta_s, 0),
#             max(center_color[2] - self.delta_v, 0)
#         ], dtype=np.uint8)
#         upper_bound = np.array([
#             min(center_color[0] + self.delta_h, 179),
#             min(center_color[1] + self.delta_s, 255),
#             min(center_color[2] + self.delta_v, 255)
#         ], dtype=np.uint8)
#         return lower_bound, upper_bound

#     def assign_team_color(self, frame, player_detections):
#         for _, player_detection in player_detections.items():
#             bbox = player_detection["bbox"]
#             image_hsv = self.img_to_hsv(frame, bbox)
#             dominant_color1, dominant_color2 = self.find_dominant_color(image_hsv)
#             self.player_colors.append(dominant_color1)
#             self.player_colors.append(dominant_color2)

#     def finalize_team_colors(self):
#         kmeans = KMeans(n_clusters=2, n_init=self.n_init)
#         kmeans.fit(self.player_colors)
#         team_colors_hsv = kmeans.cluster_centers_.astype(int)

#         # Convert team colors from HSV to RGB
#         hsv_reshaped = np.uint8(team_colors_hsv).reshape(1, 2, 3)
#         team_colors_rgb = cv2.cvtColor(hsv_reshaped, cv2.COLOR_HSV2RGB).reshape(2, 3)

#         self.team_colors[1] = team_colors_rgb[0].tolist()
#         self.team_colors[2] = team_colors_rgb[1].tolist()

#         print(f"Final Team Colors (HSV): {team_colors_hsv}")
#         print(f"Final Team Colors (RGB): {team_colors_rgb}")

#     def get_player_team(self, frame, player_bbox, player_id):
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]
        
#         new_img = self.img_to_hsv(frame, player_bbox)
#         new_img_no_brown = self.exclude_colors(new_img, [(np.array([10, 20, 100], dtype=np.uint8), np.array([30, 255, 200], dtype=np.uint8))])
#         player_dom_color1, player_dom_color2 = self.find_dominant_color(new_img_no_brown)

#         player_color1 = np.mean(player_dom_color1, axis=0)
#         player_color2 = np.mean(player_dom_color2, axis=0)
#         distance_to_team1 = min(np.linalg.norm(player_color1 - self.team_colors[1]), np.linalg.norm(player_color2 - self.team_colors[1]))
#         distance_to_team2 = min(np.linalg.norm(player_color1 - self.team_colors[2]), np.linalg.norm(player_color2 - self.team_colors[2]))

#         if distance_to_team1 < distance_to_team2:
#             team = 1
#         else:
#             team = 2

#         self.player_team_dict[player_id] = team
#         return team





# import cv2
# import numpy as np
# from sklearn.cluster import KMeans

# class TeamAssigner:
#     def __init__(self, delta_h=20, delta_s=70, delta_v=70, n_init=10):
#         self.delta_h = delta_h
#         self.delta_s = delta_s
#         self.delta_v = delta_v
#         self.n_init = n_init
#         self.player_colors = []
#         self.team_colors = {}
#         self.player_team_dict = {}

#     def img_to_hsv(self, frame, bbox):
#         # Extract the bounding box coordinates
#         x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
#         bbox_height = y2 - y1

#         # Define the region from just above the shoes (10% from bottom) to the bottom half
#         shoe_level = y2 - int(0.2 * bbox_height)
#         bottom_half_start = y1 + int(bbox_height / 2)
#         region_y1 = min(shoe_level, bottom_half_start)  # Ensure it doesn't go above the bottom half

#         # Extract the region of interest
#         region_of_interest = frame[region_y1:y2, x1:x2]

#         # Convert the extracted region to HSV
#         image_hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)
#         return image_hsv

#     def exclude_colors(self, image_hsv, colors_to_exclude):
#         for lower, upper in colors_to_exclude:
#             mask = cv2.inRange(image_hsv, lower, upper)
#             image_hsv[mask > 0] = 0  # Set excluded color pixels to black
#         return image_hsv

#     def find_dominant_color(self, image_hsv):
#         colors_to_exclude = [
#             (np.array([10, 20, 100], dtype=np.uint8), np.array([30, 255, 200], dtype=np.uint8)),  # Brown
#             (np.array([35, 50, 50], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))    # Green
#         ]
#         image_hsv = self.exclude_colors(image_hsv, colors_to_exclude)
#         pixels = image_hsv.reshape(-1, 3)
#         pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black pixels (used for excluding unwanted colors)
        
#         # Perform KMeans with more clusters
#         kmeans = KMeans(n_clusters=5, n_init=self.n_init)
#         kmeans.fit(pixels)
#         dominant_colors = kmeans.cluster_centers_.astype(int)

#         # Select the two most prominent colors
#         unique, counts = np.unique(kmeans.labels_, return_counts=True)
#         sorted_indices = np.argsort(counts)[-2:]  # Get the indices of the two largest clusters

#         return dominant_colors[sorted_indices[0]], dominant_colors[sorted_indices[1]]

#     def create_hsv_range(self, center_color):
#         lower_bound = np.array([
#             max(center_color[0] - self.delta_h, 0),
#             max(center_color[1] - self.delta_s, 0),
#             max(center_color[2] - self.delta_v, 0)
#         ], dtype=np.uint8)
#         upper_bound = np.array([
#             min(center_color[0] + self.delta_h, 179),
#             min(center_color[1] + self.delta_s, 255),
#             min(center_color[2] + self.delta_v, 255)
#         ], dtype=np.uint8)
#         return lower_bound, upper_bound

#     def set_team_colors(self, team_color1_rgb, team_color2_rgb):
#         team_color1_hsv = cv2.cvtColor(np.uint8([[team_color1_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
#         team_color2_hsv = cv2.cvtColor(np.uint8([[team_color2_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

#         self.team_colors[1] = team_color1_hsv.tolist()
#         self.team_colors[2] = team_color2_hsv.tolist()

#         print(f"Set Team Colors (HSV): {self.team_colors}")

#     def assign_team_color(self, frame, player_detections):
#         for _, player_detection in player_detections.items():
#             bbox = player_detection["bbox"]
#             image_hsv = self.img_to_hsv(frame, bbox)
#             dominant_color1, dominant_color2 = self.find_dominant_color(image_hsv)
#             self.player_colors.append(dominant_color1)
#             self.player_colors.append(dominant_color2)

#     def finalize_team_colors(self):
#         kmeans = KMeans(n_clusters=2, n_init=self.n_init)
#         kmeans.fit(self.player_colors)
#         team_colors_hsv = kmeans.cluster_centers_.astype(int)

#         # Convert team colors from HSV to RGB
#         hsv_reshaped = np.uint8(team_colors_hsv).reshape(1, 2, 3)
#         team_colors_rgb = cv2.cvtColor(hsv_reshaped, cv2.COLOR_HSV2RGB).reshape(2, 3)

#         self.team_colors[1] = team_colors_rgb[0].tolist()
#         self.team_colors[2] = team_colors_rgb[1].tolist()

#         print(f"Final Team Colors (HSV): {team_colors_hsv}")
#         print(f"Final Team Colors (RGB): {team_colors_rgb}")

#     def get_player_team(self, frame, player_bbox, player_id):
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]

#         new_img = self.img_to_hsv(frame, player_bbox)
#         new_img_no_brown = self.exclude_colors(new_img, [
#             (np.array([10, 20, 100], dtype=np.uint8), np.array([30, 255, 200], dtype=np.uint8)),  # Exclude brown
#             (np.array([35, 50, 50], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))    # Exclude green
#         ])

#         # Count pixels matching each team color
#         lower_team1, upper_team1 = self.create_hsv_range(np.uint8(self.team_colors[1]))
#         lower_team2, upper_team2 = self.create_hsv_range(np.uint8(self.team_colors[2]))

#         mask_team1 = cv2.inRange(new_img_no_brown, lower_team1, upper_team1)
#         mask_team2 = cv2.inRange(new_img_no_brown, lower_team2, upper_team2)

#         count_team1 = np.count_nonzero(mask_team1)
#         count_team2 = np.count_nonzero(mask_team2)

#         # Debugging: Visualize masks
#         # cv2.imshow(f'Mask Team 1 - Player {player_id}', mask_team1)
#         # cv2.imshow(f'Mask Team 2 - Player {player_id}', mask_team2)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()

#         print(f"Player {player_id}: Team 1 pixel count = {count_team1}, Team 2 pixel count = {count_team2}")

#         if count_team1 > count_team2:
#             team = 1
#         else:
#             team = 2

#         self.player_team_dict[player_id] = team
#         return team









#############



import cv2
import numpy as np

class TeamAssigner:
    def __init__(self, delta_h=10, delta_s=50, delta_v=50):
        self.delta_h = delta_h
        self.delta_s = delta_s
        self.delta_v = delta_v
        self.team_colors = {}
        self.team_colors_rgb = {}

        self.player_team_dict = {}

    def img_to_hsv(self, frame, bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bbox_height = y2 - y1
        bbox_width = x2-x1

        # Define the region for the middle third of the image height
        third_height = bbox_height // 6
        quarter_width = bbox_width/4
        region_y1 = y1 + 2*third_height
        region_y2 = y1 + 3 * third_height
        region_x1 = x1+ 2*quarter_width -5
        region_x2 = x1+ 2* quarter_width + 5

        # Extract the region of interest
        region_of_interest = frame[region_y1:region_y2, int(region_x1):int(region_x2)]
        # image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


        # top_half_image = image[0:int(image.shape[0] / 2), :]

        # Convert the extracted region to HSV
        image_hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_RGB2HSV)
        return image_hsv


    def exclude_colors(self, image_hsv, colors_to_exclude):
        for lower, upper in colors_to_exclude:
            mask = cv2.inRange(image_hsv, lower, upper)
            image_hsv[mask > 0] = 0  # Set excluded color pixels to black
        return image_hsv

    def create_hsv_range(self, center_color):
        lower_bound = np.array([
            max(center_color[0] - self.delta_h, 0),
            max(center_color[1] - self.delta_s, 0),
            max(center_color[2] - self.delta_v, 0)
        ], dtype=np.uint8)
        upper_bound = np.array([
            min(center_color[0] + self.delta_h, 179),
            min(center_color[1] + self.delta_s, 255),
            min(center_color[2] + self.delta_v, 255)
        ], dtype=np.uint8)
        return lower_bound, upper_bound

    def set_team_colors(self, team_color1_rgb, team_color2_rgb):
        team_color1_hsv = cv2.cvtColor(np.uint8([[team_color1_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        team_color2_hsv = cv2.cvtColor(np.uint8([[team_color2_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        self.team_colors[1] = team_color1_hsv.tolist()
        self.team_colors[2] = team_color2_hsv.tolist()
        self.team_colors_rgb[1] = team_color1_rgb
        self.team_colors_rgb[2] = team_color2_rgb

        print(f"Set Team Colors (HSV): {self.team_colors}")

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        new_img = self.img_to_hsv(frame, player_bbox)
        new_img_no_brown = self.exclude_colors(new_img, [
            (np.array([10, 20, 100], dtype=np.uint8), np.array([30, 255, 200], dtype=np.uint8))  # Exclude brown
        ])

        # Count pixels matching each team color
        lower_team1, upper_team1 = self.create_hsv_range(np.uint8(self.team_colors[1]))
        lower_team2, upper_team2 = self.create_hsv_range(np.uint8(self.team_colors[2]))

        mask_team1 = cv2.inRange(new_img_no_brown, lower_team1, upper_team1)
        mask_team2 = cv2.inRange(new_img_no_brown, lower_team2, upper_team2)

        count_team1 = np.count_nonzero(mask_team1)
        count_team2 = np.count_nonzero(mask_team2)

        print(f"Player {player_id}: Team 1 pixel count = {count_team1}, Team 2 pixel count = {count_team2}")

        if count_team1 > count_team2:
            team = 1
        else:
            team = 2

        self.player_team_dict[player_id] = team
        return team
