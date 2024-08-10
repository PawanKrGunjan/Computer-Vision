import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing import ImagePreprocessor

class LicensePlateExtractor:
    def __init__(self, img_thresh):
        self.img_thresh = img_thresh
        self.contours_dict = []
        self.possible_contours = []

    def filter_contours(self):
        MIN_AREA = 80
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0
        cnt = 0

        for d in self.contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']

            if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                self.possible_contours.append(d)

    def find_chars(self, contour_list):
        MAX_DIAG_MULTIPLYER = 5
        MAX_ANGLE_DIFF = 12.0
        MAX_AREA_DIFF = 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 3
        
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            # Remove duplicates from contour_list
            unmatched_contour_idx = [d['idx'] for d in contour_list if d['idx'] not in matched_contours_idx]
            unmatched_contour = [d for d in contour_list if d['idx'] in unmatched_contour_idx]
            
            # Recursive call
            recursive_contour_list = self.find_chars(unmatched_contour)
            
            for idx_list in recursive_contour_list:
                matched_result_idx.append(idx_list)

            break

        return matched_result_idx
    
    def possible_plates(self, matched_result):
        PLATE_WIDTH_PADDING = 1.3 # 1.3
        PLATE_HEIGHT_PADDING = 1.5 # 1.5
        MIN_PLATE_RATIO = 3
        MAX_PLATE_RATIO = 10
        w, h = self.img_thresh.shape
        plate_imgs = []
        plate_infos = []

        for i, matched_chars in enumerate(matched_result):
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
            
            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
            
            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
            
            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        
            img_rotated = cv2.warpAffine(self.img_thresh, M=rotation_matrix, dsize=(w, h))
            
            img_cropped = cv2.getRectSubPix(
                img_rotated, 
                patchSize=(int(plate_width), int(plate_height)), 
                center=(int(plate_cx), int(plate_cy))
            )
            
            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                continue
        
            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })
        return plate_imgs, plate_infos
        
    def extract_license_plate(self):
        contours, _ = cv2.findContours(self.img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        temp_result = np.zeros(self.img_thresh.shape, dtype=np.uint8)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
            self.contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        self.filter_contours()
        result_idx = self.find_chars(self.possible_contours)
        
        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(self.possible_contours, idx_list))
        plate_imgs, plate_infos = self.possible_plates(matched_result)

        return plate_imgs, plate_infos

"""
# Define paths
paths = os.path.join(os.getcwd(), 'ANPR_IMAGAS1')
img_path = os.path.join(paths, os.listdir(paths)[1])

# Preprocess Image
preprocessor = ImagePreprocessor(img_path)
#height, width, channel = preprocessor.image.shape
license_plate = LicensePlateExtractor(preprocessor.img_thresh)

plate_imgs, plate_infos= license_plate.extract_license_plate()

result_idx=license_plate.find_chars(license_plate.possible_contours)
n= len(plate_imgs)
for i in range(n):
    plt.subplot(n, 1, i+1)
    print(i)
    plt.imshow(plate_imgs[i], cmap='gray')
    plt.axis('off')
    plt.savefig('Car-Plates(Rotated).png',bbox_inches = 'tight')
    plt.show()
"""