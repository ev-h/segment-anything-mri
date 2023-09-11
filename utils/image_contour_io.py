import cv2
import pydicom
import numpy as np

class ImageContourPair:
  def __init__(self, image_file_path, contour_file_path):
      self.image_file_path = image_file_path
      self.contour_file_path = contour_file_path
      self._load_dicom_image()
      self._load_scd_contour_points()

  def _load_dicom_image(self):
      ds = pydicom.dcmread(self.image_file_path)
      self.image = ds.pixel_array

  def _load_scd_contour_points(self):
      # Load contour points
      self.contour_points = np.loadtxt(self.contour_file_path, delimiter=" ").astype(int)

  def get_mask(self):
      # Turn contour points to mask
      mask = np.zeros(self.get_image_shape(), dtype="uint8")
      cv2.fillPoly(mask, [self.contour_points], 1)
      return mask

  def get_image_shape(self):
      return self.image.shape

  def get_contour_center_point(self):
      max_x = np.max(self.contour_points[:, 0])
      max_y = np.max(self.contour_points[:, 1])
      min_x = np.min(self.contour_points[:, 0])
      min_y = np.min(self.contour_points[:, 1])
      return (int((max_x+min_x)/2), int((max_y+min_y)/2))

  def get_crop_coords(self):
      crop_center_point = self.get_contour_center_point()
      start_x = int(crop_center_point[1] - 75)
      start_y = int(crop_center_point[0] - 75)
      end_x = int(start_x + 150)
      end_y = int(start_y + 150)
      return start_x, end_x, start_y, end_y

  def get_cropped_image(self):
      start_x, end_x, start_y, end_y = self.get_crop_coords()
      return self.image[start_x:end_x, start_y:end_y]
  
  def get_cropped_mask(self):
      start_x, end_x, start_y, end_y = self.get_crop_coords()
      mask = self.get_mask()
      return mask[start_x:end_x, start_y:end_y]