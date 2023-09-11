import os
import pandas as pd

class SunnyBrookDataFrames:
  def __init__(self, data_root):
    self.data_root = data_root
    self._init_patient_df()
    self._init_contour_df()
    self._init_image_df()
    self._init_merged_df()

  def _init_patient_df(self):
    def add_leading_zero(x):
      tks = x.split('-')
      return ''.join([tks[i]+'-' for i in range(len(tks)-1)]+[f'{int(tks[-1]):02d}'])

    patient_csv = os.path.join(self.data_root, "scd_patientdata.csv")
    self.patient_df = pd.read_csv(patient_csv, usecols=['PatientID', 'OriginalID'])
    self.patient_df['OriginalID_v0'] = self.patient_df['OriginalID']
    self.patient_df['OriginalID'] = self.patient_df['OriginalID_v0'].apply(add_leading_zero)


  def _init_contour_df(self):
    contour_root = os.path.join(self.data_root, "scd_manualcontours/SCD_ManualContours")
    contour_items = []
    for sd in os.listdir(contour_root):
        if 'SC-' not in sd:
            continue

        sd_path = os.path.join(contour_root, sd, "contours-manual/IRCCI-expert")
        for f in os.listdir(sd_path):
            if '-icontour-manual.txt' not in f:
                continue

            contour_items.append({'OriginalID': sd,
                                  'ImageID': int(f.split('-')[2]),
                                  'ContourFilePath': os.path.join(sd_path, f).replace("\\", "/")})

    self.contour_df = pd.DataFrame.from_dict(contour_items)


  def _init_image_df(self):
    def extract_cinesax_index(dir_name):
        return int(dir_name.replace('CINESAX_', ''))

    dcm_items = []
    for i in range(1,6):
        scd_images_dir_path = os.path.join(self.data_root, f"SCD_IMAGES_0{i}")
        for patient_dir in os.listdir(scd_images_dir_path):
            if 'SCD' not in patient_dir:
                continue

            patient_dir_path = os.path.join(scd_images_dir_path, patient_dir)

            # Handle multiple CINESAX dirs in the same patient dir
            # Take the CINESAX dir with the highest suffix int
            subdirs = [x for x in os.listdir(patient_dir_path) if 'CINESAX_' in x]
            subdirs.sort(key=extract_cinesax_index)
            subdir = subdirs[-1]

            subdir_path = os.path.join(patient_dir_path, subdir)
            for f in os.listdir(subdir_path):
                if '.dcm' not in f:
                    continue

                dcm_items.append({'PatientID': patient_dir,
                                  'ImageID': int(f.replace('.dcm', '').split('-')[2]),
                                  'ImageFilePath': os.path.join(subdir_path, f).replace("\\", "/")})

    self.image_df = pd.DataFrame.from_dict(dcm_items)

  def _init_merged_df(self):
    self.merged_df = self.contour_df.merge(self.patient_df, how='inner', on='OriginalID')
    self.merged_df = self.merged_df.merge(self.image_df, how='inner', on=['PatientID', 'ImageID'])

  def get_merged_df(self):
    return self.merged_df.copy()