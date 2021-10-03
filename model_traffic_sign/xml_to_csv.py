# Convert XML files to CSV (for traffic sign model using Colab) 
# $ python xml_to_csv.py -d data
# Date: Sep 1, 2021
# Jeongkyu Lee

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import random
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default = 'data',
            help="path for the training file directory, e.g., -d data")
args = vars(ap.parse_args())
dir_name = args["dir"]

def xml_to_csv(path):
    xml_list = []
    record_list = ['TEST'] * 1 + ['TRAIN'] * 4 + ['VALIDATION'] * 1
    
    xml_files =  glob.glob(path + '/*.xml')

    for xml_file in glob.glob(path + '/*.xml'):
        record_type = random.choice(record_list)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (record_type,
                     '/content/'+dir_name+'/'+root.find('filename').text,
                     member[0].text,
                     int(member[4][0].text)/int(root.find('size')[0].text),
                     int(member[4][1].text)/int(root.find('size')[1].text),
                     None,
                     None,
                     int(member[4][2].text)/int(root.find('size')[0].text),
                     int(member[4][3].text)/int(root.find('size')[1].text),
                     None,
                     None
                     )
            xml_list.append(value)
    column_name = ['type','filename', 'class', 'xmin', 'ymin', 'xr', 'yr', 'xmax', 'ymax', 'xl', 'yl']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), dir_name)
    print(image_path)
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(dir_name+'/traffic_labels.csv', index=None)
    print('Successfully converted xml to csv.')


main()
