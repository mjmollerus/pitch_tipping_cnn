import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import cv2
import os

def download_videos(pitcher_name,pitch_labels):
    '''
    inputs: pitcher name, pitch types
    outputs: saves video files in corresponding folders
    '''
    
    os.mkdir(f'../data/{pitcher_name}/video')
    os.mkdir(f'../data/{pitcher_name}/image')
    
    url_lists = []
    
    url_list_files = os.listdir(f'../data/{pitcher_name}/vid_lists')
    for file_name in url_list_files:
        url_list = []
        with open(f'../data/{pitcher_name}/vid_lists/{file_name}','r') as file:
            for line in file:
                url_list.append(line[:-1])
        url_lists.append(url_list)
    
    
    for url_list,pitch_label in zip(url_lists,pitch_labels):
        
        path = f'../data/{pitcher_name}/video/{pitch_label}'
        os.mkdir(path)
        
        img_path = f'../data/{pitcher_name}/image/{pitch_label}'
        os.mkdir(img_path)
        
        for url in url_list:
            response = requests.get(f'https://baseballsavant.mlb.com/{url}')
            page = response.text
            soup = BeautifulSoup(page,'lmxl')

            video_url = soup.find('source').get('src')

            file_name = video_url.split('/')[-1]
            r = requests.get(video_url, stream=True)

            with open(f'../data/{pitcher_name}/video/{pitch_label}/{file_name}','wb') as f:
                for chunk in r.iter_content(chunk_size = 1024*1024):
                    if chunk: 
                        f.write(chunk) 
        
    print('got videos')
    return

def vid_to_image(pitcher_name,pitch_labels):
    '''
    inputs: pitcher name, pitch types
    outputs: saves first frame of video files separate folder, deletes video files
    '''
    file_paths = [f'../data/{pitcher_name}/video/{x}' for x in pitch_labels]
    for file_path,pitch_label in zip(file_paths,pitch_labels):
        vid_list = os.listdir(file_path)
        vid_list.pop(0)

        pitch = 0
        for vid_file in vid_list:
            vidcap = cv2.VideoCapture(file_path+'/'+vid_file)
            success,image = vidcap.read()
            count = 0
            while count == 0:
                cv2.imwrite(f'../data/{pitcher_name}/image/{pitch_label}/pitch%d.jpg' % pitch,                            image)     # save frame as JPEG file      
                count += 1
                pitch += 1
    
    os.rmdir(f'../data/{pitcher_name}/videos/')
    print('got images')
    return

if __name__ == '__main__':
    download_videos(pitcher_name,pitch_labels)
    vid_to_image(pitcher_name,pitch_labels)