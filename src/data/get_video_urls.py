from pyforest import *
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import time
import cv2
import os

def get_page_url(webpages,pitcher_name,pitch_labels):
    '''
    inputs: list of urls to pages listing all pitches for type, pitcher name, pitch types
    outputs: saves lists of urls for each pitch type as txt files
    '''

    chromedriver = "/Applications/chromedriver"
	os.environ["webdriver.chrome.driver"] = chromedriver

    vid_page_urls = []
    
    if not os.path.exists(f'../data/{pitcher_name}'):
        os.mkdir(f'../data/{pitcher_name}')
        os.mkdir(f'../data/{pitcher_name}/vid_lists')
    
    for webpage,pitch_label in zip(webpages,pitch_labels):
        driver = webdriver.Chrome(chromedriver)
        driver.get(webpage)
        time.sleep(2)

        open_table = driver.find_element_by_xpath('//tr[contains(@class, "search_row")]')
        open_table.click()
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source)


        table = soup.find('table',class_ = 'table table-bordered table-hover table-condensed')
        vid_icons = table.find_all('a')
        pitch_page_urls = [vid.get('href') for vid in vid_icons]
        vid_page_urls.append(pitch_page_urls)
        
        with open(f'../data/{pitcher_name}/vid_lists/{pitch_label}.txt','w') as file:
            for url in vid_page_urls:
                file.write(f'{url}\n')
        
    print('got urls')

if __name__ == '__main__':
	get_page_url(webpages,pitcher_name,pitch_labels)



