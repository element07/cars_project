import datetime
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import time
from random import randint
import os

script_start = time.time()

# scraping one single page with fully detailed offer
def single_offer_scrap(offer_url):

    try:
        html = requests.get(offer_url, allow_redirects=False).text
        soup = BeautifulSoup(html, 'lxml')
        
        # getting available categories
        cols = []
        for cat in soup.find_all('span', attrs = {'class': 'offer-params__label'}):
            cols.append(cat.text)
        
        # getting categories values
        car_stats = []
        for cat_value in soup.find_all('div', attrs = {'class': 'offer-params__value'}):
            car_stats.append(cat_value.text.strip())

        # get price
        #if type(soup.find('div', attrs = {'class': 'offer-price'})['data-price']) != 'NoneType':
        price = soup.find('span', attrs = {'class': 'offer-price__number'}).text.strip()

        df_offer = pd.DataFrame()
        df_offer['columns'] = cols
        df_offer['car_stats'] = car_stats
        df_offer = df_offer.append({'columns': 'Cena', 'car_stats': price}, ignore_index=True)

        return df_offer
    except:
        # if page doesn't response, is broken, offer has expired etc. then this return just empty df to merge later on while executing script
        return pd.DataFrame(columns=['columns', 'car_stats'])


# getting list of offers per one page
def get_offers(page_url):

    try:
        html = requests.get(page_url).text
        soup = BeautifulSoup(html, 'lxml')

        # get urls of every offer on single page
        offers_list = []

        # OLD NOT WORKING
        offers_scrap_url = soup.find_all('h2', attrs = {'class': 'e1b25f6f12'})

        # NEW FIXED 2023-01-16
        offers_scrap_url = soup.find_all('a', attrs = {'target': '_self'}, href = True)

        for hrefs in offers_scrap_url:

            #hrf = hrefs.find('a', href=True) 
            #offers_list.append(hrf['href'])
            offers_list.append(hrefs['href'])

        return offers_list
    except Exception as e:
        print(e,' get_offers error')
        pass       

page_url = 'https://www.otomoto.pl/osobowe/alfa-romeo'

# scraping data from one page (all the offers on 1 page) into dataframe

def scrap_whole_page(page_url, i_mark = '', i_page = 0, max_page = '?', mark = 'mark', elapsed_times = list()):

    offers_list = get_offers(page_url)
    df = pd.DataFrame(columns=['columns'])

    if len(offers_list) > 0:
        i=i_mark+str(i_page)
        i_int = 0
        # elapsed_times = []
        
        for url in offers_list:

            start_time = time.time()

            #creating index for every new offer (column) to create distinct names (needed to pivot dataframe at the end)
            i_int+=1
            i+=str(i_int)    

            df_offer = single_offer_scrap(url)
            df_offer.rename(columns={'car_stats': 'car_stats{}'.format(i)}, inplace=True)
            # outer join is required as number of characteristics per offer may be different cause not all of them are required
            df = pd.merge(df, df_offer,on = 'columns', how = 'outer')

            # creating some info to track what's happening while script is going (it takes a lot of time)
            #time.sleep(randint(1,3))
            end_time = time.time()
            elapsed_time = np.round(end_time - start_time,2)
            elapsed_times.append(elapsed_time)
            os.system('cls')

            print('Current page URL: {}'.format(page_url))
            print('Current URL: {}\n'.format(url))
            print('Current page: {} out of {}'.format(i_page, max_page))
            print('Current mark category: {}'.format(i_mark))
            print('Processing offer {} on page {} for mark: {}\nTime elapsed: {}'.format(i_int, i_page, i_mark, elapsed_time))
            print('Giving index: {}\n'.format(i))
            print('Average elapsed time per offer: {}'.format(sum(elapsed_times)/len(elapsed_times)))#np.mean(elapsed_times)))
            print('Total executing time: {}'.format(time.time()-script_start))

            # necessary reset of index to do not add another value to the string, but update previous one
            i=i_mark+str(i_page)

    return df, elapsed_times

def scrape_by_mark(marks_list, name_suffix = ''):
    #creating empty df with column to enable join
    df = pd.DataFrame(columns=['columns'])
    elapsed_times = [0]
    for mark in marks_list:
        try:
            mark_url = 'https://www.otomoto.pl/osobowe/{}'.format(mark)
            html = requests.get(mark_url, allow_redirects=False)
            html.raise_for_status()

            soup = BeautifulSoup(html.text, 'lxml')
            
            # EDIT 2023-01-16 -> scraping last page number doesnt work
            #pages_number = soup.find_all('a', attrs= {'class': 'ooa-xdlax9 ekxs86z0'})[-1] # old class name = 'ooa-g4wbjr ekxs86z0'

            #last_page = int(pages_number.text)
            last_page = 2

            for page in range(1,last_page+1):
                url = 'https://www.otomoto.pl/osobowe/{}?page={}'.format(mark,page)

                df_temp, elapsed_times_temp = scrap_whole_page(url, i_mark = mark, i_page = page, max_page = last_page, elapsed_times = elapsed_times)
                
                elapsed_times = elapsed_times + elapsed_times_temp

                df = pd.merge(df, df_temp,on = 'columns', how = 'outer')
        except Exception as e:
            print(e)
            pass

    # pivoting the data to make all the cars charateristicts appear as columns and each car as single row in df
    df = df.melt(value_vars=df.columns[1:], id_vars = 'columns').pivot(index = 'variable', columns = 'columns').reset_index(drop=True)
    
    file_date = str(datetime.datetime.now().date())
    df.to_excel('scraping/cars_{} {}.xlsx'.format(name_suffix,file_date)) 
    df.to_csv('scraping/cars_{} {}.csv'.format(name_suffix,file_date)) 

# required list of marks cause after 500th page url of website doesn't change 
# so we have to split the offers by some categories to make pages per category < 500

marks_full = ['nissan', 'toyota', 'bmw', 'audi', 'opel', 'skoda', 'peugeot', 'renault', 'citroen', 'alfa-romeo', 'chevrolet',
'chrysler', 'dacia', 'dodge', 'fiat', 'ford', 'honda', 'hyundai', 'jaguar', 'jeep', 'land-rover', 'mazda',
'mercedes-benz', 'mini', 'mitsubishi', 'porsche', 'seat', 'suzuki', 'kia', 'subaru', 'volkswagen', 'volvo']

marks_full = ['nissan', 'toyota']

scrape_by_mark(marks_full[:11], name_suffix='0-11')
#scrape_by_mark(marks_full[11:22], name_suffix='11-22')
#scrape_by_mark(marks_full[22:32], name_suffix='22-32')

# merging files
'''
path = 'otomoto scraping/scraping results/excel'
full_data = pd.DataFrame()
for dir in os.listdir(path):
    df = pd.read_excel(path+'/'+dir)
    full_data = pd.concat([full_data, df])full_data.append(df)

#full_data = full_data.dropna(how = 'all')
full_data.to_excel(path+'/full_scraping 2022-09-01.xlsx')
'''


print('Finished.')