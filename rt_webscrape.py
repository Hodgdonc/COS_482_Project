## TODO keep track of the number of movies you're scraping and start where you left off when you change pages

# imports all needed libraries needed for this assignment and sets the options and location 
# of the chrome webdriver needed for selenium.

from bs4 import BeautifulSoup as bs
import requests
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium import webdriver
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument("--disable-javascript")
options.add_argument('--incognito')
#options.add_argument('--headless')

driver = webdriver.Chrome(executable_path=r"Users/Matt/Desktop/School/DataSci/Hw1",options=options)

# creates a datafram with all of the needed columns including "title", "author", etc.
df = pd.DataFrame({'title': pd.Series(dtype='str'),
                   'critic_score': pd.Series(dtype='str'),
                   'audience_score': pd.Series(dtype='str')
})

#specifies the website to scrape and use selenium
driver.get("https://www.rottentomatoes.com/browse/movies_at_home/sort:critic_highest")

WebDriverWait(driver, 1).until(
    lambda s: s.find_element(By.CLASS_NAME,"js-tile-link").is_displayed()
)

i = 0
while True:
    try:
        # waits until the head-block class is displayed
        WebDriverWait(driver, 1).until(
            lambda s: s.find_element(By.CLASS_NAME,"js-tile-link").is_displayed()
        )
    except TimeoutException:
        break

    url = driver.current_url
    response = requests.get(url)
    html = response.content
    #creates of beautiful soup object of the current url
    soup = bs(html, "lxml")
    
    time.sleep(1)
    #makes a list of each div with a class of "item-info"
    all_item_info = soup.find_all(class_="js-tile-link")
    #for each item in the list, gather all the relevant data about the books
    for item_info in all_item_info: 
        try:
            movie_title = item_info.find("span", class_="p--small").get_text(strip=True)
        except:
            movie_title = pd.NA
        try:
            critic_rating = item_info.find("score-pairs",attrs={"criticsscore":True}).get("criticsscore")
        except:
            critic_rating = pd.NA
        try:
            audience_rating = item_info.find("score-pairs", attrs={"audiencescore":True}).get("audiencescore")
        except:
            audience_rating = pd.NA
        
        #adds each of these values as a new row to the dataframe
        df = pd.concat([df,pd.Series({'title': movie_title,
                    'critic_score': critic_rating,
                    'audience_score': audience_rating,
                    }).to_frame().T], ignore_index=True)
            
    
    #makes sure the next button exists before trying to click it
    try:
        load_more = driver.find_element(By.XPATH,'//button[text()="Load more"]')
        if load_more: driver.execute_script("arguments[0].click();", load_more)
    except: 
        print("gooooooooooo")
        pass
    
    #after each page it increments i by 1, once i is greater than 33 we have reached
    #the last page so it breaks out of the while statement
    i += 1
    if i > 33:
        break

#drop the duplicate titles - load more keeps the very first movie you encountered
#at the same spot on the top of the page, so you'll be scraping all the stuff you
#just scraped each time through the loop before getting to the new stuff. Simple
#solution is to drop duplicates on title

df = df.drop_duplicates(subset=['title'])

#saves the dataframe as a csv file
df.to_csv('rotten_tomatoes.csv')