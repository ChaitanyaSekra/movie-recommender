from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import csv

# Setup
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.imdb.com/chart/top/")
wait = WebDriverWait(driver, 10)

# Wait for the movie table to load
wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ipc-title__text")))

# Get top 50 movie links for demonstration
movie_links = driver.find_elements(By.CSS_SELECTOR, "a.ipc-title-link-wrapper")[:250]
movie_urls = [link.get_attribute("href") for link in movie_links]

movies_data = []

for url in movie_urls:
    driver.get(url)
    time.sleep(1)

    try:
        title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1 span"))).text
    except:
        title = "N/A"

    try:
        rated = driver.find_element(By.XPATH, "//a[contains(@href, 'tt_ov_pg#certificates')]").text.strip()
    except:
        rated = "N/A"

    try:
        year = driver.find_element(By.XPATH, "//a[contains(@href, 'tt_ov_rdat')]").text.strip()
    except:
        year = "N/A"

    try:
        rating = driver.find_element(By.CLASS_NAME, "imUuxf").text
    except:
        rating= "N/A"
    
    try:
        duration = driver.find_element(By.CSS_SELECTOR, "li[data-testid='title-techspec_runtime'] div").text
    except:
        duration = "N/A"
    
    i=1
    genres=list()
    while True:
        try:
            genre_elements=driver.find_element(By.XPATH, f"//a[contains(@href, 'tt_ov_in_{i}')]").text.strip()
            genres.append(genre_elements)
            i+=1
        except:
            break
    if not genres:
        genres = 'N/A'


    try:
        director = driver.find_element(By.XPATH, "//a[contains(@href, 'tt_ov_dr')]").text.strip()
    except:
        director = "N/A"

    i=1
    stars=list()
    while True:
        try:
            star_elements=driver.find_element(By.XPATH, f"//a[contains(@href, 'tt_ov_st_{i}')]").text.strip()
            stars.append(star_elements)
            i+=1
        except:
            break
    if not stars:
        stars = 'N/A'


    movie = {
        "Title": title,
        "Rated":rated,
        "Rating": rating,
        "Year": year,
        "Duration": duration,
        "Genre": genres,
        "Director": director,
        "Stars": stars
    }

    movies_data.append(movie)

# Save to CSV
with open("detailed_movies.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Title", "Rating","Rated", "Year", "Duration", "Genre", "Director", "Stars"])
    writer.writeheader()
    writer.writerows(movies_data)

driver.quit()