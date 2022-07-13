from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import requests
import json


options = Options()
options.binary_location = "C:\\Users\\armaa\\AppData\\Local\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
driver_path = "chromedriver.exe"

driver = webdriver.Chrome(options=options, executable_path=driver_path)

dataset_path = "D:\\Pictures\\Skribbl Dataset\\images"
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)


def download_images(query, num_images, driver):
    folder_path = os.path.join(dataset_path, query)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    def scroll_to_end(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    # def download_image(url, num):
    #     try:
    #         print(url)
    #         with open(os.path.join(folder_path, query + str(num) + ".jpeg"), 'wb') as file:
    #             file.write(requests.get(url).content)
    #             file.close()
    #     except Exception:
    #         print("fraudulent image")

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
    driver.get(search_url.format(q=query))
    driver.maximize_window()

    time.sleep(1)

    scroll_to_end(driver)

    for i in range(1, num_images):
        try:
            img = driver.find_element(By.XPATH, '//*[@id="islrg"]/div[1]/div[' + str(i) + ']/a[1]/div[1]/img')
            img.screenshot(os.path.join(folder_path, query + str(i) + ".jpg"))
            time.sleep(0.5)
        except Exception:
            continue

    # image_count = 0
    # results_start = 0
    # while image_count < num_images:
    #     thumbnail_results = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    #     number_results = len(thumbnail_results)
    #
    #     for img in thumbnail_results[results_start:number_results]:
    #         try:
    #             img.click()
    #         except Exception:
    #             continue
    #
    #         actual_images = driver.find_elements(By.CSS_SELECTOR, 'img.n3VNCb')
    #         for actual_image in actual_images:
    #             if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
    #                 download_image(actual_image.get_attribute('src'), image_count)
    #                 time.sleep(1)
    #                 image_count += 1
    #
    #         if image_count > num_images:
    #             break
    #
    #     results_start = len(thumbnail_results)


with open("modified_words.json") as file:
    data = json.load(file)

for word in data:
    download_images(word.replace("/", " ") + " drawing", 100, driver)

driver.close()
