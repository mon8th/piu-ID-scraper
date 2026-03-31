from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os 
import requests
import re
from urllib.parse import urljoin
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

FACILITIES = {
    '01': {'name': 'Engineering', 'departments': {
        '01': 'CE',
        '03': 'ARC',
        '05': 'IE'
    }},
    '02': {'name': 'Information Technology', 'departments': {
        '01': 'CS',
        '02': 'DAD',
        '03': 'MIS'
    }},
    '04': {'name': 'Economics and Administrative Sciences', 'departments': {
        '01': 'BAF',
        '02': 'BUS',
        '03': 'IR',
        '04': 'ITL'
    }}
}

def create_student_ids(years, student_max=35):
    student_ids = []
    for year in years:
        for facil_code, facil_info in FACILITIES.items():
            for dept_code, dept_name in facil_info['departments'].items(): 
                for i in range(1, student_max + 1):
                    student_id = f"{year}{facil_code}{dept_code}{i:03d}"
                    student_ids.append(student_id)
    return student_ids

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        browser = webdriver.Chrome(options=options)
    except:
        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    return browser

def scrape_student(student_id, browser=None):
    should_close = False
    if browser is None:
        browser = setup_driver()
        should_close = True
    
    try: 
        url = f"https://my.paragoniu.edu.kh/qr?student_id={student_id}" 
        browser.get(url)
        
        try:
            wait = WebDriverWait(browser, 5) 
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".v-data-table table tbody tr, .key-value-table tr")))
        except TimeoutException:
            print(f"Timeout waiting for data for student ID: {student_id}")
        
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        student_data = {'student_id': student_id}  
        
        table_rows = soup.select(".v-data-table table tbody tr")
        if not table_rows:
            table_rows = soup.select(".key-value-table tr")
        
        for row in table_rows:
            cells = row.find_all('td')
            if len(cells) == 2:
                key = cells[0].text.strip()
                value = cells[1].text.strip()
                student_data[key] = value
        
        if len(student_data) == 1:
            print(f"Failed to scrape data for student ID: {student_id}")
            return None
        
        profile_image_url = None
        elements_with_style = soup.select('[style*="background-image"]')
        for element in elements_with_style:
            style = element.get('style', '')
            if 'background-image' in style:
                match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                if match:
                    profile_image_url = match.group(1)
                    break
        
        if profile_image_url:
            if profile_image_url.startswith('/'):
                profile_image_url = urljoin(url, profile_image_url)
                
            os.makedirs('images', exist_ok=True)
            image_path = os.path.join('images', f"{student_id}.jpg") 
            if not os.path.exists(image_path):
                response = requests.get(profile_image_url)
                if response.status_code == 200:
                    with open(image_path, 'wb') as file:
                        file.write(response.content)
                    student_data['profile_picture_path'] = image_path 
                else:
                    print(f"Failed to download image for student ID: {student_id}")
        else:
            print(f"No profile image found for student ID: {student_id}")
        
        # Simplified output to reduce terminal clutter
        print(f"✓ {student_id}: {student_data.get('Name', 'Not found')}")
        return student_data
            
    finally: 
        if should_close:
            browser.quit()

def save_csv(results, filename='students.csv'):
    if not results or len(results) == 0:
        print("No data to save.")
        return
    
    # Fixed: handle fieldnames properly
    all_fields = set()
    for result in results:
        if result:
            all_fields.update(result.keys())
    
    fieldnames = list(all_fields)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result:
                writer.writerow(result)
    print(f"Data saved to {filename}")
            
def save_batch(results, batch_num):
    if not results:
        return
    
    csv_filename = f'students_batch_{batch_num}.csv'
    save_csv(results, filename=csv_filename)
    
    json_filename = f'student_data_batch_{batch_num}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Saved batch {batch_num} with {len(results)} records")

def main():
    years = ['25'] 
    student_max = 57  
    
    student_ids = create_student_ids(years, student_max)
    print(f"Generated {len(student_ids)} student IDs.")
    
    # Number of parallel workers
    max_workers = 5
    # Batch size for saving progress
    batch_size = 50
    
    all_results = []
    batch_num = 1
    
    # Create a pool of browser instances to be reused
    browsers = [setup_driver() for _ in range(max_workers)]
    
    try:
        for i in range(0, len(student_ids), batch_size):
            batch_ids = student_ids[i:i+batch_size]
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {
                    executor.submit(scrape_student, student_id, browsers[idx % len(browsers)]): student_id
                    for idx, student_id in enumerate(batch_ids)
                }
                
                for future in as_completed(future_to_id):
                    student_id = future_to_id[future]
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                    except Exception as exc:
                        print(f"{student_id} generated an exception: {exc}")
                        
            all_results.extend(batch_results)
            save_batch(batch_results, batch_num)
            batch_num += 1
    
    finally:
        for browser in browsers:
            try:
                browser.quit()
            except:
                pass
    
    save_csv(all_results, filename='students_complete.csv')
    
    with open('student_data_complete.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"Scraped {len(all_results)} students successfully out of {len(student_ids)} attempts")

if __name__ == "__main__":
    main()