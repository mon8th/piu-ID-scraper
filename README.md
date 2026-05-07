# Paragon University ID Data Scraper
A web scraper tool that extracts student information from Paragon International University's student portal using generated student IDs. The tool saves student data in both CSV and JSON formats and downloads profile pictures.

## Features
- Generates student IDs based on year, faculty, and department codes
- Scrapes student profile information from the university portal
- Downloads student profile pictures
- Saves data in both CSV and JSON formats
- Uses multithreading for faster data collection
- Implements batch processing to prevent data loss
- Deduplicates results using the site's actual ID Number to prevent false entries

## Requirements
- Python 3.6+
- Chrome browser
- Internet connection

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mon8th/piu-ID-scraper
cd piu-ID-scraper
```

2. Create and activate a virtual environment:
```bash
# Create venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install selenium beautifulsoup4 webdriver-manager requests
```

4. Ensure Chrome is installed on your system:
https://developer.chrome.com/docs/chromedriver/downloads

## Usage
Run the scraper with default settings:
```bash
python data_scraping/chhay2.py
```

By default, the script will:
- Generate student IDs for the year '22' and all faculties/departments
- Scrape data with 5 concurrent workers
- Process students in batches of 50
- Save results in both CSV and JSON formats inside a `data_22/` folder

## How It Works
1. **Student ID Generation**: The script generates student IDs following the pattern `[Year][Faculty Code][Department Code][Student Number]` (e.g., "2201001" for 2022, Engineering Faculty, CE Department, first student).
2. **Concurrent Scraping**: Multiple Chrome browser instances are launched in headless mode to scrape data in parallel.
3. **Data Extraction**: For each student ID, the script:
   - Visits the student's profile page
   - Extracts information from the page's HTML
   - Downloads the student's profile picture if available
4. **Batch Processing**: Data is processed in batches to prevent data loss in case of errors.
5. **Data Storage**: Results are saved inside `data_[year]/`:
   - Individual batch CSV files (`students_batch_X.csv`)
   - Individual batch JSON files (`student_data_batch_X.json`)
   - Complete dataset (`students_complete.csv` and `student_data_complete.json`)
   - Profile pictures in `data_[year]/images/`

## Known Issues

### Duplicate Entries
The university portal sometimes returns a real student's data for non-existent IDs. This means two different generated IDs can return the same person.

The script handles this by deduplicating on the site's actual `ID Number` field before saving the final output. The lower student ID always takes priority. Batch files may still contain duplicates but `students_complete.csv` and `student_data_complete.json` will be clean.

## Customization
Edit the main function in [data_scraping/chhay2.py](data_scraping/chhay2.py) to adjust:
- Target years (modify the `years` list)
- Maximum number of students per department (modify `student_max`)
- Number of concurrent workers (modify `max_workers`)
- Batch size (modify `batch_size`)

## Faculty and Department Codes
The script supports the following faculties and departments:
- **01**: Engineering
  - **01**: CE (Civil Engineering)
  - **03**: ARC (Architecture)
  - **05**: IE (Industrial Engineering)
- **02**: Information Technology
  - **01**: CS (Computer Science)
  - **02**: DAD (Digital Arts and Design)
  - **03**: MIS (Management Information Systems)
- **04**: Economics and Administrative Sciences
  - **01**: BAF (Banking and Finance)
  - **02**: BUS (Business Administration)
  - **03**: IR (International Relations)
  - **04**: ITL (International Trade and Logistics)

## Troubleshooting
- If you encounter issues with Chrome not starting, make sure ChromeDriver is properly installed or let the script download it automatically.
- If you see "Timeout waiting for data" messages, try increasing the timeout value in the `scrape_student` function.
- If the script fails to find data on student pages, the website structure might have changed, requiring updates to the CSS selectors.
- When done, deactivate the virtual environment:
```bash
deactivate
```
