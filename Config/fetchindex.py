from urllib.parse import urljoin
import pandas as pd
from Config.config import config
from Config.indexjson import get_index_json
import requests
from bs4 import BeautifulSoup
import os

def fetch_nse_index_csv(base_url, file_name, file_path=config['PATHS']['INDEXES_DIR']):

    # Headers to mimic a browser visit
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    # Send a GET request to the URL
    response = requests.get(base_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for all anchor tags with href attribute
        links = soup.find_all('a', href=True)

        csv_link = None
        for link in links:
            href = link['href']
            if href.endswith('.csv'):
                # Ensure correct full URL
                csv_link = urljoin(base_url, href)
                break

        if csv_link:
            print(f"Downloading from: {csv_link}")
            csv_response = requests.get(csv_link, headers=headers)

            if csv_response.status_code == 200:
                filename = os.path.join(file_path, file_name + '.csv')
                with open(filename, 'wb') as f:
                    f.write(csv_response.content)
                print(f"CSV saved as: {filename}")
            else:
                print("Failed to download CSV file.")
        else:
            print("CSV link not found on the page.")
    else:
        print(f"Failed to retrieve page. Status code: {response.status_code}")

def read_csv_file(file_path):
    content = pd.read_csv(file_path)
    return content

def fetch_all_nse_index_csv():
    json_data = get_index_json()
    for index_name, index_info in json_data.items():
        url = index_info['URL']
        status = index_info['Status']
        if status.lower() == 'working':
            fetch_nse_index_csv(url, index_name)
        else:
            print('Skipping index:', index_name, ';  Reason:', status)