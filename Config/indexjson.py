from Config.config import config 
import csv
import json
import os

def csv_to_json(csv_file_path=config['PATHS']['INDEX_LINKS'], json_file_path=config['PATHS']['INDEX_JSON']):

    # Initialize the dictionary to store the data
    data_dict = {}
    
    # Read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Process each row in the CSV
        for row in csv_reader:
            index_name = row['Indexes']
            
            # Replace invalid filename characters with underscore
            invalid_chars = r'<>:"/\\|?*'
            for char in invalid_chars:
                index_name = index_name.replace(char, '_')
            
            if row['Status'].lower() == 'working':

                # Create a dictionary for the current index
                index_data = {
                    'URL': row['URL'],
                    'Status': row['Status'],
                    'Category': row['Category'],
                    'Note': row['Note']
                }
                
                # Add to the main dictionary
                data_dict[index_name] = index_data
    
    # Write to JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, indent=4)


def get_index_url(index_name, json_file_path=config['PATHS']['INDEX_JSON']):
    if not os.path.exists(json_file_path):
        csv_to_json()
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data.get(index_name, {})

def get_index_json(json_file_path=config['PATHS']['INDEX_JSON']):
    if not os.path.exists(json_file_path):
        csv_to_json()
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data