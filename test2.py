import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarm import Swarm, Agent
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from stride_ai_agents import AgentManager



from stride_ai_agents import AgentManager  # Importing Stride-AI for agent orchestration

# Define the web scraping agents using Stride-AI
page_loader_agent = Agent(
    name='Page Loader Agent',
    model='gpt-3.5-turbo',
    instructions='Load the given webpage and handle network retries to ensure successful loading.',
    functions=[]
)

data_extractor_agent = Agent(
    name='Data Extractor Agent',
    model='gpt-3.5-turbo',
    instructions='Extract relevant property information from the loaded webpage, including property type, rooms, size, price, address, energy mark, and listing date.',
    functions=[]
)

data_storage_agent = Agent(
    name='Data Storage Agent',
    model='gpt-3.5-turbo',
    instructions='Store the extracted data into an Excel file.',
    functions=[]
)

# Define tasks for each agent
tasks = [
    {
        "description": 'Load the website "https://www.boliga.dk/nye-boliger" and ensure the page is properly loaded for data extraction.',
        "agent": page_loader_agent
    },
    {
        "description": 'Extract information from the loaded webpage, such as property type, rooms, size, price, address, energy mark, and listing date.',
        "agent": data_extractor_agent
    },
    {
        "description": 'Save the extracted data to an Excel file.',
        "agent": data_storage_agent
    }
]

# Web scraping function implementation with retry logic and custom headers
def web_scraper(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    # Setup retry strategy
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    
    try:
        response = http.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting property details such as type, rooms, size, price, address, etc.
    data = []
    for item in soup.find_all('div', class_='search-list-item'):  # Updated class to match Boliga's structure
        property_type = item.find('div', class_='property-type').get_text(strip=True) if item.find('div', class_='property-type') else 'N/A'
        rooms = item.find('div', class_='rooms').get_text(strip=True) if item.find('div', class_='rooms') else 'N/A'
        size = item.find('div', class_='size').get_text(strip=True) if item.find('div', class_='size') else 'N/A'
        price = item.find('div', class_='price').get_text(strip=True) if item.find('div', class_='price') else 'N/A'
        address = item.find('div', class_='address').get_text(strip=True) if item.find('div', class_='address') else 'N/A'
        energy_mark = item.find('div', class_='energy-mark').get_text(strip=True) if item.find('div', class_='energy-mark') else 'N/A'
        listing_date = item.find('div', class_='listing-date').get_text(strip=True) if item.find('div', class_='listing-date') else 'N/A'
        data.append([property_type, rooms, size, price, address, energy_mark, listing_date])

    # Create a DataFrame and return it
    df = pd.DataFrame(data, columns=['Property Type', 'Rooms', 'Size', 'Price', 'Address', 'Energy Mark', 'Listing Date'])
    return df

# Create the swarm
swarm = Swarm()

# Run the agent tasks using Stride-AI agents
agent_manager = AgentManager(swarm)
results = []
for task in tasks:
    response = agent_manager.run_task(
        agent=task['agent'],
        messages=[{"role": "user", "content": task['description']}]
    )
    results.append(response)

# Manually run the web scraper function since actual scraping cannot be done by the language model
url = "https://www.boliga.dk/nye-boliger"
scraping_result = web_scraper(url)

# Export the results to an Excel file if scraping was successful
if not scraping_result.empty:
    scraping_result.to_excel('web_scraping_results.xlsx', index=False)
    print("\nWeb Scraping Result:")
    print(scraping_result)
else:
    print("Failed to retrieve data from the website.")