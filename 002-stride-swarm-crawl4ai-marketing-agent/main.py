import os
import asyncio
import logging
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import CosineStrategy
import dotenv
from openai import OpenAI
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import json
import re
# Ensure the swarm module is installed and available
# from swarm import Agent
import time

# Set up logging
log_file = 'property_scraper.log'
logging.basicConfig(level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()
logger.info("Environment variables loaded")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger.info("OpenAI client initialized")

# Function to create a folder for the website
def create_website_folder(url):
    domain = urlparse(url).netloc
    folder_name = domain.split('.')[0]
    os.makedirs(folder_name, exist_ok=True)
    logger.info(f"Created folder: {folder_name}")
    return folder_name

# Function to scrape a website using Crawl4AI
async def scrape_website(url):
    """
    Scrape a website using Crawl4AI's AsyncWebCrawler and clean the content.
    """
    logger.info(f"Starting to scrape website: {url}")
    folder_name = create_website_folder(url)
    start_time = time.time()
    try:
        async with AsyncWebCrawler(verbose=True) as crawler:
            logger.info("Initializing AsyncWebCrawler")
            result = await asyncio.wait_for(
                crawler.arun(
                    url=url,
                    extraction_strategy=CosineStrategy(
                        semantic_filter="property listings",
                        word_count_threshold=10,
                        max_dist=0.2,
                        top_k=3
                    ),
                    bypass_cache=True,
                ),
                timeout=300  # 5 minutes timeout
            )
        logger.info("Website scraping completed")
    except asyncio.TimeoutError:
        logger.error("Scraping timed out after 5 minutes")
        return None, None
    except Exception as e:
        logger.error(f"An error occurred during scraping: {str(e)}")
        return None, None

    logger.info(f"Scraping took {time.time() - start_time:.2f} seconds")
    
    # Clean and structure the content
    logger.info("Starting content cleaning and property extraction process")
    soup = BeautifulSoup(result.html, 'html.parser')
    
    # Extracting property details
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
    
    logger.info(f"Extracted {len(data)} property listings")
    
    # Save the extracted property data to a file
    property_data_file = os.path.join(folder_name, "property_data.json")
    with open(property_data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Property data saved to {property_data_file}")
    
    return data, result.extracted_content

# Define Swarm agents
class WebScraperAgent:
    async def run(self, url):
        logger.info(f"WebScraperAgent started for URL: {url}")
        result = await scrape_website(url)
        logger.info("WebScraperAgent completed")
        return result

class AnalystAgent:
    async def run(self, content):
        logger.info("AnalystAgent started")
        result = analyze_property_data(content)
        logger.info("AnalystAgent completed")
        return result

# Function to analyze property data
def analyze_property_data(data):
    """
    Analyze the scraped property data using OpenAI.
    """
    logger.info("Starting property data analysis")
    analysis = generate_completion(
        "real estate analyst",
        "Analyze the following property data and provide key insights for potential buyers or investors.",
        json.dumps(data)
    )
    logger.info("Property data analysis completed")
    return {"analysis": analysis}

# Function to generate completions using OpenAI
def generate_completion(role, task, content):
    """
    Generate a completion using OpenAI's GPT model.
    This function demonstrates how to interact with OpenAI's API.
    """
    logger.info(f"Generating completion for role: {role}")
    response = client.chat.completions.create(
        model="gpt-4o",  # Using GPT-4o for high-quality responses
        messages=[
            {"role": "system", "content": f"You are a {role}. {task}"},
            {"role": "user", "content": content}
        ]
    )
    logger.info("Completion generated successfully")
    return response.choices[0].message.content

class UserInterfaceAgent:
    async def run(self):
        logger.info("UserInterfaceAgent started")
        url = input("Please enter a URL to analyze (e.g., a Boliga property listing page): ")
        logger.info(f"User entered URL: {url}")
        
        scraper_agent = WebScraperAgent()
        property_data, extracted_content = await scraper_agent.run(url)
        
        if property_data is None or extracted_content is None:
            logger.error("Scraping failed. Please check the logs for more information.")
            return

        folder_name = create_website_folder(url)
        
        analyst_agent = AnalystAgent()
        analysis = await analyst_agent.run(property_data)
        
        # Save analysis
        analysis_file = os.path.join(folder_name, "property_analysis.md")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis['analysis'])
        logger.info(f"Property analysis saved to {analysis_file}")
        
        # Create and save comprehensive property report
        property_report = f"""# Comprehensive Property Report

## Extracted Property Data
{json.dumps(property_data, indent=2)}

## Property Analysis
{analysis['analysis']}
"""
        report_file = os.path.join(folder_name, "property-report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(property_report)
        logger.info(f"Comprehensive property report saved to {report_file}")
        
        print(f"All output files have been saved in the '{folder_name}' folder.")
        print("Analysis completed. Thank you for using our property analysis assistant!")
        print("Thank you for using the Stride Swarm AI Agent - to have our team implement AI agents into your business, book a call at https://executivestride.com/apply")
        logger.info("UserInterfaceAgent completed")

# Main execution
async def main():
    logger.info("Main execution started")
    ui_agent = UserInterfaceAgent()
    await ui_agent.run()
    logger.info("Main execution completed")
    print(f"Logs have been saved to {os.path.abspath(log_file)}")

if __name__ == "__main__":
    logger.info("Script started")
    asyncio.run(main())
    logger.info("Script completed")
