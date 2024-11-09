import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os

# Global set to store unique URLs
visited_urls = set()
output_dir="crawled_pages"

def crawl(seed_url, max_pages=250, delay=1, output_dir=output_dir):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the queue with the seed URL and set the page counter to 0
    queue = [seed_url]
    page_count = 0

    # Loop while there are URLs in the queue and the page limit hasn't been reached
    while queue and page_count < max_pages:
        # Pop the first URL from the queue
        url = queue.pop(0)

        # Only proceed if the URL hasn't been visited before
        if url not in visited_urls:
            try:
                # Send a GET request to fetch the URL content with a 10-second timeout
                response = requests.get(url, timeout=10)
                # Raise an error if the request was unsuccessful (status code 4xx or 5xx)
                response.raise_for_status()

                # Process the fetched page and save its content to the output directory
                save_page(url, response.text, output_dir)

                # Mark the URL as visited by adding it to the set of visited URLs
                visited_urls.add(url)
                # Increment the count of crawled pages
                page_count += 1

                # Print a success message indicating the page has been crawled
                print(f"Page {page_count}: {url} crawled successfully.")

                # Parse the page content to find all the links ('a' tags) on the page
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    # Convert relative links to absolute URLs
                    absolute_url = urljoin(url, link['href'])
                    # Ensure the link is valid (e.g., same domain) and hasn't been visited
                    if is_valid(absolute_url, seed_url) and absolute_url not in visited_urls:
                        # Add valid URLs to the queue for future crawling
                        queue.append(absolute_url)

                # Wait for a specified delay before making the next request
                time.sleep(delay)

            # Handle exceptions like network errors or invalid URLs
            except (requests.RequestException, Exception) as e:
                # Print a failure message and continue with the next URL
                print(f"Failed to crawl {url}: {e}")
                continue


def save_page(url, content, output_dir):
    # Parse domain and path to create filename
    parsed_url = urlparse(url)
    file_name = f"{parsed_url.netloc}.txt"
    file_path = os.path.join(output_dir, file_name)

    # Use BeautifulSoup to parse the page content
    soup = BeautifulSoup(content, 'html.parser')

    # Remove scripts, styles, and other irrelevant elements
    for script in soup(["script", "style"]):
        script.extract()  # Remove these tags from the HTML content

    # Get the text content of the page
    text_content = soup.get_text()

    # Break lines and strip leading/trailing spaces
    lines = [line.strip() for line in text_content.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Save the cleaned text content to a .txt file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

    print(f"Content from {url} saved to {file_path}")
    

def is_valid(url, seed_url):
    # Only crawl URLs from the same domain as the seed URL
    return urlparse(url).netloc == urlparse(seed_url).netloc

if __name__ == "__main__":
    seed_page = "https://www.cookingclassy.com/"  # Replace with the actual seed page
    crawl(seed_page, max_pages=250, delay=1)
