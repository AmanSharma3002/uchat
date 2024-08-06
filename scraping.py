import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import load_configuration
import asyncio
import aiohttp

groq_api_key, api_url, headers = load_configuration()

async def scrape_jina_ai(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get("https://r.jina.ai/" + url) as response:
            text = await response.text()
            return text
        
async def scrape_multiple(urls: list) -> list:
    tasks = []
    for url in urls:
        tasks.append(scrape_jina_ai(url))
    
    # Gather all tasks concurrently
    results = await asyncio.gather(*tasks)
    return results

async def scrape_website(urls):
    """
    Scrape and process website content using AsyncChromiumLoader.

    Args:
    - urls (list): List of URLs to scrape.

    Returns:
    - docs_transformed (list): Transformed documents from the scraped URLs.
    """
    #loader = AsyncChromiumLoader(urls, headless=True)
    #html2text = Html2TextTransformer()
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)

    try:
        #docs = await loader.aload()
        #docs_transformed = html2text.transform_documents(docs)
        docs = await scrape_multiple(urls)
        docs_transformed = r_splitter.create_documents(docs)
        return docs_transformed
    except Exception as e:
        print(f"Error occurred during document loading: {e}")
        return []

"""def extract_links(input_string):

    Extract links from a string.

    Args:
    - input_string (str): Input string containing links.

    Returns:
    - links_list (list): List of extracted links.

    pattern = r'\{([^}]+)\}'  # Matches text inside curly brackets
    match = re.search(pattern, input_string)
    
    if match:
        links_str = match.group(1)  # Get the content inside the curly brackets
        links_list = links_str.split(',')  # Split by commas to get individual links
        links_list = [link.strip() for link in links_list]  # Strip whitespace around links
        return links_list
    else:
        return []

def get_all_links(main_url, max_links=100):

    Get all links from a main URL.

    Args:
    - main_url (str): Main URL to extract links from.
    - max_links (int): Maximum number of links to retrieve.

    Returns:
    - links (list): List of extracted links.

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(main_url)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.startswith('#'):
            continue
        full_url = urljoin(main_url, href)
        parsed_url = urlparse(full_url)
        if parsed_url.scheme and parsed_url.netloc:
            if not parsed_url.path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', 
                                             '.js', '.css', '.pdf', '.doc', '.docx', '.xls', '.xlsx', 
                                             '.ppt', '.pptx', '.zip', '.rar', '.tar', '.gz')):
                links.append(full_url)
                if len(links)==max_links:
                    break
    
    links_text = "\n".join(links)
    try:
        client = Groq(api_key=groq_api_key)

        prompt = f"This is a list of urls found form a website can you filterout useful links from this text which can provide information about website such as contact, about, faqs, blogs, product pages etc. :\n\n{links_text}\n\n Return  10 comma seperated links in curly brackets"
        chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant", temperature=0
            )
        llm_output = chat_completion.choices[0].message.content
        print(llm_output)
        links = extract_links(llm_output)
        links.append(main_url)

    except Exception as e:
        print("Error analyzing contact info with llm:", e)
    
    driver.quit()
    print(f"Found {len(links)} links")
    return links"""
