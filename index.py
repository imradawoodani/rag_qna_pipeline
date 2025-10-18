import logging
import sys
import asyncio
import requests
import pickle
import time
import re
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree as ET
from typing import List, Dict

from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from playwright.async_api import async_playwright
except ImportError:
    logging.error("Playwright not installed. Please run 'pip install playwright' and 'playwright install'.")
    sys.exit(1)
try:
    import trafilatura
except ImportError:
    trafilatura = None

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout 
)
logging.getLogger("playwright").setLevel(logging.WARNING) # Make playwright less noisy
logger = logging.getLogger(__name__)

logger.info("Initial logging configured. Starting imports!")


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        scheme = 'https'
        netloc = p.netloc.lower().replace('www.', '')
        path = re.sub(r'/+', '/', p.path or '/')
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return url

def extract_main_text_from_html(html: str) -> str:
    if trafilatura:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
        if extracted and len(extracted) > 150:
            return clean_text(extracted)
    
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    return clean_text(text)

def extract_title(soup: BeautifulSoup) -> str:
    og = soup.find('meta', property='og:title')
    if og and og.get('content'):
        return og.get('content').strip()
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    h1 = soup.find('h1')
    return h1.get_text().strip() if h1 else ""


async def scrape_url_with_playwright(url: str) -> Dict[str, str]:
    logger.info(f"Scraping with Playwright: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        title_text, main_content = "", ""
        try:
            await page.goto(url, timeout=90000, wait_until='networkidle')
            html_content = await page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            title_text = extract_title(soup)
            main_content = extract_main_text_from_html(html_content)
        except Exception as e:
            logger.error(f"Playwright failed for {url}: {e}")
        finally:
            await context.close()
            await browser.close()
    return {'url': url, 'title': title_text, 'content': main_content, 'length': len(main_content)}

def discover_eluvio_pages(base_url: str = "https://eluv.io") -> List[str]:
    logger.info("Discovering URLs via sitemap and common paths!")
    discovered_urls = set()
    
    def fetch_sitemap_urls(base: str) -> List[str]:
        urls = []
        sitemap_url = urljoin(base, "/sitemap_index.xml")
        try:
            resp = requests.get(sitemap_url, timeout=20)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for loc in root.iter():
                    if 'loc' in loc.tag and loc.text:
                        urls.append(loc.text.strip())
        except Exception as e:
            logger.warning(f"Could not fetch sitemap: {e}")
        return urls

    common_paths = [
        "/", "/about", "/about/news", "/careers", "/content-fabric", "/content-fabric/blockchain", 
        "/content-fabric/technology", "/av-core/fabric-core", "/av-core/core-utilities", 
        "/monetization/analytics", "/monetization/media-wallet", "/monetization/creator-studio", 
        "/video-intelligence/video-editor", "/video-intelligence/ai-search", 
    ]
    
    for path in common_paths:
        discovered_urls.add(canonicalize_url(urljoin(base_url, path)))
        
    sitemap_urls = fetch_sitemap_urls(base_url)
    for url in sitemap_urls:
        if "eluv.io" in urlparse(url).netloc:
            discovered_urls.add(canonicalize_url(url))

    logger.info(f"Discovered {len(discovered_urls)} unique URLs.")
    return sorted(list(discovered_urls))

async def main():
    logger.info("Starting Eluvio RAG index creation.")
    urls = discover_eluvio_pages()
    logger.info(f"Found {len(urls)} URLs to scrape.")

    scraped_data = []
    for url in urls:
        data = await scrape_url_with_playwright(url)
        if data['content'] and len(data['content']) > 150:
            scraped_data.append(data)
            logger.info(f"Successfully scraped {url}: {data['length']} characters")
        else:
            logger.warning(f"No significant content extracted from {url}")
        time.sleep(1)

    if not scraped_data:
        raise RuntimeError("Index build failed: No pages returned significant content.")
 
    logger.info(f"Successfully scraped {len(scraped_data)} pages. Creating text chunks!")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    enhanced_chunks = []
    for data in scraped_data:
        chunks = splitter.split_text(data['content'])
        for i, chunk_text in enumerate(chunks):
            enhanced_chunks.append({
                'content': f"{data['title']}. {chunk_text}",
                'metadata': {'url': data['url'], 'title': data['title'], 'chunk_index': i}
            })

    logger.info(f"Created {len(enhanced_chunks)} enhanced chunks. Loading embedding model.")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    logger.info("Creating FAISS index.")
    db = FAISS.from_texts(
        texts=[chunk['content'] for chunk in enhanced_chunks],
        embedding=embedder,
        metadatas=[chunk['metadata'] for chunk in enhanced_chunks]
    )
    
    logger.info("Saving index and chunks.")
    db.save_local("eluvio_index")
    with open('text_chunks.pkl', 'wb') as f:
        pickle.dump(enhanced_chunks, f)
    
    stats = {
        'total_pages': len(scraped_data),
        'total_chunks': len(enhanced_chunks),
        'total_characters': sum(len(c['content']) for c in enhanced_chunks),
    }
    stats['avg_chunk_size'] = stats['total_characters'] / stats['total_chunks'] if stats['total_chunks'] else 0
    with open('index_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    logger.info("Index creation complete!")
    logger.info(f"Pages scraped: {stats['total_pages']}")
    logger.info(f"Chunks created: {stats['total_chunks']}")

if __name__ == "__main__":
    asyncio.run(main())