#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from markdownify import markdownify as md

def url_to_filename(url: str) -> str:
    """
    Converts a URL into a safe and descriptive filename.
    Example: 'https://example.com/some/page/' -> 'some_page.md'
    """
    parsed_url = urlparse(url)
    path = parsed_url.path
    if not path or path == '/':
        return "index.md"
    
    filename = path.strip('/')
    filename = re.sub(r'[\/\?=&%#]', '_', filename)
    if not filename.endswith('.md'):
        filename += '.md'
    return filename
def process_and_save_docs(docs: list, output_dir: str):
    """
    Processes crawled documents to:
    1. Clean HTML content.
    2. Replace internal links with relative links to local Markdown files.
    3. Convert the final HTML to Markdown.
    4. Save each document as a separate .md file.
    """
    if not docs:
        print("[!] No documents to process.")
        return

    # --- THE FIX: Normalize all URLs by stripping trailing slashes for consistent matching ---
    scraped_urls = {doc.metadata['source'].split('#')[0].rstrip('/') for doc in docs}
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Processing {len(docs)} files for link replacement and saving to '{output_dir}/'")

    for doc in docs:
        current_url = doc.metadata['source']
        raw_html = doc.page_content
        soup = Soup(raw_html, "lxml")

        title = soup.find("title").get_text().strip() if soup.find("title") else "No Title"
        body = soup.find("main") or soup.find("article") or soup.find("body")

        if not body:
            print(f"   [!] No content body found for {current_url}. Skipping.")
            continue

        for tag in body.find_all(['img', 'svg', 'script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            tag.decompose()

        for a_tag in body.find_all("a", href=True):
            href = a_tag.get('href')
            if not href or href.startswith('mailto:'):
                continue

            absolute_link = urljoin(current_url, href)
            
            # --- THE FIX: Normalize the link's target URL before checking ---
            absolute_link_clean = absolute_link.split('#')[0].rstrip('/')

            if absolute_link_clean in scraped_urls:
                # The destination is another scraped page. Rewrite the link.
                # We use the *original* absolute link (with potential fragment) to generate the filename
                # and the fragment for the new link.
                
                # Generate the filename for the target page
                new_filename = url_to_filename(absolute_link_clean)
                
                # Preserve any fragment identifier (e.g., #section-one)
                url_fragment = urlparse(absolute_link).fragment
                new_href = f"./{new_filename}"
                if url_fragment:
                    new_href += f"#{url_fragment}"
                
                a_tag['href'] = new_href

        content_md = md(str(body), heading_style="ATX", bullets="*")
        final_markdown = f"# {title}\n\n{content_md}"

        filename = url_to_filename(current_url.rstrip('/'))
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_markdown)
            print(f"   - Saved {current_url} -> {filepath}")
        except OSError as e:
            print(f"   [!] Error saving file {filepath}: {e}")

def main():
    """Main function to parse arguments and run the scraper."""
    parser = argparse.ArgumentParser(
        description="A LangChain-powered web scraper that saves each page as a separate, inter-linked Markdown file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("url", help="The starting URL to scrape (e.g., 'example.com').")
    parser.add_argument(
        "-d", "--depth", 
        type=int, 
        default=1, 
        help="The profundity (depth) of the crawl. Default is 1."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Directory to save the Markdown files. Default is 'output'."
    )

    args = parser.parse_args()

    # Ensure the URL has a scheme (e.g., https://) for the loader
    parsed_url = urlparse(args.url)
    if not parsed_url.scheme:
        start_url = 'https://' + args.url
    else:
        start_url = args.url

    # --- Step 1: Crawl and collect raw HTML ---
    print(f"[*] Starting crawl on {start_url} with max depth {args.depth}...")
    try:
        loader = RecursiveUrlLoader(
            url=start_url,
            max_depth=args.depth,
            extractor=lambda x: x,  # Extract raw HTML content
            prevent_outside=True,
            use_async=True,
            timeout=10,
            check_response_status=True
        )
        documents = loader.load()
        print(f"[+] Crawl complete. Found {len(documents)} pages.")
    except Exception as e:
        print(f"[!] An error occurred during crawling: {e}")
        documents = []

    # --- Step 2: Process, relink, and save all collected documents ---
    process_and_save_docs(documents, args.output)
    
    print("\n[+] Scraping process finished.")

if __name__ == "__main__":
    main()