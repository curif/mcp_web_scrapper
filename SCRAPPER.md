
# LangChain Web Scraper to Markdown

This script crawls a website up to a specified depth and converts each page into a clean, self-contained Markdown file. It intelligently rewrites internal links to point to the corresponding local Markdown files, creating a fully browsable offline version of the site's content.

It is designed to extract meaningful content by removing common non-essential elements like navigation bars, headers, footers, and images.

## Key Features

*   **Depth-Controlled Crawling**: Specify how many levels of links to follow from the start URL (`--depth`).
*   **Host-Locked Scraping**: The scraper will only follow links that belong to the same host as the initial URL.
*   **HTML to Markdown Conversion**: Each page's main content is converted into clean, readable Markdown.
*   **Automatic Internal Link Replacement**: Links pointing to other scraped pages are automatically rewritten to point to the local `.md` files, creating a navigable offline archive.
*   **Content Cleaning**: Automatically removes common clutter like `<nav>`, `<header>`, `<footer>`, `<aside>`, `<img>`, `<svg>`, and `<script>` tags before conversion.
*   **Customizable Output**: Specify a directory where all the generated Markdown files will be saved (`--output`).

## Setup and Installation

1.  **Clone or Download**: Get the script and the `requirements.txt` file into a local directory.

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**: Install the required Python packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

Run the script from your terminal, providing the start URL and any desired options.

**Command Structure:**
```bash
python3 scraper.py [URL] [OPTIONS]
```

### Arguments

| Argument | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `url` | | The starting URL to scrape (e.g., `example.com`). | (Required) |
| `--depth` | `-d` | The profundity of the crawl. `0` scrapes only the start page. `1` scrapes the start page and pages it links to. | `1` |
| `--output`| `-o` | The directory where the Markdown files will be saved. | `output` |

### Examples

**1. Scrape a Single Page**

This will scrape only the homepage of `toscrape.com` and save it as `scraped_content/index.md`.

```bash
python3 scraper.py toscrape.com --depth 0 --output scraped_content
```

**2. Scrape a Website One Level Deep**

This will scrape the homepage of `quotes.toscrape.com` and all the pages it links to. The resulting `.md` files will be saved in a directory named `my_quotes`.

```bash
python3 scraper.py quotes.toscrape.com --depth 1 --output my_quotes
```

** real cases **
```bash
python3 scrapper.py --output ./output --dept 5 https://google.github.io/adk-docs/
```

## How It Works: The Magic of Link Replacement

The script follows a two-step process to ensure that internal links work correctly:

1.  **Crawl First**: It crawls the entire target site up to the specified depth, collecting the raw HTML of every valid page. This creates a master list of all scraped URLs.
2.  **Process and Relink Later**: After the crawl is complete, it processes each page's HTML. For every link (`<a>` tag) found, it checks if the link's destination URL is in the master list of scraped URLs.
    *   If it's an **internal link** to another scraped page, the `href` is rewritten to point to the local file (e.g., `/about-us/` becomes `./about-us.md`).
    *   If it's an **external link**, it is left untouched.

This method ensures a fully inter-linked and navigable local documentation set.

## Example Output

If you run `python3 scraper.py quotes.toscrape.com -d 1 -o scraped_quotes`, your output directory will look like this:

**File Structure:**
```
scraped_quotes/
├── index.md
├── login.md
├── page_2.md
├── tag_books.md
├── tag_humor.md
└── ...and so on for every page found.
```

**File Content (`index.md`):**

A link that originally pointed to `/page/2/` on the website will be converted into a relative link in the Markdown file.

**Original HTML:**
```html
<li class="next">
    <a href="/page/2/">Next →</a>
</li>
```

**Resulting Markdown in `index.md`:**
```markdown
* [Next →](./page_2.md)
```

---

## Requirements File (`requirements.txt`)

For reference, this is the content of the `requirements.txt` file.

```text
# Core LangChain framework and community packages
langchain
langchain-community

# For parsing HTML content
beautifulsoup4
lxml

# For converting cleaned HTML to Markdown
markdownify
```