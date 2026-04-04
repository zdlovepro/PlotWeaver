import requests
from bs4 import BeautifulSoup

url = "https://steamdt.com/hanging"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def fetch_page(url: str) -> str:
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding  # 自动检测编码
    return resp.text

def parse_page(html: str):
    soup = BeautifulSoup(html, "html.parser")
    # 示例：获取页面标题
    title = soup.title.string if soup.title else ""
    # 示例：获取所有链接
    links = [a["href"] for a in soup.select("a[href]")]
    return {"title": title, "links": links}

if __name__ == "__main__":
    html = fetch_page(url)
    data = parse_page(html)
    print(data)