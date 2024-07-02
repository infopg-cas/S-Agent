import requests
from bs4 import BeautifulSoup

LOOKUP_KEYWORD = None  # current lookup keyword
LOOKUP_LIST = None  # list of paragraphs containing current lookup keyword
LOOKUP_CNT = None  # current lookup index
PAGE_INFO = ""


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
        sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])


def search_wiki(entity: str, **kwargs):
    global PAGE_INFO, LOOKUP_KEYWORD, LOOKUP_LIST, LOOKUP_CNT
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(search_url).text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:
        result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        return True, f"Could not find {entity}. Similar: {result_titles[:5]}."
    else:
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):
            search_wiki(entity)
        else:
            PAGE_INFO = ""
            for p in page:
                if len(p.split(" ")) > 2:
                    PAGE_INFO += clean_str(p)
                    if not p.endswith("\n"):
                        PAGE_INFO += "\n"
            obs = get_page_obs(PAGE_INFO)
            LOOKUP_KEYWORD = LOOKUP_LIST = LOOKUP_CNT = None
            return True, obs
            # return True, f"Find the page that contains the entity '{entity}' that store in local now, there are multiple sentences. Try to use loop up tool to find keyword that you want in the page."


def lookup(keyword, **kwargs):
    global LOOKUP_CNT, LOOKUP_LIST, LOOKUP_KEYWORD
    if LOOKUP_KEYWORD != keyword:
        LOOKUP_KEYWORD = keyword
        LOOKUP_LIST = construct_lookup_list(keyword)
        LOOKUP_CNT = 0
    if LOOKUP_CNT >= len(LOOKUP_LIST):
        obs = "No more results.\n"
    else:
        obs = f"(Result {LOOKUP_CNT + 1} / {len(LOOKUP_LIST)}) " + LOOKUP_LIST[LOOKUP_CNT]
        LOOKUP_CNT += 1
    return True, obs


def construct_lookup_list(keyword):
    global PAGE_INFO
    if PAGE_INFO is None:
        return []
    # paragraphs = PAGE_INFO.split("\n")
    # paragraphs = [p.strip() for p in paragraphs if p.strip()]
    #
    # # find all sentence
    # sentences = []
    # for p in paragraphs:
    #     sentences += p.split('. ')
    #     sentences = [s.strip() + '.' for s in sentences if s.strip()]
    #
    # parts = sentences
    # parts = [p for p in parts if keyword.lower() in p.lower()]

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(PAGE_INFO)
    parts = [sentence for sentence in sentences if keyword.lower() in sentence.lower()]
    return parts
