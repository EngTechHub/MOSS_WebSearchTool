"""Simple command-line example for Custom Search.

Command-line application that does a search.
"""

__author__ = "xtzhang"
import os
import re
import unicodedata as ucd
import warnings
warnings.filterwarnings('ignore')

import heapq
import time
import pprint

from googleapiclient.discovery import build
import re
import requests
#from HTMLParser import HTMLParser
from html.parser import HTMLParser
from re import sub
from sys import stderr
from traceback import print_exc 
import spacy
# 必须导入pytextrank，虽然表面上没用上，
import pytextrank
import nltk
import fasttext
from bs4 import BeautifulSoup
#import fasttext.util
import json
import heapq
import re
import time
from urllib import parse

import requests
from bs4 import BeautifulSoup


from textrank_utils import top_sentence
from score_utils import score, score_2, score_3

import logging

logger = logging.getLogger()


class prey(object):
    def __init__(self, value, sentence):
        self.value =  value
        self.sentence = sentence
    # 重写 < 符号用于sorted
    def __lt__(self, other):
        return self.value < other.value
    def __gt__(self, other):
        return self.value > other.value
    def __le__(self, other):
        return self.value <= other.value
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return self.value != other.value
    def __ge__(self, other):
        return self.value >= other.value

def containenglish(str0):
    import re
    return bool(re.search('[a-z]', str0))


def clean_html(html: str) -> str:
    """Remove HTML markup from the given string."""
    # Remove inline JavaScript/CSS, HTML comments, and HTML tags
    cleaned_html = re.sub(
        r"(?is)<(script|style).*?>.*?(</\1>)|<!--(.*?)-->[\n]?|<(?s).*?>", "", html.strip()
    )

    # Deal with whitespace and HTML entities
    cleaned_html = re.sub(
        r"&nbsp;|  |\t|&.*?;[0-9]*&.*?;|&.*?;", "", cleaned_html
    )

    # Normalize the text
    # cleaned_html = ucd.normalize('NFKC', cleaned_html).replace(' ', '')

    return cleaned_html.strip()

def select(new):
    if len(new) < 10:
        oral = new
    elif len(new) // 10 < 10:
        oral = new[:20]
    elif len(new) // 10 > 50:
        oral = new[:50]
    else:
        oral = new[:len(new) // 10] 
    return oral

def get_web_response(url):
    logger.info(f"[ENGINE] try to get web response from url={url}")
    try:
        response = requests.get(url=url, timeout=5)
        response.encoding = 'utf-8'
        return response
    except requests.exceptions.RequestException as e:
        logger.info(f"requests post fail, url={url}, exception is {e}")
        return None

def extract_description(soup):
    description = soup.find(attrs={"name": "description"})
    if description:
        content = description.get('content')
        if content:
            return content
    return None

def summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh, measure_en, measure_zh, snippet,title):
    logger.info(f"summ_web starts for url={url} query={q}")
    #start_time = time.time()
    url = parse.unquote(url)
    
    response = get_web_response(url)
    if response is None:
        return {"title":title, "url": url, "summ": snippet, "note": "fail to get ... use snippet", "type": "snippet"}

    soup = BeautifulSoup(response.text, "html.parser")
    description = extract_description(soup)

    if description:
        if all(key_word in description for key_word in q.split()):
            return {"title":title, "url": url, "summ": description, "note": "use description as summ", "type": "description"}

    text = clean_html(response.text)
    sentences = re.split("\n|。|\.", text)

    ft = ft_en if is_eng else ft_zh
    measure = measure_en if is_eng else measure_zh
    nlp = nlp_en if is_eng else nlp_zh

    scored_sentences = []
    for sentence in sentences:
        if 3 <= len(sentence) <= 200:
            scored_sentence = {
                'ft': -1 * score(q, sentence, ft) if ft else None,
                'score_2': -1 * score_2(q, sentence),
                'measure': -1 * score_3(q, sentence, measure=measure) if measure else None,
                'sentence': sentence
            }
            scored_sentences.append(scored_sentence)

    top_sentences = heapq.nsmallest(5, scored_sentences, key=lambda x: x['ft'] or float('inf')) + \
                    heapq.nsmallest(10, scored_sentences, key=lambda x: x['score_2']) + \
                    heapq.nsmallest(5, scored_sentences, key=lambda x: x['measure'] or float('inf'))

    stop_word = "." if is_eng else "。"
    combined_text = stop_word.join([sentence['sentence'] for sentence in top_sentences])

    if len(combined_text) < 3:
        return {"title":title, "url": url, "summ": snippet, "note": "bad web, fail to summ, use snippet,", "type": "snippet"}

    try:
        summary = top_sentence(text=combined_text, limit=3, nlp=nlp)
        summary = "".join(summary)
    except Exception as e:
        return {"title":title, "url": url, "summ": snippet, "note": "unknown summ error , use snippet", "type": "snippet"}

    if any(key_word in summary for key_word in q.split()):
        return {"title":title, "url": url, "summ": summary, "note": "good summ and use it", "type": "my_summ"}

    return {"title":title, "url": url, "summ": snippet, "note": "poor summ , use snippet", "type": "snippet"}
    
def search_api(q, SERPER_KEY):
    import requests
    import json
    url = "https://google.serper.dev/search"

    if containenglish(q): 
        payload = json.dumps({"q": q,})
    else:
        payload = json.dumps({"q": q})#,"gl": "cn","hl": "zh-cn"})
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    response_dict = json.loads(response.text)

    return response_dict

def bing_search_api(q, bing_subscription_key, mkt="zh-CN"):
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    params = { "q": q, "mkt": mkt, "count": "20"}
    headers = { "Ocp-Apim-Subscription-Key": bing_subscription_key }

    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def filter_urls(urls, snippets, titles, black_list=None, topk=3):
    if black_list is None:
        black_list = ["enoN, youtube.com, bilibili.com", "zhihu.com"]

    filtered_urls, filtered_snippets, filtered_titles = [], [], []
    count = 0
    for url, snippet, title in zip(urls, snippets, titles):
        if all(domain not in url for domain in black_list) and url.split(".")[-1] != "pdf":
            filtered_urls.append(url)
            filtered_snippets.append(snippet)
            filtered_titles.append(title)
            count += 1
            if count >= topk:
                break

    return filtered_urls, filtered_snippets, filtered_titles

def engine(q, SERPER_KEY,ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh, topk=3):
    start_time = time.time()
    is_eng = containenglish(q)

    response = search_api(q, SERPER_KEY)

    if "answerBox" in response.keys():
        url = response["answerBox"].get("link", response["organic"][0]["link"])
        answerBox = response["answerBox"]
        summ = ""
        title = q
        if "title" in answerBox:
            title = answerBox["title"]

        if "answer" in answerBox:
            summ = answerBox["answer"]

        if "snippet" in answerBox:
            summ = answerBox["snippet"]

        print("[EnGINE] answerBox")
        print("[ENGINE] query cost:", time.time() - start_time)
        return {"0": {"url": url, "title": title, "summ": summ, "note": "directly return answerBox, thx google !", "type": "answerBox", "origin": answerBox}}

    raw_urls = [i["link"] for i in response["organic"]]
    raw_snippets = [i["snippet"] for i in response["organic"]]
    raw_titles = [i["title"] for i in response["organic"]]
    urls, snippets, titles = filter_urls(raw_urls, raw_snippets, raw_titles, topk=topk)

    results = {}
    for i, url in enumerate(urls):
        try:
            summ = summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh, measure_en, measure_zh, snippets[i], titles[i])
        except:
            summ = {"url": url, "summ": snippets[i], "note": "unbelievable error, use snippet !", "type": "snippet", "title":titles[i]}

        results[str(i)] = summ

    print("[ENGINE] query cost:", time.time() - start_time)
    return results   

def bing_engine(q: str, bing_subscription_key: str, ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh, topk=3):
    start_time = time.time()
    is_eng = containenglish(q)

    response = bing_search_api(q, bing_subscription_key)
    webpages = {w["id"]: w for w in response["webPages"]["value"]}
    raw_urls = []
    raw_snippets = []
    raw_titles = []
    for i, item in enumerate(response["rankingResponse"]["mainline"]["items"]):
        if item["answerType"] == "WebPages":
            webpage = webpages[item["value"]["id"]]
            if webpage is not None:
                raw_urls.append(webpage["url"])
                raw_snippets.append(webpage["snippet"])
                raw_titles.append(webpage["name"])

        if item["answerType"] == "News":
            if item["value"]["id"] == response["news"]["id"]:
                logger.info(f"got news at index {i} with length {len(response['news']['value'])} for query={q}")
                for n in response["news"]["value"]:
                    raw_urls.append(n["url"])
                    raw_snippets.append(n["description"])
                    raw_titles.append(n["name"])

    urls, snippets, titles = filter_urls(raw_urls, raw_snippets, raw_titles, topk=topk)

    results = {}
    for i, url in enumerate(urls):
        try:
            summ = summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh, measure_en, measure_zh, snippets[i], titles[i])
        except:
            summ = {"url": url, "summ": snippets[i], "note": "unbelievable error, use snippet !", "type": "snippet", "title":titles[i]}

        results[str(i)] = summ

    logger.info("[ENGINE] cost {}s for query={}".format(time.time() - start_time, q))
    return results


if __name__ == "__main__":
    import time
    print("loading embeddings ...")
    ft_en = fasttext.load_model('cc.en.300.bin')
    ft_zh = fasttext.load_model('cc.zh.300.bin')
    nlp_en = spacy.load("en_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")
    from score_utils import score_measure
    measure_en = None#score_measure("en")
    measure_zh = None#score_measure("zh")
    print("embeddings loaded ...")

    start_time = time.time()
    #engine("复旦 排名")#yes
    #engine("张文宏")#yes
    #engine("relative molecular mass of carbon dioxide", measure_en, measure_zh)#yes
    #engine("爱因斯坦场方程 解的数目")#yes
    #engine("Stable Diffusion introduction", measure_en, measure_zh)#yes
    #engine("quick sort", measure_en, measure_zh)#yes
    #engine("document image rectification", ft_en, ft_zh, measure_en, measure_zh)#yes
    #engine("忽如一夜春风来，千树万树梨花开 季节", ft_en, ft_zh, measure_en, measure_zh)#no
    print(bing_engine("奔驰c 比亚迪model y 比较", os.getenv("BING_SUB_KEY"), ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh))
    print(bing_engine("诺贝尔奖", os.getenv("BING_SUB_KEY"), ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh))
    print(bing_engine("亚运会", os.getenv("BING_SUB_KEY"), ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh))
    print(bing_engine("巴以冲突", os.getenv("BING_SUB_KEY"), ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh))
    print(time.time() - start_time)

