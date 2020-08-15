import webbrowser, re, csv, os, sys
import requests as rq
from bs4 import BeautifulSoup
from tqdm import tqdm
from functools import reduce
from itertools import takewhile

resc = lambda s: s.replace("\r", '').replace("", "").replace("\n", "")

nonsense = ['(中時電子報)',
            '想看更多新聞嗎',
            '＞＞＞更多新聞請點此＜']
nonwords = ['小', '中', '大']

fetch_table = {
        # previous
        'www.chinatimes.com':          ['div', {'class': 'article-body'}],
        #'news.ltn.com.tw':             ['div', {'class': 'whitecon articlebody'}],
        'news.tvbs.com.tw':            ['div', {'id':'news_detail_div'}],
        'home.appledaily.com.tw':      ['div', {'class': 'ncbox_cont'}],

        # current
        'news.cnyes.com':              ['div', {'itemprop': 'articleBody'}],
        'www.mirrormedia.mg':          ['article', {}],
        'domestic.judicial.gov.tw':    ['pre', {}],
        'www.coolloud.org.tw':         ['div', {'class':'field-items'}],
        'm.ctee.com.tw':               ['div', {'class': 'entry-main'}],
        'mops.twse.com.tw':            ['div', {'id': 'zoom'}],
        'www.hk01.com':                ['article', {}],
        'www.wealth.com.tw':           ['div', {'class': 'entry-main'}],
        'news.ebc.net.tw':             ['div', {'class': 'fncnews-content'}],
        'news.mingpao.com':            ['article', {}],
        'www.bnext.com.tw':            ['div', {'class': 'content'}],
        'news.ltn.com.tw':             ['div', {'itemprop': 'articleBody'}],
        'finance.technews.tw':         ['article', {}],
        'www.fsc.gov.tw':              ['div', {'id': 'maincontent'}],
        #'news.tvbs.com.tw':           prev
        'www.cw.com.tw':               ['article', {}],
        'www.businesstoday.com.tw':    ['div', {'class': 'article'}],
        'sina.com.hk':                 ['section', {'id': 'content'}],
        'www.ettoday.net':             ['article', {}],
        'hk.on.cc':                    ['div', {'class': 'breakingNewsContent'}],
        'technews.tw':                 ['div', {'class': 'content'}],
        'money.udn.com':               ['div', {'id': 'article_body'}],
        'udn.com':                     ['div', {'class': 'article-content__paragraph'}], # must latter, becuase money.udn
        'tw.news.yahoo.com':           ['article', {}],
        'www.setn.com':                ['article', {}],
        'www.managertoday.com.tw':     ['body', {}],
        'www.cna.com.tw':              ['article', {}],
        'estate.ltn.com.tw':           ['div', {'itemprop': 'articleBody'}],
        'm.ltn.com.tw':                ['div', {'itemprop': 'articleBody'}],
        'ccc.technews.tw':             ['article', {}],
        'www.hbrtaiwan.com':           ['div', {'class': 'article'}],
        'ec.ltn.com.tw':               ['p', {}],#err
        # https://ec.ltn.com.tw/article/paper/1277631
        'www.nownews.com':             ['div', {'class': 'newsContainer'}],
        'ol.mingpao.com':              ['div', {'class': 'article_wrap'}],
        'tw.nextmgz.com':              ['article', {}],
        'www.nextmag.com.tw':          ['article', {}],
        'ent.ltn.com.tw':              ['div', {'class': 'text'}],
        'www.storm.mg':                ['article', {}],
        'house.ettoday.net':           ['article', {}],
        }

encoding = {
            'www.coolloud.org.tw': 'utf-8',
            'news.mingpao.com': 'utf-8',
            'hk.on.cc': 'utf-8',
        }

def rm_tag(s):
    return re.sub('<[^>]*>', '', s)

def download(url):
    downcount = 0
    while True:
        try:
            request = rq.get(url, timeout = 10)
            for domain in encoding:
                if domain in url:
                    request.encoding = encoding[domain]
            if request.status_code != 200:
                return None, "- status: {} -".format(request.status_code)
        except rq.exceptions.Timeout:
            downcount += 1
            if downcount > 5:
                return None, "- timeout too many times -"
            continue
        except Exception as e:
            return None, "- exception: {} -".format(e)

        return request.text, None


def rename_dialog(ques, alert, default):
    print()
    name = input(ques + '?\n(enter to cont, input to rename):').strip()
    if name:
        print()
        yn = input(alert.format(name) + '?\n(enter to cont, input to break):').strip()
        if yn:
            print("terminated")
            exit(0)
        return name
    else:
        return default

def handle_uee(filename):
    return repr(filename)[1:-1]

def writerow(writer, row, log):
    try:
        writer.writerow(row)
    except UnicodeEncodeError:
        index, _, _ = row
        log(index, "- Encode Error -")
        writer.writerow([handle_uee(r) for r in row])

def find_article_args_by(url):
    for domain in fetch_table:
        if domain in url:
            tag, attr = fetch_table[domain]
            return { 'name': tag, 'attrs': attr }
    print("cannot find domain pattern in", url)
    webbrowser.open(url), exit(0)

mode = 'a'

def fetch_news(index, url, record, log):
    if url.startswith('http'):
        html, errmsg = download(url)
    else:
        html = open(url).read()
        errmsg = 0
    if errmsg:
        log(index, errmsg, url)
        record([index, '', ''])
        return

    soup = BeautifulSoup(html, "html.parser")
    title = resc(soup.title.get_text() if soup.title else '')

    #article = soup.find(**find_article_args_by(url))
    articles = soup.findAll(**find_article_args_by(url))

    if not articles:
        log(index, '- cannot fetch text from article -', url, find_article_args_by(url))
        record([index, title, ''])
        return

    paragraphs = reduce(lambda a, b: a + b, [a.findChildren("p") for a in articles])

    for a in articles[1:]:
        paragraphs.extend(a.findChildren("p"))

    if 'ec.ltn.com.tw' in url:
        p = [s for article in articles for s in article.get_text().split() \
                            if not any([s.startswith(n) for n in nonsense])
                            and not any([s == n for n in nonwords])]
        p = list(takewhile(lambda x: x not in ['一手掌握經濟脈動', '點我訂閱自由財經Youtube頻道', '不用抽', '不用搶', '現在用APP看新聞', '保證天天中獎', '點我下載APP', '按我看活動辦法'], p))
        content = resc(' '.join(p))
        record([index, title, rm_tag(content)])

    elif not paragraphs or (('tvbs' in url or 'www.chinatimes.com' in url) and len(paragraphs) < 5):
        content = resc(' '.join([s for article in articles for s in article.get_text().split() \
                            if not any([s.startswith(n) for n in nonsense])
                            and not any([s == n for n in nonwords])]))
        record([index, title, rm_tag(content)])
    else:
        paragraphs += reduce(lambda a, b: a + b, [a.find_all(re.compile('^h[1-6]$')) for a in articles])
        content = resc(' '.join([s for s in [p.get_text().strip() for p in paragraphs] \
                                    if not any([s.startswith(n) for n in nonsense])
                                    and not any([s == n for n in nonwords])]))
        record([index, title, content])

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    for fn in sys.argv[1:]:
        print("start crawling {}...".format(fn))
        if not fn.endswith('.csv'):
            fetch_news(0, fn, print, eprint)
        else:

            if input("Enter to continue else break:"):
                print("terminated"), exit(0)

            log_loc = os.path.join('..', '..', 'data', 'log.txt')
            cont_loc = os.path.join('..', '..', 'data', 'content.csv')
            log_loc = rename_dialog("save log to: {}".format(log_loc),
                                    "relocate log file to {}",
                                    log_loc)
            cont_loc = rename_dialog("save content to: {}".format(cont_loc),
                                     "relocate content file to {}",
                                     cont_loc)

            ctx = open(os.path.join(fn), 'r')
            csvr = csv.reader(ctx); next(csvr, None)
            rows = list(csvr)
            with open(log_loc, mode, encoding="UTF-8") as logfile:
                log = lambda *ss: logfile.write(' '.join([str(s) for s in ss]) + '\n')
                with open(cont_loc, mode, newline='', encoding="UTF-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writerow(writer, ['index', 'title', 'content'], log)
                    for index, url, *_ in tqdm(rows, ascii = True):
                        fetch_news(index, url, lambda *a: writerow(writer, *a, log), log)
