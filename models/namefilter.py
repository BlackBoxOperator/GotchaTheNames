import re, os

familyNameFile = os.path.join('..', 'data', 'familyName.txt')

def exist_chinese(s):
    return len(re.findall(r'[\u4e00-\u9fff]+', s))

def all_chinese(s):
    n = ''.join(s.split())
    m = re.findall(r'[\u4e00-\u9fff]+', n)
    return m and len(m[0]) == len(n)

def exist_english(s):
    return any([c.encode('utf-8').isalpha() for c in s])

def all_english(s, exc = ''):
    n = ''.join(s.split())
    return all([c.encode('utf-8').isalpha() or c in exc for c in n])

def namefilter():

    fname = [n for r in open(familyNameFile).read().split("\n") for n in r.split()]

    def ft(namelist, doc, useFname = True):

        namelist = [n.replace('\u3000', '').strip() for n in namelist if not n.startswith('APP')]

        # remove space in chinese name, and unit name
        namelist = [n if exist_english(n) else ''.join(n.split()) for n in namelist if len(n) > 1]


        # keep one lang only (need think the familyName part)
        namelist = [name for name in namelist
                if all_english(name) or (all_chinese(name) and ('．' not in name) and name[0] in fname)]

        # remove pronoun
        namelist = [name for name in namelist \
                if not (len(name) in [2,4] \
                and name[1] in '男女某哥姐父母娘爸董嫌夫妻婦家氏姓檢')]

        # uniq the set
        namelist = list(set(namelist))

        # 美濃區東門羅文俊

        dominators = {}
        subsim = sorted(namelist, key=len, reverse=True)

        # make dominators
        for n in subsim:
            for k in dominators:
                if n in k:
                    dominators[k].append(n)
                    break
            else: dominators[n] = [n]


        namelist = [max([(doc.count(v) * (4**i), v) for i, v in enumerate(reversed(lst))])[1]
                for lst in dominators.values()]

        # remove dup name
        """
        namelist = list(reversed([this for idx, this in enumerate(namelist)
                        if not any(this in other and len(this) != 3 for other in namelist[idx + 1:])]))

        namelist = [this for idx, this in enumerate(namelist)
                        if not any(this in other and len(this) != 3 for other in namelist[idx + 1:])]
        """

        # remove app
        #namelist = [name for name in namelist if not name.startswith('APP')]

        # name tfidf/w2v
        return namelist

    return ft
