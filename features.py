"""
features.py (final) — URL normalisation, no is_https feature.
Must match the Colab feats() exactly so the model gets identical inputs.
"""
import re, math

def _entropy(s):
    if not s: return 0.0
    fr = {}
    for c in s: fr[c] = fr.get(c, 0) + 1
    n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in fr.values())

SUSP = ['login','secure','account','update','verify','banking','confirm','password',
        'signin','paypal','ebay','amazon','support','service','free','lucky','bonus','click','webscr']
BRANDS = ['paypal','amazon','ebay','apple','microsoft','netflix','facebook',
          'google','bank','instagram','whatsapp','outlook','office365']
SHORT = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd','cutt.ly']
CT = ['.com','.org','.net','.edu','.gov']
ST = ['.xyz','.tk','.ml','.ga','.cf','.gq','.top','.click','.link','.online','.site','.work']

def _normalise(url):
    u = str(url).strip().lower()
    u = re.sub(r'^https?://', '', u)
    u = re.sub(r'^www\.', '', u)
    return u

def extract_features(url):
    norm = _normalise(url)
    parts = norm.split('/', 1)
    d = parts[0]
    pa = '/' + parts[1] if len(parts) > 1 else ''
    u = norm
    f = {}
    f['url_length']=len(u); f['domain_length']=len(d); f['path_length']=len(pa)
    f['num_dots']=u.count('.'); f['num_hyphens']=u.count('-'); f['num_underscores']=u.count('_')
    f['num_slashes']=u.count('/'); f['num_question_marks']=u.count('?'); f['num_equal_signs']=u.count('=')
    f['num_at_signs']=u.count('@'); f['num_ampersands']=u.count('&'); f['num_percent']=u.count('%')
    f['num_digits']=sum(c.isdigit() for c in u); f['num_subdomains']=max(0, d.count('.')-1)
    f['url_entropy']=round(_entropy(u),4); f['domain_entropy']=round(_entropy(d),4)
    f['digit_ratio']=round(f['num_digits']/max(len(u),1),4)
    host = d.split(':')[0]
    f['has_ip']=int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host)))
    f['has_port']=int(':' in d)
    f['is_shortened']=int(any(s in d for s in SHORT))
    f['has_suspicious_word']=int(any(w in u for w in SUSP))
    f['has_double_slash']=int('//' in pa); f['has_prefix_suffix']=int('-' in d)
    f['path_depth']=len([x for x in pa.split('/') if x])
    tld = '.'+d.split('.')[-1] if '.' in d else ''
    f['is_common_tld']=int(tld in CT); f['is_suspicious_tld']=int(tld in ST)
    f['brand_impersonation']=int(any(b in u and not d.endswith(b+'.com') for b in BRANDS))
    f['num_suspicious_words']=sum(1 for w in SUSP if w in u)
    f['digits_in_domain']=sum(c.isdigit() for c in d)
    f['long_domain']=int(len(d) > 30)
    return f

def feature_names():
    return list(extract_features("example.com/path").keys())

def analyze_ssl(url):
    https = str(url).lower().startswith('https')
    return {"scheme":"https" if https else "http",
            "verdict":"Encrypted (HTTPS)" if https else "Not encrypted (HTTP)",
            "risk":"Low" if https else "Elevated"}

def analyze_dns(url):
    norm=_normalise(url); d=norm.split('/',1)[0]; host=d.split(':')[0]
    labels=host.split('.') if host else []; tld='.'+labels[-1] if labels else ''
    is_ip=bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))
    return {"tld":tld or "(none)","num_subdomains":max(0,len(labels)-2),
            "is_ip_literal":is_ip,"risk":"High" if (is_ip or tld in ST) else "Low"}

_TLD_REGION={".uk":"United Kingdom",".us":"United States",".ru":"Russia",".cn":"China",
             ".de":"Germany",".in":"India",".tk":"Tokelau (free-registrar)",
             ".ml":"Mali (free-registrar)",".ga":"Gabon (free-registrar)"}
def analyze_geo(url):
    norm=_normalise(url); d=norm.split('/',1)[0]; labels=d.split(':')[0].split('.')
    tld='.'+labels[-1] if labels else ''; region=_TLD_REGION.get(tld,"Unknown / generic TLD")
    return {"inferred_region":region,"risk":"Elevated" if "free-registrar" in region else "Neutral"}

def analyze_tokens(url):
    u=_normalise(url); found=[w for w in SUSP if w in u]; brands=[b for b in BRANDS if b in u]
    return {"suspicious_tokens":found,"brand_terms":brands,"has_at_symbol":"@" in u,
            "has_hex_encoding":"%" in u,
            "risk":"High" if (len(found)>=2 or brands) else ("Medium" if found else "Low")}

def threat_score(url):
    f=extract_features(url); score=0; reasons=[]
    def add(p,m):
        nonlocal score; score+=p; reasons.append((p,m))
    if f['has_ip']:                  add(20,"IP address used as host")
    if f['is_suspicious_tld']:       add(15,"Suspicious top-level domain")
    if f['brand_impersonation']:     add(20,"Possible brand impersonation")
    if f['is_shortened']:            add(10,"URL shortener detected")
    if f['num_at_signs']:            add(10,"Contains @ symbol")
    if f['num_suspicious_words']>=2: add(10,"Multiple suspicious keywords")
    if f['digits_in_domain']>=2:     add(5,"Digits mixed into domain")
    if f['long_domain']:             add(5,"Unusually long domain")
    score=min(score,100)
    level="High" if score>=60 else "Medium" if score>=30 else "Low"
    return {"score":score,"level":level,"reasons":reasons}

def full_analysis(url):
    return {"features":extract_features(url),"ssl":analyze_ssl(url),
            "dns":analyze_dns(url),"geo":analyze_geo(url),
            "tokens":analyze_tokens(url),"threat":threat_score(url)}