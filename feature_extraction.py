import re
import math
import urllib.parse
from urllib.parse import urlparse
import tldextract
import ipaddress


def extract_features(url):
    """
    Extract features from a URL for phishing detection.
    Returns a list of feature values.
    """
    features = []
    
    # Parse URL
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    # Extract domain components
    ext = tldextract.extract(url)
    domain_name = ext.domain
    suffix = ext.suffix
    
    # 1. URL Length
    features.append(len(url))
    
    # 2. Domain Length
    features.append(len(domain) if domain else 0)
    
    # 3. Path Length
    features.append(len(path))
    
    # 4. Query Length
    features.append(len(query))
    
    # 5. Number of dots in domain
    features.append(domain.count('.') if domain else 0)
    
    # 6. Number of hyphens in domain
    features.append(domain.count('-') if domain else 0)
    
    # 7. Number of underscores in domain
    features.append(domain.count('_') if domain else 0)
    
    # 8. Number of slashes in URL
    features.append(url.count('/'))
    
    # 9. Number of question marks
    features.append(url.count('?'))
    
    # 10. Number of equals signs
    features.append(url.count('='))
    
    # 11. Number of ampersands
    features.append(url.count('&'))
    
    # 12. Number of percent signs
    features.append(url.count('%'))
    
    # 13. Has IP address in domain
    try:
        ipaddress.ip_address(domain.split(':')[0])
        features.append(1)
    except:
        features.append(0)
    
    # 14. Has HTTPS
    features.append(1 if parsed.scheme == 'https' else 0)
    
    # 15. Number of subdomains
    subdomain_parts = ext.subdomain.split('.') if ext.subdomain else []
    features.append(len([s for s in subdomain_parts if s]))
    
    # 16. Domain name length
    features.append(len(domain_name))
    
    # 17. TLD length
    features.append(len(suffix))
    
    # 18. Has suspicious keywords
    suspicious_keywords = ['secure', 'account', 'update', 'verify', 'bank', 'login', 'signin', 
                          'confirm', 'click', 'here', 'www', 'webscr', 'paypal', 'ebay']
    keyword_count = sum(1 for keyword in suspicious_keywords if keyword.lower() in url.lower())
    features.append(keyword_count)
    
    # 19. Has @ symbol (suspicious)
    features.append(1 if '@' in url else 0)
    
    # 20. Has double slash in path
    features.append(1 if '//' in path else 0)
    
    # 21. Has port number
    features.append(1 if ':' in domain and not domain.startswith('[') else 0)
    
    # 22. Number of digits in domain
    features.append(sum(1 for c in domain if c.isdigit()) if domain else 0)
    
    # 23. Number of special characters
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/', '~', '`']
    features.append(sum(1 for c in url if c in special_chars))
    
    # 24. URL entropy (measure of randomness)
    if len(url) > 0:
        entropy = -sum((url.count(c) / len(url)) * math.log2(url.count(c) / len(url)) 
                      for c in set(url) if url.count(c) > 0)
        features.append(entropy)
    else:
        features.append(0)
    
    # 25. Has shortening service
    shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 
                          'short.link', 'rebrand.ly', 'cutt.ly']
    features.append(1 if any(service in url.lower() for service in shortening_services) else 0)
    
    return features


def get_feature_names():
    """
    Return list of feature names for reference.
    """
    return [
        'url_length', 'domain_length', 'path_length', 'query_length',
        'dots_in_domain', 'hyphens_in_domain', 'underscores_in_domain',
        'slashes', 'question_marks', 'equals', 'ampersands', 'percent_signs',
        'has_ip', 'has_https', 'subdomain_count', 'domain_name_length',
        'tld_length', 'suspicious_keywords', 'has_at_symbol', 'double_slash',
        'has_port', 'digits_in_domain', 'special_chars', 'entropy',
        'has_shortening_service'
    ]
