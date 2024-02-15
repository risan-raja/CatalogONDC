import re

class SparseDocTextPreprocessor:
    def __init__(self):
        self.regex_cleaner_basic = re.compile(r'\bselling price\b|\bmrp\b|\d+|\b\w\b', re.IGNORECASE)
        self.regex_remove_punct = re.compile(r'[^\w\s]', re.IGNORECASE)
        self.regex_remove_extra_spaces = re.compile(r'\s{2,}', re.IGNORECASE)
    
    def clean_text(self, doc):
        clean_doc = self.regex_remove_punct.sub(' ', doc)
        clean_doc = self.regex_cleaner_basic.sub('', clean_doc)
        clean_doc = self.regex_remove_extra_spaces.sub(' ', clean_doc)
        return clean_doc
