import json

class Destructure:
    def __init__(self):
        self.key_prefs = self.get_key_prefs()
    
    def get_key_prefs(self):
        with open('key_preferences.json', 'r') as f:
            data = json.load(f)
        return data

    def clean_key(self, k):
        if '_' in k:
            return k.replace('_', ' ')
        else:
            return k

    def get_text_and_metadata(self, doc):
        text_payload = []
        skip_keys = ['l1', 'l2', 'l3', 'l4']
        for key in doc:
            if key in self.key_prefs['indexed_cols']:
                if key in skip_keys:
                    text_payload.append(("", doc[key]))
                    continue
                text_payload.append((self.clean_key(key), doc[key]))

        total_text = []
        for x, y in text_payload:
            if x == '':
                total_text.append(y)
            else:
                total_text.append(f'{x}->{y}'.replace('\n',' '))

        total_text = '|'.join(total_text).strip()
        del text_payload
        metadata_available = [k for k in doc.keys() if k in self.key_prefs['shared_id']+self.key_prefs['indexed_cols'] and k not in self.key_prefs['ignored_cols']]
        metadata_payload = {k: doc[k] for k in metadata_available}
        return total_text, metadata_payload
