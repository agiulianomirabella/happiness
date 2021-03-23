import json

def pretty_dict(d):
    print(json.dumps(d, sort_keys=True, indent=4))

