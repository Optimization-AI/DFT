import json
from datasets import Dataset
import argparse
import os
BASE_DIR = './data'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--push_to_hub', action='store_true')
    args = parser.parse_args()

    files = [filename for filename in os.listdir(BASE_DIR) if filename.startswith(args.dataset)]
    if len(files) == 0:
        raise ValueError(f"Can not find any matched dataset with prefix {args.dataset}")

    results = {}
    results["real"] = []
    for i in range(len(files)):
        results[f"generated_{i}"] = []

    for i, file in enumerate(files):
        f = open(os.path.join(BASE_DIR, file), "r")
        js = json.load(f)
        for j in range(len(js)):
            results[f"generated_{i}"].append(
                [{"content" : js[j]["prompt"], "role" : "user"},
                {"content" : js[j]["generated"], "role" : "assistant"}]
            )
            if i == 0:
                results[f"real"].append(
                    [{"content" : js[j]["prompt"], "role" : "user"},
                    {"content" : js[j]["real"], "role" : "assistant"}]
                )
            
    with open(os.path.join(BASE_DIR, f"merged_{args.dataset}.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    if args.push_to_hub:
        ds = Dataset.from_dict(results)
        ds.push_to_hub(args.dataset)