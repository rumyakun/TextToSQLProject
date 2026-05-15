
import sys
from pathlib import Path

# Add the current directory to sys.path to allow importing from .
sys.path.append(str(Path.cwd()))

from backend.keyword_extract import preprocess_query
from backend.keyword_references import load_reference_values
import json

def test():
    query = "컴융부 2학년 전공핵심과목 보여줘"
    print(f"Testing query: {query}")
    
    references = load_reference_values()
    for label, values in references.items():
        print(f"Reference count for {label}: {len(values)}")
        if values:
            print(f"  Example values: {values[:3]}")

    result = preprocess_query(query)
    # Use ensure_ascii=True to avoid encoding issues in console output
    print(json.dumps(result, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    test()
