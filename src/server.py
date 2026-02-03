from fastapi import FastAPI
from typing import List
import numpy as np

app = FastAPI()

# Placeholder for demonstration
known_ids = []
known_attributes = []

def match_attributes(query_attrs, threshold=0.5):
    matches = []
    for i, attr in enumerate(known_attributes):
        diff = np.abs(attr - query_attrs)
        if all(diff < threshold):
            matches.append(known_ids[i])
    return matches

@app.get("/query")
def query_person(attributes: List[float]):
    attributes = np.array(attributes)
    matched_ids = match_attributes(attributes)
    return {"matched_ids": matched_ids}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
