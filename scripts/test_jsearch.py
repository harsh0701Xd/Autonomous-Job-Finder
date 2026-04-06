"""
scripts/test_jsearch.py
Quick test to verify JSearch API key is working.
"""
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("JSEARCH_API_KEY")
if not key:
    print("ERROR: JSEARCH_API_KEY not found in .env file")
    exit(1)

print(f"Key loaded: {key[:8]}...")

try:
    r = httpx.get(
        "https://jsearch.p.rapidapi.com/search",
        headers={
            "X-RapidAPI-Key":  key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        },
        params={
            "query":     "Data Scientist Delhi NCR",
            "num_pages": "1",
        },
        timeout=15.0,
    )

    print(f"Status: {r.status_code}")
    data  = r.json()
    jobs  = data.get("data", [])
    print(f"Jobs returned: {len(jobs)}")

    if jobs:
        first = jobs[0]
        print(f"First job : {first.get('job_title')} at {first.get('employer_name')}")
        print(f"Location  : {first.get('job_city')}, {first.get('job_country')}")
        print(f"Apply URL : {first.get('job_apply_link', 'N/A')[:60]}...")
    else:
        print("No jobs returned — check your API key or query")

except Exception as e:
    print(f"ERROR: {e}")