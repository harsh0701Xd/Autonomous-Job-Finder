"""
test_scripts/test_url_validator.py

Live test for the URL Validator (core/url_validator.py).
Sends a batch of test URLs through validate_urls() and prints pass/drop
decisions with reasons. Tests cover: live job pages, dead links (404),
redirects to foreign domains (aggregators), and timeouts.

Usage:
    python test_scripts/test_url_validator.py

No API keys required. Uses real HTTP requests only.
"""

import asyncio
import sys
import uuid
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from core.state.session_state import RawJob
from core.url_validator import validate_urls


# -- Test URLs -----------------------------------------------------------------
# Mix of expected pass and expected drop cases.
TEST_CASES = [
    # (label, url, expected_outcome)
    # Live job pages -- should PASS
    ("LinkedIn Jobs (live)",   "https://www.linkedin.com/jobs/view/3900000001/",        "pass"),
    ("Naukri (live)",          "https://www.naukri.com/job-listings-data-scientist",    "pass"),
    # HTTP 404 dead link -- should DROP
    ("404 dead link",          "https://httpbin.org/status/404",                        "drop"),
    # HTTP 410 gone -- should DROP
    ("410 gone",               "https://httpbin.org/status/410",                        "drop"),
    # Redirect to different domain (aggregator pattern) -- should DROP
    ("Domain redirect (agg)", "https://httpbin.org/redirect-to?url=https://google.com", "drop"),
    # Valid company career page -- should PASS (403 = benefit of the doubt)
    ("Career page (403 ok)",   "https://careers.google.com/jobs/results/",              "pass"),
    # Timeout / unreachable -- should PASS (fail-open)
    ("RemoteOK (live)",        "https://remoteok.com/remote-jobs",                      "pass"),
    # httpbin 200 -- should PASS
    ("httpbin 200",            "https://httpbin.org/status/200",                        "pass"),
]


def _make_job(label: str, url: str) -> RawJob:
    return RawJob(
        job_id          = str(uuid.uuid4()),
        title           = label,
        company         = "TestCo",
        location        = "Remote",
        jd_text         = f"Test JD for {label}",
        apply_url       = url,
        source          = "test",
        matched_profile = "Data Scientist",
    )


async def main():
    print("\n" + "=" * 60)
    print("  URL Validator -- Live Test")
    print("=" * 60)

    jobs = [_make_job(label, url) for label, url, _ in TEST_CASES]
    expected = {_make_job(label, url).job_id: exp
                for (label, url, exp), job in zip(TEST_CASES, jobs)}
    job_map  = {job.job_id: (TEST_CASES[i][0], TEST_CASES[i][2])
                for i, job in enumerate(jobs)}

    print(f"\nRunning {len(jobs)} URL checks concurrently...\n")

    results = await validate_urls(jobs)

    print(f"{'Label':<35} {'URL Result':<10} {'Expected':<10} {'Reason'}")
    print("-" * 80)

    passed = dropped = mismatches = 0
    for job_id, is_valid, reason in results:
        label, exp = job_map[job_id]
        outcome    = "PASS" if is_valid else "DROP"
        ok         = (is_valid and exp == "pass") or (not is_valid and exp == "drop")
        flag       = "" if ok else " ← UNEXPECTED"
        if not ok:
            mismatches += 1
        if is_valid:
            passed += 1
        else:
            dropped += 1

        print(f"{label:<35} {outcome:<10} {exp.upper():<10} {reason}{flag}")

    print("-" * 80)
    print(f"\nPassed: {passed} | Dropped: {dropped} | Mismatches: {mismatches}")
    if mismatches == 0:
        print("All results matched expected outcomes.")
    else:
        print(f"WARNING: {mismatches} result(s) did not match expectations.")
        print("(Network conditions may cause false positives -- re-run to confirm.)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
