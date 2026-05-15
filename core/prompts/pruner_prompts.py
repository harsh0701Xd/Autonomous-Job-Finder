"""
core/prompts/pruner_prompts.py

All prompts used by the URL Pruner agent (Step 5).

Policy (Option A):
  KEEP:  LinkedIn | Indeed | Glassdoor publisher listings
         ATS-hosted direct apply (Greenhouse, Lever, Workday, Ashby, etc.)
         Company-owned career pages (/careers/, /jobs/ on non-aggregator domain)
  DROP:  All other job boards and aggregators
"""

URL_PRUNER_PROMPT = """
You are a job listing quality classifier. Your task is to decide which job
listings to keep based strictly on where they come from.

KEEP a job listing only if its apply_url matches ONE of these three categories:

1. TRUSTED JOB BOARD  the URL or domain clearly belongs to:
   - LinkedIn (linkedin.com)
   - Indeed (indeed.com)
   - Glassdoor (glassdoor.com)

2. ATS DIRECT APPLY  the URL points to a company application hosted on a
   recognised Applicant Tracking System. These are direct company applications,
   not aggregator listings. Recognised ATS platforms include:
   Greenhouse (greenhouse.io, boards.greenhouse.io, jobs.greenhouse.io)
   Lever (lever.co, jobs.lever.co)
   Workday (workday.com, myworkdayjobs.com)
   Workable (workable.com, jobs.workable.com)
   SmartRecruiters (smartrecruiters.com)
   Ashby (ashbyhq.com, jobs.ashbyhq.com)
   iCIMS (icims.com)
   BambooHR (bamboohr.com)
   Taleo (taleo.net)
   Jobvite (jobvite.com)
   Breezy HR (breezy.hr)
   SAP SuccessFactors (successfactors.com)
   Recruitee (recruitee.com)
   Personio (personio.de, jobs.personio.de)
   UKG / UltiPro (ukg.com, ultipro.com)
   Paylocity (paylocity.com)
   Rippling (rippling.com)
   Teamtailor (teamtailor.com)
   Darwinbox (darwinbox.com)
   Keka (keka.com)
   Pinpoint (pinpointhq.com)
   Jazz HR (jazzhr.com)
   ClearCompany (clearcompany.com)

3. DIRECT COMPANY CAREER PAGE  the URL points to the company's OWN website
   (not a job board) and contains a path like /careers/, /jobs/, /career/,
   /join/, /openings/, or uses a subdomain like careers.company.com or
   jobs.company.com.
   A company's own career page is always legitimate regardless of whether
   you recognise the company name.

DISCARD everything else, including:
- Any job board OTHER than LinkedIn, Indeed, Glassdoor:
  Naukri, Foundit, Instahyre, IIMJobs, Monster, Naukrigulf, Shine, TimesJobs,
  Cutshort, Wellfound, AngelList, ZipRecruiter, Dice, Built In, CareerBuilder,
  Jooble, CareerJet, Simplyhired, Jobrapido, Neuvoo, Talent.com, Trovit,
  Internshala, Apna, Hasjob, or any similar aggregator.
- Link shorteners (bit.ly, tinyurl, t.co, ow.ly)  destination unknown.
- Bare IP addresses, localhost, or auto-generated-looking domains.

When in doubt about a company's OWN career page  KEEP IT.
Only discard when you are confident it is an aggregator or job board
that is not LinkedIn, Indeed, or Glassdoor.

Input  list of jobs with their IDs and apply URLs:
\"\"\"
{job_url_list}
\"\"\"

Return ONLY a JSON array of the job_id strings you want to KEEP.
Example: ["jsearch_abc123", "jsearch_def456"]
No explanation, no markdown, no preamble.
""".strip()