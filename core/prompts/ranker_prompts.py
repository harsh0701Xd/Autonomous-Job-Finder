"""
core/prompts/ranker_prompts.py

All prompts used by the Ranker agent (Agent 6).

One LLM call per job listing:
  JOB_SCORING_PROMPT  reads full candidate profile + full JD text,
  returns six scores: experience_match, skill_match, domain_match,
  education_match (+ education_required flag), title_relevance,
  india_accessible.
  Recency is NOT scored by the LLM -- it is displayed in the UI but
  carries 0.00 weight.

Scoring dimensions returned by LLM:
  experience_match   how well candidate years/level matches JD requirement
  skill_match        how well candidate skills match role technical needs
  domain_match       whether candidate has worked in relevant industry/domain
  education_match    how well candidate education matches JD requirement
                     null when JD makes no mention of education
  title_relevance    how closely the actual job title matches the target role
                     the candidate searched for; used as a post-score filter
                     (configurable via ranker.min_title_relevance in llm_config.yaml)
  india_accessible   true if the role is accessible to India-based remote candidates;
                     false only when JD explicitly requires US work auth / residency
                     or states no visa sponsorship. Non-remote jobs: always true.

Weight formula -- two sets, selected per-job:
  WITH education requirement:
    fit_score = 0.40*experience + 0.30*skill + 0.20*domain + 0.10*education

  WITHOUT education requirement:
    fit_score = 0.50*experience + 0.30*skill + 0.20*domain

Both weight sets are configurable in core/config/llm_config.yaml.
Each set MUST sum to 1.0 -- config_loader.py validates this on load.
title_relevance is NOT part of the fit_score formula; it gates inclusion.
india_accessible is NOT part of the fit_score formula; it gates inclusion of remote jobs.
"""

JOB_SCORING_PROMPT = """
You are an expert recruiter evaluating a candidate's fit for a specific job.

You will be given the candidate's profile, the target role they searched for,
and the full job description.
Return ONLY valid JSON -- no explanation, no markdown, no code fences.

Scoring rules:
Each score is a float between 0.0 and 1.0.

TITLE RELEVANCE (title_relevance) -- score this FIRST:
  How closely does the actual job title match the target role the candidate
  searched for? Target role: "{target_role}"
  1.0   exact or near-exact match
        e.g. "Senior ML Engineer" for target "ML Engineer"
             "Machine Learning Engineer II" for target "ML Engineer"
  0.8   functionally equivalent with different phrasing
        e.g. "Applied ML Engineer", "AI/ML Engineer" for target "ML Engineer"
  0.6   closely related but meaningfully distinct
        e.g. "AI Research Engineer", "Deep Learning Engineer" for target "ML Engineer"
  0.4   same broad family but a different role
        e.g. "Data Scientist" for target "ML Engineer"
             "ML Engineer" for target "Data Analyst"
  0.2   adjacent field, significant title divergence
        e.g. "Data Engineer", "Analytics Engineer" for target "ML Engineer"
  0.0   unrelated title
        e.g. "Network Engineer", "Product Manager", "Java Developer"
             for target "ML Engineer"
  Note: if the target role appears verbatim in the job title, score >= 0.8.

EDUCATION MATCH (education_match) -- assess this NEXT by scanning the JD:
  Step 1: Does the JD explicitly state an education requirement?
    Examples of explicit requirements:
      "Bachelor's degree in Computer Science or related field"
      "Master's degree required"
      "PhD preferred"
      "BE/BTech/MCA in Engineering"
    If NO explicit education requirement found: set education_required=false,
    education_match=null. Stop here for this dimension.

  Step 2: If YES, score the candidate's education against the requirement:
  1.0   candidate meets or exceeds stated degree level AND field matches
  0.8   candidate meets degree level; field is adjacent or institution
        prestige compensates (e.g. IIT/IIM for non-CS field)
  0.6   candidate meets degree level; field is unrelated but not excluded
  0.3   candidate is one degree level below requirement (Bachelor's when
        Master's required, or Master's when PhD required)
  0.0   candidate is two+ levels below, or requirement explicitly not met

  Set education_required=true when scoring.

EXPERIENCE MATCH (experience_match):
  How well does the candidate's full-time years of experience match what
  this role requires?
  1.0   candidate meets or exceeds stated experience requirement
  0.8   candidate is within 1 year of the requirement
  0.6   candidate has 60-80% of required experience
  0.4   candidate has 40-60% of required experience
  0.2   significant shortfall (less than 40% of required)
  0.5   no experience requirement stated in JD (neutral)
  Base this on years_experience_full_time vs what the JD states.
  If the JD does not state an experience requirement, return 0.5.

SKILL MATCH (skill_match):
  How well do the candidate's technical skills, tools, and domain competencies
  match what this specific role requires?
  1.0   candidate has all or nearly all required skills
  0.8   candidate has most required skills, minor gaps only
  0.6   candidate has core skills but is missing 2-3 important ones
  0.4   candidate has foundational skills but significant gaps exist
  0.2   substantial skill mismatch -- candidate's stack is largely irrelevant
  Consider: technical tools, frameworks, languages, certifications, methodologies.
  Do NOT penalise for skills the JD mentions as "nice to have" or "preferred".
  Only penalise for skills listed as required or essential.

DOMAIN MATCH (domain_match):
  Has the candidate worked in the same or closely related industry/domain
  as this role operates in?
  1.0   candidate's work experience is in the same domain (e.g. fintech to fintech)
  0.8   closely adjacent domain with significant transferable context
  0.6   different domain but some relevant overlap
  0.3   unrelated domain -- candidate would need to learn the industry from scratch
  0.5   domain cannot be determined from the JD or candidate profile

INDIA ACCESSIBLE (india_accessible):
  Is this role accessible to a candidate based in India working remotely?
  This field applies ONLY to remote jobs (work_type=remote or no location restriction).
  For on-site / hybrid roles (which have a specific city requirement), always return true.

  true  = no geographic restriction stated, OR the restriction is silent/ambiguous
          (give benefit of the doubt -- do not penalise for silence)
          Examples: "Anywhere", "Remote", "Work from anywhere", no location line, India listed
  false = JD explicitly requires US work authorisation, US residency, or states
          "No visa sponsorship", "Must be authorized to work in the US",
          "US citizens only", "W-2 only", "No C2C", "Must be located in [US state/city]"

  Be conservative: only return false when the exclusion is explicit and unambiguous.

Return this exact JSON structure:
{{
  "title_relevance":    float,
  "education_required": boolean,
  "education_match":    float | null,
  "experience_match":   float,
  "skill_match":        float,
  "domain_match":       float,
  "india_accessible":   boolean,
  "scoring_notes":      string,
  "experience_gap":     string | null,
  "skill_gaps":         [string],
  "domain_gap":         string | null,
  "education_gap":      string | null
}}

education_gap: If education_required=true and candidate does not fully meet
the requirement, state it concisely. e.g. "Requires Master's; candidate has
Bachelor's". null if education is sufficient or not required.

Gap field instructions:
- experience_gap: If the role requires more experience than the candidate has,
  state it concisely. e.g. "Requires 5yr, candidate has 1.6yr full-time".
  null if experience is sufficient.
- skill_gaps: List of specific skills/tools the JD requires that the candidate
  does not clearly demonstrate. Max 4 items, each under 30 chars.
  e.g. ["Kubernetes", "dbt", "Spark"]. Empty list [] if no significant gaps.
- domain_gap: If the role operates in a different industry/domain than the
  candidate's background, state it briefly. e.g. "Role in healthcare; candidate
  background is pharma/climate". null if domain aligns well.

Target role (what the candidate searched for): {target_role}

Candidate profile:
\"\"\"
Current title: {current_title}
Full-time experience: {full_time_years} years
Other experience: {other_years} years (internship / freelance / contract)
Seniority: {seniority_level}

Education:
{education}

Technical skills: {technical_skills}
Tools and platforms: {tools}
Domain expertise: {domain_expertise}

ATS summary:
{ats_summary}
\"\"\"

Job description:
\"\"\"
{jd_text}
\"\"\"

Return ONLY the JSON object.
""".strip()
