"""
frontend/app.py

Autonomous Job Finder  Streamlit Frontend

Single-file app with step-based navigation:
  Step 1: Onboarding   preferences + resume upload
  Step 2: Confirm      profile selection (human-in-the-loop gate) -- SINGLE profile via radio
  Step 3: Results      ranked jobs dashboard

Talks to the FastAPI backend at API_BASE_URL via httpx.
All state persisted in st.session_state across reruns.
"""

import time
import httpx
import streamlit as st

#  Config

API_BASE_URL  = "http://localhost:8000/api/v1"
POLL_INTERVAL = 2
MAX_POLLS     = 75   # 150s ceiling — avoids collision with uvicorn timeout-keep-alive=120s

st.set_page_config(
    page_title  = "Autonomous Job Finder",
    page_icon   = "",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

#  Session state initialisation

def _init_state():
    defaults = {
        "step":               1,
        "session_id":         None,
        "suggested_profiles": [],
        "confirmed_profiles": [],
        "results":            None,
        "error":              None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


#  API helpers

def _api_post(path: str, **kwargs) -> dict:
    r = httpx.post(f"{API_BASE_URL}{path}", timeout=180.0, **kwargs)
    if not r.is_success:
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        raise httpx.HTTPStatusError(
            f"HTTP {r.status_code}: {detail}",
            request=r.request,
            response=r,
        )
    return r.json()


def _api_get(path: str) -> dict:
    r = httpx.get(f"{API_BASE_URL}{path}", timeout=30.0)
    r.raise_for_status()
    return r.json()


#  Shared styling helpers

def _badge(label: str, color: str) -> str:
    colors = {
        "blue":  ("background:#E6F1FB;color:#185FA5",),
        "green": ("background:#EAF3DE;color:#3B6D11",),
        "amber": ("background:#FAEEDA;color:#854F0B",),
        "red":   ("background:#FCEBEB;color:#A32D2D",),
        "gray":  ("background:#F1EFE8;color:#5F5E5A",),
    }
    style = colors.get(color, colors["gray"])[0]
    return (
        f'<span style="font-size:11px;padding:2px 10px;border-radius:12px;'
        f'font-weight:500;{style}">{label}</span>'
    )


#  Step 1: Onboarding

def show_onboarding():
    st.markdown("## Find your next role")
    st.markdown(
        "Upload your resume and set your preferences. "
        "Our AI pipeline will parse your profile, suggest matching roles, "
        "and find real job listings in under 60 seconds."
    )

    st.divider()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### Your resume")
        uploaded = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.success(f" {uploaded.name} ready to upload")

    with col_right:
        st.markdown("#### Preferences")

        work_type = st.selectbox(
            "Work arrangement",
            ["office", "remote"],
            format_func=lambda x: {
                "office": "On-site / Hybrid",
                "remote": "Remote only",
            }[x],
        )

        INDIA_LOCATIONS = [
            "Bengaluru", "Chennai", "Delhi", "Gurgaon", "Hyderabad",
            "Kolkata", "Mumbai", "Noida", "Greater Noida", "Pune",
            "Ahmedabad", "Bhopal", "Chandigarh", "Indore", "Jaipur",
            "Kochi", "Lucknow", "Patna", "Ranchi",
        ]

        if work_type == "office":
            location = st.selectbox("Location", options=INDIA_LOCATIONS)
        else:
            location = None
            st.caption("Remote only  searching across all geographies.")

        seniority = st.selectbox(
            "Seniority target",
            ["step_up", "same_level"],
            format_func=lambda x: {
                "step_up":    "Step up",
                "same_level": "Same level",
            }[x],
        )
        st.caption(
            '"Same" shows roles you\'re qualified for today. '
            '"Step up" includes stretch roles where you may not meet every requirement '
            '— expect fewer matches.'
        )

    st.divider()

    ready = bool(uploaded)

    if st.button(
        "Parse resume and find profiles ",
        type="primary",
        disabled=not ready,
        use_container_width=True,
    ):
        _run_onboarding(
            uploaded  = uploaded,
            location  = location,
            work_type = work_type,
            seniority = seniority,
        )

    if not ready:
        st.caption("Upload your resume to continue.")


def _run_onboarding(uploaded, location, work_type, seniority):
    MAX_BYTES = 10 * 1024 * 1024
    if uploaded.size > MAX_BYTES:
        st.error(
            f"Your file is {uploaded.size // (1024*1024)} MB — the limit is 10 MB. "
            "Try compressing the PDF or removing embedded images before uploading."
        )
        return

    with st.spinner("Creating session..."):
        try:
            sess = _api_post(
                "/sessions",
                json={
                    "location":             location or "",
                    "work_type":            work_type,
                    "seniority_preference": seniority,
                },
            )
            st.session_state.session_id = sess["session_id"]
        except httpx.ConnectError:
            st.error(
                "Cannot reach the job finder service. "
                "Make sure the backend is running (`docker-compose up`) and try again."
            )
            return
        except httpx.HTTPStatusError as e:
            st.error(f"Session creation failed ({e.response.status_code}). Please try again.")
            return
        except Exception as e:
            st.error(f"Unexpected error creating session: {e}")
            return

    with st.spinner("Parsing your resume with Claude Sonnet… (15–20 seconds)"):
        try:
            filename = uploaded.name
            if filename.lower().endswith(".pdf"):
                content_type = "application/pdf"
            elif filename.lower().endswith(".docx"):
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                content_type = uploaded.type or "application/octet-stream"

            data = _api_post(
                f"/sessions/{st.session_state.session_id}/resume",
                files={"file": (filename, uploaded.getvalue(), content_type)},
            )
        except httpx.ConnectError:
            st.error(
                "Lost connection to the backend while uploading. "
                "Check that the Docker container is still running and try again."
            )
            st.session_state.session_id = None   # clear stale ID so retry creates a fresh session
            return
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 422:
                st.error("The server rejected your file. Only PDF and DOCX files are supported.")
            elif status_code == 413:
                st.error("Your resume file is too large for the server. Try a smaller or compressed version.")
            elif status_code == 500:
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text[:300]
                st.error(f"Resume upload failed with a server error (500).\n\n**Detail:** {detail}")
            else:
                st.error(f"Upload failed (HTTP {status_code}). Please try again.")
            st.session_state.session_id = None   # clear stale ID so retry creates a fresh session
            return
        except Exception as e:
            st.error(f"Resume upload failed: {e}")
            st.session_state.session_id = None   # clear stale ID so retry creates a fresh session
            return

    if data.get("status") == "parse_failed":
        reason = data.get("parse_failure_reason", "")
        if "scanned" in reason.lower() or "image" in reason.lower():
            hint = "Your PDF appears to be a scanned image. Export a text-based PDF from Word or Google Docs."
        elif "password" in reason.lower() or "encrypt" in reason.lower():
            hint = "Your file appears to be password-protected. Remove the password and try again."
        elif "empty" in reason.lower() or "no text" in reason.lower():
            hint = "No readable text was found. Make sure your file isn't blank or image-only."
        else:
            hint = "Try a text-based PDF exported directly from Word or Google Docs."
        st.error(f"Could not parse your resume. {hint}")
        if reason:
            st.caption(f"Reason: {reason}")
        return

    if data.get("status") == "awaiting_confirmation":
        st.session_state.suggested_profiles = data.get("suggested_profiles", [])
        st.session_state.step = 2
        st.rerun()
    else:
        st.error(
            f"The parser returned an unexpected status: '{data.get('status')}'. "
            "Please start over."
        )


#  Step 2: Profile confirmation (SINGLE profile via radio buttons)

def show_confirmation():
    st.markdown("## Choose your target role")
    st.markdown(
        "We analysed your resume and found these matching profiles. "
        "**Select one role** to search jobs for."
    )

    st.divider()

    profiles = st.session_state.suggested_profiles

    # Guard: parser succeeded but returned no profiles (unlikely but possible)
    if not profiles:
        st.error(
            "We couldn't identify any matching job profiles from your resume. "
            "This can happen if the resume is very short or in an unsupported format. "
            "Please start over and try a different file."
        )
        if st.button("↩ Start over", key="empty_profiles_restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _init_state()
            st.rerun()
        return

    # Render each profile as a bordered card with title + rationale always visible.
    # Radio button sits inside each card so the selection is visually anchored.
    # We use a hidden radio widget to track state, then render cards manually.
    selected_index = st.radio(
        "Select a profile",
        options=range(len(profiles)),
        format_func=lambda i: profiles[i]["title"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Per-profile cards — always show rationale, highlight selected
    for i, p in enumerate(profiles):
        is_selected = (i == selected_index)
        confidence_color = {"high": "#3B6D11", "medium": "#854F0B", "low": "#A32D2D"}.get(
            p.get("confidence", "low"), "#5F5E5A"
        )
        confidence_label = {"high": "High match", "medium": "Medium match", "low": "Lower match"}.get(
            p.get("confidence", "low"), ""
        )
        stretch_tag = (
            '<span style="font-size:11px;padding:2px 8px;border-radius:10px;'
            'background:#FAEEDA;color:#854F0B;font-weight:500;margin-left:6px">Stretch</span>'
            if p.get("is_stretch") else ""
        )
        border_color  = "#E05252" if is_selected else "var(--color-border-tertiary)"
        bg_color      = "var(--color-background-secondary)" if is_selected else "transparent"
        radio_dot     = "🔴" if is_selected else "⚪"

        card_html = (
            f'<div style="border:1.5px solid {border_color};border-radius:10px;'
            f'padding:12px 16px;margin-bottom:8px;background:{bg_color}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<span style="font-size:14px">{radio_dot}</span>'
            f'<span style="font-size:14px;font-weight:600;color:var(--color-text-primary)">'
            f'{p["title"]}</span>'
            f'{stretch_tag}'
            f'<span style="margin-left:auto;font-size:11px;font-weight:500;color:{confidence_color}">'
            f'{confidence_label}</span>'
            f'</div>'
            f'<p style="font-size:12px;color:var(--color-text-secondary);margin:0;'
            f'line-height:1.5;padding-left:22px">{p.get("match_reason", "")}</p>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    selected_profile = profiles[selected_index]

    st.divider()

    col1, col2 = st.columns([3, 1])
    with col2:
        st.caption(f"1 profile selected")

    if st.button(
        f"Search jobs for {selected_profile['title']} →",
        type="primary",
        use_container_width=True,
    ):
        _run_confirmation(selected_titles=[selected_profile["title"]])

    if st.button("↩ Start over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _init_state()
        st.rerun()


def _run_confirmation(selected_titles: list):
    with st.spinner("Confirming your profile…"):
        try:
            _api_post(
                f"/sessions/{st.session_state.session_id}/confirm",
                json={
                    "selected_titles": selected_titles,
                    "custom_profiles": [],
                },
            )
        except httpx.ConnectError:
            st.error(
                "Lost connection to the backend. "
                "Check that Docker is still running and try confirming again."
            )
            return
        except httpx.TimeoutException:
            st.error(
                "The request timed out. The pipeline may still be running — "
                "try refreshing the page in 60 seconds to see if results are ready."
            )
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                st.error("This session has already been confirmed. Try starting a new search.")
            elif e.response.status_code == 404:
                st.error("Session not found — it may have expired. Please start over.")
            else:
                st.error(
                    f"Confirmation failed (HTTP {e.response.status_code}). "
                    "Please try again or start a new search."
                )
            return
        except Exception as e:
            st.error(f"Confirmation failed: {e}")
            return

    st.info(
        "**Heads up:** Fit scores are AI estimates based on keyword and domain alignment — "
        "not a guarantee of interview success. Always read the full JD before applying.",
        icon="ℹ️",
    )

    AGENT_SEQUENCE = ["job_search", "url_pruner", "ranker", "finalise"]
    AGENT_MESSAGES = {
        "job_search": ("🔍", "Searching job boards…", "Querying JSearch for fresh listings — takes 5–10 seconds"),
        "url_pruner": ("🧹", "Filtering listing quality…", "Removing aggregator links and low-quality listings"),
        "ranker":     ("⚖️", f"Scoring jobs for {selected_titles[0]}…", "Each listing scored by Claude Haiku — expect 60–90 seconds"),
        "finalise":   ("📊", "Finalising results…", "Aggregating metrics and saving session data"),
        "complete":   ("✅", "Done!", ""),
        "error":      ("❌", "Pipeline error", ""),
    }

    started_at     = time.time()
    network_errors = 0
    progress       = st.progress(0, text="🔍 Searching job boards…")
    sub_caption_ph = st.empty()
    elapsed_ph     = st.empty()

    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            status = _api_get(f"/sessions/{st.session_state.session_id}/status")
            network_errors = 0
        except httpx.ConnectError:
            network_errors += 1
            if network_errors >= 3:
                st.error(
                    "Lost connection to the backend (3 consecutive failures). "
                    "Check that Docker is still running. "
                    "Your results may still be processing — try refreshing in 60 seconds."
                )
                return
            elapsed_ph.caption(f"⚠️ Network hiccup — retrying… ({network_errors}/3)")
            continue
        except Exception:
            network_errors += 1
            elapsed_ph.caption(f"⚠️ Status check failed — retrying… ({network_errors}/3)")
            if network_errors >= 5:
                st.error("Too many status check failures. Try refreshing the page in 60 seconds.")
                return
            continue

        pipeline_status = status.get("status", "searching")
        current_agent   = status.get("current_agent") or pipeline_status

        icon, msg, sub = AGENT_MESSAGES.get(
            current_agent,
            AGENT_MESSAGES.get(pipeline_status, ("⏳", "Processing…", "")),
        )

        if current_agent in AGENT_SEQUENCE:
            agent_idx = AGENT_SEQUENCE.index(current_agent)
            base_pct  = int((agent_idx / len(AGENT_SEQUENCE)) * 90)
            inner_pct = min(int((i % 10) / 10 * 20), 18)
            pct = min(base_pct + inner_pct, 92)
        else:
            pct = min(int((i / MAX_POLLS) * 90), 92)

        elapsed_secs = int(time.time() - started_at)
        progress.progress(pct, text=f"{icon} {msg}")
        if sub:
            sub_caption_ph.caption(sub)
        elapsed_ph.caption(f"⏱ {elapsed_secs}s elapsed")

        if status.get("results_ready") or pipeline_status == "complete":
            progress.progress(100, text="✅ Done!")
            sub_caption_ph.empty()
            elapsed_ph.caption(f"✅ Completed in {elapsed_secs}s")
            break

        if pipeline_status == "error":
            error_msg = status.get("message") or status.get("error") or ""
            progress.progress(pct, text="❌ Pipeline error")
            sub_caption_ph.empty()
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("The AI scoring service hit a rate limit. Wait 60 seconds and try again.")
            elif "timeout" in error_msg.lower():
                st.error("The job scoring step timed out. Try again in a few seconds.")
            elif error_msg:
                st.error(f"Pipeline error: {error_msg}")
            else:
                st.error("An unexpected error occurred. Please start a new search.")
            return
    else:
        elapsed_secs = int(time.time() - started_at)
        # One last attempt to fetch results — pipeline may have finished
        # just as the poll loop expired
        try:
            final_status = _api_get(f"/sessions/{st.session_state.session_id}/status")
            if final_status.get("results_ready") or final_status.get("status") == "complete":
                results = _api_get(f"/sessions/{st.session_state.session_id}/results")
                st.session_state.results = results
                st.session_state.step    = 3
                st.rerun()
                return
        except Exception:
            pass
        st.warning(
            f"The pipeline is taking longer than expected ({elapsed_secs}s). "
            "**Try refreshing the page** — your results may already be ready."
        )
        return

    try:
        results = _api_get(f"/sessions/{st.session_state.session_id}/results")
        st.session_state.results = results
        st.session_state.step    = 3
        st.rerun()
    except httpx.ConnectError:
        st.error("Results are ready but we lost connection fetching them. Try refreshing the page.")
    except httpx.HTTPStatusError as e:
        st.error(f"Could not fetch results (HTTP {e.response.status_code}). Try refreshing the page.")
    except Exception as e:
        st.error(f"Could not fetch results: {e}")


#  Step 3: Results dashboard

def show_results():
    results = st.session_state.results
    jobs    = results.get("jobs", [])

    st.markdown("""
<style>
[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
    border-width: 2px !important;
    border-radius: 10px !important;
    padding: 4px 2px !important;
}
</style>
""", unsafe_allow_html=True)

    high_fit_count   = sum(1 for j in jobs if j.get("fit_score", 0) >= 0.70)
    medium_fit_count = sum(1 for j in jobs if 0.50 <= j.get("fit_score", 0) < 0.70)

    col_title, col_restart = st.columns([5, 1])
    with col_title:
        st.markdown("## Your job matches")
    with col_restart:
        if st.button("New search", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _init_state()
            st.rerun()

    m1, m2, m3 = st.columns(3)
    m1.metric("Jobs found",   len(jobs))
    m2.metric("Strong fit",   high_fit_count,   help="Fit score ≥ 70%")
    m3.metric("Moderate fit", medium_fit_count, help="Fit score 50–70%")

    st.divider()

    if not jobs:
        st.markdown("""
<div style="text-align:center;padding:48px 24px;
     border:0.5px dashed var(--color-border-secondary);
     border-radius:12px;margin-top:16px">
  <p style="font-size:32px;margin:0 0 12px">🔍</p>
  <p style="font-size:16px;font-weight:500;margin:0 0 8px">No matching jobs found</p>
  <p style="font-size:13px;color:var(--color-text-secondary);margin:0 0 20px;line-height:1.6">
    The search ran successfully but no listings passed the quality filter.<br>
    Try: switching to Remote mode · a broader job title · a different city
  </p>
</div>
""", unsafe_allow_html=True)
        return

    st.markdown(
        "<p style='font-size:12px;color:var(--color-text-tertiary);"
        "background:var(--color-background-secondary);border-radius:8px;"
        "padding:8px 12px;line-height:1.5;margin-bottom:12px'>"
        "Fit scores are AI estimates based on keyword and domain alignment — "
        "not a guarantee of interview success. Always read the full JD before applying."
        "</p>",
        unsafe_allow_html=True,
    )

    col_label, col_slider, col_count = st.columns([1.2, 4, 1.2])
    with col_label:
        st.markdown(
            "<p style='font-size:13px;color:var(--color-text-secondary);padding-top:6px'>Min fit score</p>",
            unsafe_allow_html=True,
        )
    with col_slider:
        min_score = st.slider(
            "Min fit score",
            min_value=0, max_value=100, value=0, step=5,
            format="%d%%", label_visibility="collapsed",
        )
    filtered = [j for j in jobs if int(j.get("fit_score", 0) * 100) >= min_score]
    with col_count:
        st.markdown(
            f"<p style='font-size:12px;color:var(--color-text-tertiary);"
            f"padding-top:6px;text-align:right'>{len(filtered)} of {len(jobs)}</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='font-size:13px;font-weight:500;"
        "color:var(--color-text-secondary);margin-bottom:10px'>Ranked jobs</p>",
        unsafe_allow_html=True,
    )

    for job in filtered:
        _render_job_card(job)


def _fit_color(pct: int) -> str:
    if pct >= 70: return "#3B6D11"
    if pct >= 50: return "#854F0B"
    return "#888780"


def _posted_label(posted_date: str) -> tuple[str, str]:
    """Return (absolute_date, relative_label) for a posted_date ISO string."""
    if not posted_date:
        return "", ""
    try:
        from datetime import datetime, timezone
        dt  = datetime.fromisoformat(posted_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days = (now - dt).days

        absolute = dt.strftime("%d %b %Y").lstrip("0")  # e.g. "14 Apr 2026" (cross-platform)

        if days == 0:    relative = "today"
        elif days == 1:  relative = "yesterday"
        elif days <= 7:  relative = f"{days}d ago"
        elif days <= 30: relative = f"{days // 7}w ago"
        elif days <= 365:relative = f"{days // 30}mo ago"
        else:            relative = f"{days // 365}y ago"

        return absolute, relative
    except Exception:
        return "", ""


def _render_job_card(job: dict):
    fit_pct       = int(job.get("fit_score",        0) * 100)
    exp_pct       = int(job.get("experience_score", 0) * 100)
    skill_pct     = int(job.get("skill_score",      0) * 100)
    domain_pct    = int(job.get("domain_score",     0) * 100)
    edu_required  = job.get("education_required",   False)
    edu_score_raw = job.get("education_score")
    edu_pct       = int(edu_score_raw * 100) if edu_required and edu_score_raw is not None else None
    sparse_jd     = job.get("sparse_jd", False)

    score_color = _fit_color(fit_pct)

    def _tile(label: str, pct: int) -> str:
        color = _fit_color(pct)
        return (
            f'<div style="text-align:center;background:var(--color-background-primary);'
            f'border:0.5px solid var(--color-border-tertiary);'
            f'border-radius:8px;padding:6px 4px">'
            f'<p style="font-size:9px;color:var(--color-text-tertiary);margin:0 0 2px;'
            f'text-transform:uppercase;letter-spacing:.05em">{label}</p>'
            f'<p style="font-size:14px;font-weight:500;color:{color};margin:0">{pct}%</p>'
            f'</div>'
        )

    matched_via  = job.get("matched_via", [])
    profile_name = matched_via[0] if matched_via else (job.get("matched_profile") or "")
    badge_color  = "blue" if "Senior" in profile_name or "Lead" in profile_name else "gray"
    badge_html   = _badge(profile_name, badge_color) if profile_name else ""

    # --- POSTING DATE: prominent, near top, both absolute and relative ---
    abs_date, rel_label = _posted_label(job.get("posted_date", ""))
    if abs_date and rel_label:
        posted_html = (
            f'<span style="font-size:12px;font-weight:500;'
            f'color:var(--color-text-secondary);'
            f'background:var(--color-background-secondary);'
            f'border-radius:6px;padding:2px 8px;white-space:nowrap">'
            f'📅 {abs_date} &nbsp;·&nbsp; {rel_label}'
            f'</span>'
        )
    elif abs_date:
        posted_html = (
            f'<span style="font-size:12px;font-weight:500;'
            f'color:var(--color-text-secondary);'
            f'background:var(--color-background-secondary);'
            f'border-radius:6px;padding:2px 8px">'
            f'📅 {abs_date}'
            f'</span>'
        )
    else:
        posted_html = (
            f'<span style="font-size:11px;color:var(--color-text-tertiary)">'
            f'📅 Date not listed'
            f'</span>'
        )

    notes = job.get("scoring_notes", "")
    notes_html = (
        f'<p style="font-size:12px;color:var(--color-text-secondary);'
        f'border-left:2px solid var(--color-border-secondary);'
        f'padding-left:10px;margin:0 0 10px;line-height:1.5;font-style:italic">'
        f'{notes}</p>'
    ) if notes else ""

    work_type_label = (job.get("work_type") or "").replace("-", " ").title()
    company   = job.get("company", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    location  = job.get("location", "").replace("&", "&amp;")
    title     = job.get("title", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    apply_url = job.get("apply_url", "").replace('"', "%22").replace("{", "%7B").replace("}", "%7D")

    edu_tile_html = _tile("Education", edu_pct) if edu_pct is not None else ""
    grid_cols     = 4 if edu_pct is not None else 3   # Exp, Skill, Domain (+ Edu if present)

    sparse_jd_banner = (
        f'<div style="display:flex;align-items:center;gap:6px;'
        f'background:#FFF8E1;border:0.5px solid #FFD54F;'
        f'border-radius:6px;padding:6px 10px;margin:8px 0;font-size:12px;'
        f'color:#795548">'
        f'⚠️ <strong>Sparse job description</strong> — this listing has very little detail, '
        f'so scores may not reflect your actual fit. Check the full posting before applying.'
        f'</div>'
    ) if sparse_jd else ""

    card_html = (
        # --- Header row: title + fit score ---
        f'<div style="padding:14px 18px 6px 18px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:6px">'
        f'<div style="flex:1;min-width:0">'
        f'<p style="font-size:15px;font-weight:600;margin:0 0 3px;color:var(--color-text-primary)">{title}</p>'
        f'<p style="font-size:12px;color:var(--color-text-secondary);margin:0 0 6px;display:flex;flex-wrap:wrap;gap:4px;align-items:center">'
        f'{company}'
        f'<span style="color:var(--color-text-tertiary);font-size:10px">&middot;</span>'
        f'{location}'
        f'<span style="color:var(--color-text-tertiary);font-size:10px">&middot;</span>'
        f'{work_type_label}'
        f'</p>'
        # --- POSTING DATE: prominent, below company line ---
        f'{posted_html}'
        f'</div>'
        f'<div style="text-align:right;flex-shrink:0">'
        f'<p style="font-size:24px;font-weight:600;color:{score_color};margin:0">{fit_pct}%</p>'
        f'<p style="font-size:10px;color:var(--color-text-tertiary);margin:0">overall fit</p>'
        f'</div></div>'
        # --- Score tiles (Exp, Skill, Domain, Edu) -- NO Recency ---
        f'<div style="display:grid;grid-template-columns:repeat({grid_cols},1fr);gap:6px;margin:10px 0">'
        f'{_tile("Experience", exp_pct)}'
        f'{_tile("Skills",     skill_pct)}'
        f'{_tile("Domain",     domain_pct)}'
        f'{edu_tile_html}'
        f'</div>'
        f'{sparse_jd_banner}'
        f'{notes_html}'
        # --- Footer: profile badge + apply button ---
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:8px;padding-top:10px;margin-bottom:10px;'
        f'border-top:0.5px solid var(--color-border-tertiary)">'
        f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'
        f'{badge_html}'
        f'</div>'
        f'<a href="{apply_url}" target="_blank" '
        f'style="font-size:12px;font-weight:500;color:#185FA5;text-decoration:none;'
        f'padding:4px 12px;border:0.5px solid #B5D4F4;'
        f'border-radius:8px;background:#E6F1FB;white-space:nowrap">'
        f'Apply at {company} &rarr;'
        f'</a>'
        f'</div>'
        f'</div>'
    )

    exp_gap    = job.get("experience_gap")
    skill_gaps = job.get("skill_gaps", [])
    dom_gap    = job.get("domain_gap")
    edu_gap    = job.get("education_gap")

    with st.container(border=True):
        st.markdown(card_html, unsafe_allow_html=True)
        if exp_gap or skill_gaps or dom_gap or edu_gap:
            with st.expander("What's missing?", expanded=False):
                if exp_gap:
                    st.markdown(
                        f"**Experience** &nbsp;&nbsp;"
                        f"<span style='font-size:13px;color:var(--color-text-warning)'>{exp_gap}</span>",
                        unsafe_allow_html=True,
                    )
                if skill_gaps:
                    badges = " &nbsp;".join(
                        f"<span style='background:var(--color-background-secondary);"
                        f"border-radius:4px;padding:2px 8px;font-size:12px;"
                        f"color:var(--color-text-primary)'>{s}</span>"
                        for s in skill_gaps
                    )
                    st.markdown(f"**Skill gaps** &nbsp;&nbsp; {badges}", unsafe_allow_html=True)
                if dom_gap:
                    st.markdown(
                        f"**Domain** &nbsp;&nbsp;"
                        f"<span style='font-size:13px;color:var(--color-text-secondary)'>{dom_gap}</span>",
                        unsafe_allow_html=True,
                    )
                if edu_gap:
                    st.markdown(
                        f"**Education** &nbsp;&nbsp;"
                        f"<span style='font-size:13px;color:var(--color-text-warning)'>{edu_gap}</span>",
                        unsafe_allow_html=True,
                    )


#  Progress indicator

def _show_step_indicator():
    steps   = ["Upload", "Confirm", "Results"]
    current = st.session_state.step
    cols    = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                st.markdown(
                    f'<p style="text-align:center;font-size:12px;color:#3B6D11"> {label}</p>',
                    unsafe_allow_html=True,
                )
            elif i == current:
                st.markdown(
                    f'<p style="text-align:center;font-size:12px;font-weight:600;color:#185FA5"> {label}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p style="text-align:center;font-size:12px;color:#aaa"> {label}</p>',
                    unsafe_allow_html=True,
                )
    st.divider()


#  Main router

def main():
    _show_step_indicator()
    step = st.session_state.step
    if step == 1:
        show_onboarding()
    elif step == 2:
        show_confirmation()
    elif step == 3:
        show_results()


if __name__ == "__main__":
    main()
