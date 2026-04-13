"""
frontend/app.py

Autonomous Job Finder — Streamlit Frontend

Single-file app with step-based navigation:
  Step 1: Onboarding  — preferences + resume upload
  Step 2: Confirm     — profile selection (human-in-the-loop gate)
  Step 3: Results     — ranked jobs + company signals dashboard

Talks to the FastAPI backend at API_BASE_URL via httpx.
All state persisted in st.session_state across reruns.
"""

import time
import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL  = "http://localhost:8000/api/v1"
POLL_INTERVAL = 2      # seconds between status polls
MAX_POLLS     = 60     # give up after 2 minutes

st.set_page_config(
    page_title  = "Autonomous Job Finder",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Session state initialisation ──────────────────────────────────────────────

def _init_state():
    defaults = {
        "step":              1,       # 1=onboard, 2=confirm, 3=results
        "session_id":        None,
        "suggested_profiles": [],
        "confirmed_profiles": [],
        "results":           None,
        "error":             None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

def _api_post(path: str, **kwargs) -> dict:
    r = httpx.post(f"{API_BASE_URL}{path}", timeout=60.0, **kwargs)
    r.raise_for_status()
    return r.json()


def _api_get(path: str) -> dict:
    r = httpx.get(f"{API_BASE_URL}{path}", timeout=30.0)
    r.raise_for_status()
    return r.json()


# ── Shared styling helpers ────────────────────────────────────────────────────

def _badge(label: str, color: str) -> str:
    """Return an inline HTML badge."""
    colors = {
        "blue":   ("background:#E6F1FB;color:#185FA5",),
        "green":  ("background:#EAF3DE;color:#3B6D11",),
        "amber":  ("background:#FAEEDA;color:#854F0B",),
        "red":    ("background:#FCEBEB;color:#A32D2D",),
        "gray":   ("background:#F1EFE8;color:#5F5E5A",),
    }
    style = colors.get(color, colors["gray"])[0]
    return (
        f'<span style="font-size:11px;padding:2px 10px;border-radius:12px;'
        f'font-weight:500;{style}">{label}</span>'
    )


def _action_badge(action: str) -> str:
    mapping = {
        "apply_now":       ("Apply now",        "blue"),
        "apply_with_note": ("Apply with note",  "gray"),
        "monitor":         ("Monitor",           "amber"),
        "skip":            ("Skip",              "gray"),
    }
    label, color = mapping.get(action, ("Unknown", "gray"))
    return _badge(label, color)


def _signal_badge(sig_type: str) -> str:
    mapping = {
        "funding":          "green",
        "expansion":        "green",
        "product_launch":   "green",
        "headcount_growth": "green",
        "hiring_freeze":    "amber",
        "layoff":           "red",
        "neutral":          "gray",
    }
    color = mapping.get(sig_type, "gray")
    label = sig_type.replace("_", " ").title()
    return _badge(label, color)


def _strength_label(strength: str) -> str:
    return {"high": "●●●", "medium": "●●○", "low": "●○○"}.get(strength, "○○○")


# ── Step 1: Onboarding ────────────────────────────────────────────────────────

def show_onboarding():
    st.markdown("## Find your next role")
    st.markdown(
        "Upload your resume and set your preferences. "
        "Our AI pipeline will parse your profile, suggest matching roles, "
        "and find real job listings — in under 60 seconds."
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
            st.success(f"✓ {uploaded.name} ready to upload")

    with col_right:
        st.markdown("#### Preferences")

        location = st.text_input(
            "Location",
            placeholder="e.g. Bangalore, Delhi NCR, Remote",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            work_type = st.selectbox(
                "Work type",
                ["hybrid", "remote", "on-site", "any"],
            )
        with col_b:
            seniority = st.selectbox(
                "Seniority target",
                ["step_up", "same_level", "open"],
                format_func=lambda x: {
                    "step_up":    "Step up",
                    "same_level": "Same level",
                    "open":       "Open",
                }[x],
            )

    st.divider()

    ready = uploaded and location.strip()

    if st.button(
        "Parse resume and find profiles →",
        type="primary",
        disabled=not ready,
        use_container_width=True,
    ):
        _run_onboarding(
            uploaded   = uploaded,
            location   = location.strip(),
            work_type  = work_type,
            seniority  = seniority,
        )

    if not ready:
        st.caption("Upload your resume and enter a location to continue.")


def _run_onboarding(uploaded, location, work_type, seniority):
    with st.spinner("Creating session..."):
        try:
            sess = _api_post(
                "/sessions",
                json={
                    "location":             location,
                    "work_type":            work_type,
                    "seniority_preference": seniority,
                },
            )
            st.session_state.session_id = sess["session_id"]
        except Exception as e:
            st.error(f"Could not create session: {e}")
            return

    with st.spinner("Parsing your resume with Claude... (15-20 seconds)"):
        try:
            # Determine content type from filename rather than browser mime type
            # Browser can return empty string for some PDFs
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
        except Exception as e:
            st.error(f"Resume upload failed: {e}")
            return

    if data.get("status") == "parse_failed":
        st.error(
            f"Could not parse your resume: {data.get('parse_failure_reason', 'Unknown error')}. "
            "Try a text-based PDF rather than a scanned image."
        )
        return

    if data.get("status") == "awaiting_confirmation":
        st.session_state.suggested_profiles = data.get("suggested_profiles", [])
        st.session_state.step = 2
        st.rerun()
    else:
        st.error(f"Unexpected status: {data.get('status')}")


# ── Step 2: Profile confirmation ──────────────────────────────────────────────

def show_confirmation():
    st.markdown("## Confirm your job profiles")
    st.markdown(
        "We analysed your resume and found these matching roles. "
        "Select up to 3 to search for, or add your own."
    )

    st.divider()

    profiles = st.session_state.suggested_profiles
    selected = []

    for p in profiles:
        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
            p.get("confidence", "low"), "⚪"
        )
        stretch_tag = " *(stretch role)*" if p.get("is_stretch") else ""

        checked = st.checkbox(
            f"{confidence_color} **{p['title']}**{stretch_tag}",
            value=not p.get("is_stretch", False),
            key=f"profile_{p['title']}",
        )
        if checked:
            selected.append(p["title"])

        with st.expander("Why this role?", expanded=False):
            st.caption(p.get("match_reason", ""))

    st.divider()

    custom_input = st.text_input(
        "Add a custom role (optional)",
        placeholder="e.g. GenAI Engineer, Data Science Lead...",
    )
    custom_profiles = (
        [t.strip() for t in custom_input.split(",") if t.strip()]
        if custom_input
        else []
    )

    all_selected = selected + custom_profiles
    count_label = f"{len(all_selected)} profile{'s' if len(all_selected) != 1 else ''} selected"

    col1, col2 = st.columns([3, 1])
    with col2:
        st.caption(count_label)

    if st.button(
        f"Search jobs for {len(all_selected)} profile{'s' if len(all_selected) != 1 else ''} →",
        type="primary",
        disabled=len(all_selected) == 0,
        use_container_width=True,
    ):
        _run_confirmation(selected_titles=selected, custom_profiles=custom_profiles)

    if st.button("← Start over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _init_state()
        st.rerun()


def _run_confirmation(selected_titles: list, custom_profiles: list):
    with st.spinner("Confirming profiles..."):
        try:
            _api_post(
                f"/sessions/{st.session_state.session_id}/confirm",
                json={
                    "selected_titles": selected_titles,
                    "custom_profiles": custom_profiles,
                },
            )
        except Exception as e:
            st.error(f"Confirmation failed: {e}")
            return

    # Poll until results ready
    progress = st.progress(0, text="Searching job boards across your profiles...")
    status_messages = {
        "searching": "Searching job boards across your profiles...",
        "ranking":   "Ranking and deduplicating results...",
        "complete":  "Fetching company intelligence signals...",
    }

    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            status = _api_get(f"/sessions/{st.session_state.session_id}/status")
        except Exception:
            continue

        current = status.get("status", "searching")
        msg     = status_messages.get(current, "Processing...")
        pct     = min(int((i / MAX_POLLS) * 100), 95)
        progress.progress(pct, text=msg)

        if status.get("results_ready") or current == "complete":
            progress.progress(100, text="Done!")
            break
    else:
        st.warning("Pipeline is taking longer than expected. Try refreshing.")
        return

    # Fetch final results
    try:
        results = _api_get(f"/sessions/{st.session_state.session_id}/results")
        st.session_state.results = results
        st.session_state.step    = 3
        st.rerun()
    except Exception as e:
        st.error(f"Could not fetch results: {e}")


# ── Step 3: Results dashboard ─────────────────────────────────────────────────

def show_results():
    results  = st.session_state.results
    jobs     = results.get("jobs", [])
    signals  = results.get("hiring_signals", [])
    watchlist = results.get("watch_list", [])

    apply_now_count = sum(1 for j in jobs if j.get("recommended_action") == "apply_now")

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_restart = st.columns([5, 1])
    with col_title:
        st.markdown("## Your job matches")
    with col_restart:
        if st.button("New search", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _init_state()
            st.rerun()

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jobs found",  len(jobs))
    m2.metric("Apply now",   apply_now_count)
    m3.metric("Signals",     len(signals))
    m4.metric("Watch list",  len(watchlist))

    st.divider()

    # ── Main layout: jobs left, signals right ─────────────────────────────────
    col_jobs, col_signals = st.columns([2, 1])

    # ── Jobs column ───────────────────────────────────────────────────────────
    with col_jobs:
        st.markdown("#### Ranked jobs")

        # Filter controls
        fc1, fc2 = st.columns(2)
        with fc1:
            action_filter = st.selectbox(
                "Filter by action",
                ["All", "Apply now", "Apply with note", "Monitor"],
                label_visibility="collapsed",
            )
        with fc2:
            min_score = st.slider(
                "Min fit score",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                format="%d%%",
                label_visibility="collapsed",
            )

        action_map = {
            "All":             None,
            "Apply now":       "apply_now",
            "Apply with note": "apply_with_note",
            "Monitor":         "monitor",
        }
        filter_action = action_map[action_filter]

        filtered = [
            j for j in jobs
            if (filter_action is None or j.get("recommended_action") == filter_action)
            and int(j.get("fit_score", 0) * 100) >= min_score
        ]

        st.caption(f"Showing {len(filtered)} of {len(jobs)} jobs")

        for job in filtered:
            _render_job_card(job)

    # ── Signals column ────────────────────────────────────────────────────────
    with col_signals:
        st.markdown("#### Company signals")

        if signals:
            for sig in signals:
                _render_signal_card(sig)
        else:
            st.caption("No signals found for companies in your results.")

        if watchlist:
            st.markdown("---")
            st.markdown("**Watch list** — not posting yet")
            for sig in watchlist:
                _render_watchlist_card(sig)


def _render_job_card(job: dict):
    fit_pct = int(job.get("fit_score", 0) * 100)

    # Color code fit score
    if fit_pct >= 70:
        score_color = "#185FA5"
    elif fit_pct >= 50:
        score_color = "#854F0B"
    else:
        score_color = "#888780"

    gap_text = ""
    if job.get("gap_skills"):
        gaps = ", ".join(job["gap_skills"][:3])
        gap_text = f'<span style="font-size:11px;color:#888780">Gaps: {gaps}</span>'

    matched_via = " · ".join(job.get("matched_via", []))

    action_html   = _action_badge(job.get("recommended_action", "monitor"))
    matched_html  = _badge(matched_via, "gray") if matched_via else ""

    card_html = f"""
<div style="background:var(--background-color,#fff);border:0.5px solid rgba(0,0,0,0.1);
     border-radius:12px;padding:14px 16px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <div style="flex:1;min-width:0">
      <p style="font-size:14px;font-weight:600;margin:0 0 2px;
         white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
        {job.get("title","")}</p>
      <p style="font-size:12px;color:#666;margin:0">
        {job.get("company","")} · {job.get("location","")} · {job.get("work_type","").replace("-"," ").title()}</p>
    </div>
    <div style="text-align:right;flex-shrink:0;margin-left:12px">
      <p style="font-size:18px;font-weight:600;color:{score_color};margin:0">{fit_pct}%</p>
      <p style="font-size:11px;color:#888;margin:0">fit</p>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
    {action_html}
    {matched_html}
    {gap_text}
  </div>
  <div style="margin-top:10px">
    <a href="{job.get('apply_url','')}" target="_blank"
       style="font-size:12px;color:#185FA5;text-decoration:none">
      Apply → {job.get('company','')}
    </a>
  </div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


def _render_signal_card(sig: dict):
    direction = "+" if sig.get("is_positive", True) else "−"
    sig_html  = _signal_badge(sig.get("signal_type", "neutral"))
    strength  = _strength_label(sig.get("signal_strength", "low"))
    jobs_n    = sig.get("jobs_you_matched", 0)
    profiles  = " · ".join(sig.get("relevant_to_profiles", []))

    card_html = f"""
<div style="border:0.5px solid rgba(0,0,0,0.1);border-radius:12px;padding:12px 14px;margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
    {sig_html}
    <span style="font-size:11px;color:#888">{strength}</span>
  </div>
  <p style="font-size:13px;font-weight:600;margin:0 0 2px">{direction} {sig.get("company","")}</p>
  <p style="font-size:11px;color:#666;margin:0 0 6px;line-height:1.4">{sig.get("summary","")}</p>
  <p style="font-size:11px;color:#888;margin:0">{jobs_n} matched job{"s" if jobs_n != 1 else ""}{f" · {profiles}" if profiles else ""}</p>
  {"" if not sig.get("source_url") else f'<a href="{sig["source_url"]}" target="_blank" style="font-size:11px;color:#185FA5;text-decoration:none">Source →</a>'}
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


def _render_watchlist_card(sig: dict):
    card_html = f"""
<div style="border:0.5px dashed rgba(0,0,0,0.15);border-radius:12px;padding:12px 14px;margin-bottom:8px">
  <p style="font-size:13px;font-weight:600;margin:0 0 2px">{sig.get("company","")}</p>
  <p style="font-size:11px;color:#666;margin:0 0 4px;line-height:1.4">
    {sig.get("summary","").replace("Watch list: ","")}</p>
  {"" if not sig.get("source_url") else f'<a href="{sig["source_url"]}" target="_blank" style="font-size:11px;color:#185FA5;text-decoration:none">Source →</a>'}
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


# ── Progress indicator ────────────────────────────────────────────────────────

def _show_step_indicator():
    steps = ["Upload", "Confirm", "Results"]
    current = st.session_state.step
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                st.markdown(
                    f'<p style="text-align:center;font-size:12px;color:#3B6D11">✓ {label}</p>',
                    unsafe_allow_html=True,
                )
            elif i == current:
                st.markdown(
                    f'<p style="text-align:center;font-size:12px;font-weight:600;color:#185FA5">● {label}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p style="text-align:center;font-size:12px;color:#aaa">○ {label}</p>',
                    unsafe_allow_html=True,
                )
    st.divider()


# ── Main router ───────────────────────────────────────────────────────────────

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
