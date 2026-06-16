"""
reranker.py
================
Optional-but-automatic AI re-ranking pass.

After the embedding based categorisation runs, this takes the jobs
sitting closest to the similarity thresholds - the genuinely ambiguous
cases - and asks the LLM to read each one and confirm or correct which
of the three buckets it belongs in, with a short reason.

This always runs as part of a search, but only on a small, capped
subset of jobs (the borderline ones). Any failure - timeout, bad JSON,
auth error, anything - falls back silently to the embedding-only
result. It never blocks or breaks a search.
"""

import json
import re
import logging

from matcher import SIM_MATCH_THRESHOLD, SIM_INTEREST_THRESHOLD

logger = logging.getLogger("reranker")

MAX_RERANK_JOBS = 20
BORDERLINE_BAND = 0.07  # similarity distance from a threshold to count as "borderline"

VALID_BUCKETS = {"matches", "might_interest", "consider"}


def select_borderline_jobs(buckets, max_jobs=MAX_RERANK_JOBS):
    """Pick the jobs whose bucket placement is least certain: those
    sitting close to either similarity threshold. Returns a list of
    (bucket_name, job) tuples, most ambiguous first."""
    candidates = []
    for bucket_name, jobs in buckets.items():
        for job in jobs:
            sim = job.get("similarity", 0.0)
            dist = min(abs(sim - SIM_MATCH_THRESHOLD), abs(sim - SIM_INTEREST_THRESHOLD))
            if dist <= BORDERLINE_BAND:
                candidates.append((dist, bucket_name, job))

    candidates.sort(key=lambda c: c[0])
    return [(b, j) for _, b, j in candidates[:max_jobs]]


PROMPT_TEMPLATE = """You are helping sort job search results for a candidate into three buckets:

- "matches": strongly fits the candidate's skills, target roles, location, and pay expectations
- "might_interest": good overlap with the candidate's background, but something (location, pay, seniority, or role type) doesn't fully line up
- "consider": loosely related, worth knowing about but not a strong fit

CANDIDATE PROFILE
Summary: {summary}
Experience level: {experience_level}
Target roles: {target_roles}
Skills: {skills}

Below are {n} job listings that are borderline cases. For EACH one, decide which bucket it belongs in and give a short reason (max 12 words).

Return ONLY a JSON array, no markdown, no commentary, in this exact shape:
[{{"id": "<job id>", "bucket": "matches" | "might_interest" | "consider", "reason": "<short reason>"}}]

JOBS:
{jobs_block}
"""


def _strip_json_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _build_jobs_block(borderline):
    lines = []
    for _, job in borderline:
        desc = (job.get("description") or "")[:300].replace("\n", " ")
        lines.append(
            f"- id: {job['id']}\n"
            f"  title: {job.get('title', '')}\n"
            f"  company: {job.get('company', '')}\n"
            f"  location: {job.get('location', '')} (remote={job.get('remote', False)})\n"
            f"  job_type: {job.get('job_type', '')}\n"
            f"  current_similarity: {job.get('similarity', 0)}\n"
            f"  description: {desc}"
        )
    return "\n".join(lines)


def rerank_borderline(buckets, cv_profile, client, model, max_jobs=MAX_RERANK_JOBS):
    """
    Reviews the borderline jobs in `buckets` with the LLM and moves
    them between buckets where the model disagrees with the embedding
    based placement. Mutates and returns `buckets`.

    On any failure, returns `buckets` unchanged.
    """
    try:
        borderline = select_borderline_jobs(buckets, max_jobs=max_jobs)
        if not borderline:
            return buckets

        prompt = PROMPT_TEMPLATE.format(
            summary=cv_profile.get("summary", ""),
            experience_level=cv_profile.get("experience_level", ""),
            target_roles=", ".join(cv_profile.get("target_roles", [])),
            skills=", ".join(cv_profile.get("skills", [])),
            n=len(borderline),
            jobs_block=_build_jobs_block(borderline),
        )

        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You return only valid JSON, nothing else."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = _strip_json_fences(response.message.content)
        decisions = json.loads(raw)
        if not isinstance(decisions, list):
            return buckets
    except Exception as e:
        logger.warning("AI re-rank skipped: %s", e)
        return buckets

    decision_map = {}
    for d in decisions:
        if not isinstance(d, dict):
            continue
        job_id = d.get("id")
        bucket = d.get("bucket")
        if job_id and bucket in VALID_BUCKETS:
            decision_map[str(job_id)] = (bucket, str(d.get("reason", "")).strip())

    if not decision_map:
        return buckets

    # index every job by id so we can move it between buckets
    job_index = {}
    for bucket_name, jobs in buckets.items():
        for job in jobs:
            job_index[str(job["id"])] = (bucket_name, job)

    for job_id, (new_bucket, reason) in decision_map.items():
        if job_id not in job_index:
            continue
        old_bucket, job = job_index[job_id]

        job["ai_reviewed"] = True
        if reason:
            existing = job.get("match_reasons") or []
            job["match_reasons"] = [reason] + [r for r in existing if r != reason]

        if new_bucket != old_bucket:
            try:
                buckets[old_bucket].remove(job)
            except ValueError:
                continue
            buckets[new_bucket].append(job)
            job_index[job_id] = (new_bucket, job)

    for key in buckets:
        buckets[key].sort(key=lambda j: j.get("composite_score", j["similarity"]), reverse=True)

    return buckets
