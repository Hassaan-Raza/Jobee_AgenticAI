"""
matcher.py
================
Embedding based matching + 3 way categorisation of jobs against a CV.

Every job that comes back from job_sources.fetch_all_jobs ends up in
exactly one of three buckets:

    matches        -> strong semantic match AND fits location/salary/type
    might_interest -> decent semantic match, but something doesn't line up
    consider       -> everything else that was returned by the search

Nothing is ever dropped.

Within each bucket, jobs are sorted by a composite score:
    composite = embedding_similarity * geo_proximity_multiplier

This means Lahore jobs appear above worldwide-remote above foreign-country
above citizenship-restricted, all else being equal.
"""

import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(text):
    model = get_embedder()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_many(texts):
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)


def cosine_sim(a, b):
    return float(np.dot(a, b))


# Rough FX -> USD conversion, only used to make salary ranges *directionally*
# comparable across currencies. Not financial advice, just a sort signal.
FX_TO_USD = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "PKR": 0.0036, "INR": 0.012,
    "AED": 0.27, "CAD": 0.73, "AUD": 0.65, "SGD": 0.74, "EGP": 0.020,
    "ZAR": 0.055, "SAR": 0.27, "NZD": 0.60, "CHF": 1.13, "PLN": 0.25,
    "BRL": 0.18, "MXN": 0.058,
}

PERIOD_MULTIPLIER = {"year": 1, "month": 12, "hour": 2080, "day": 260, "week": 52}


def to_usd_per_year(amount, currency, period):
    if amount is None:
        return None
    try:
        rate = FX_TO_USD.get((currency or "USD").upper(), 1.0)
        mult = PERIOD_MULTIPLIER.get((period or "year").lower(), 1)
        return float(amount) * rate * mult
    except (TypeError, ValueError):
        return None


def _skill_overlap(cv_skills, job_text):
    job_l = job_text.lower()
    return [s for s in cv_skills if s.lower() in job_l]


# ---------------------------------------------------------------------------
# Geo-proximity scoring
# ---------------------------------------------------------------------------

# Words in job descriptions/locations that signal citizenship/visa hard limits
CITIZENSHIP_SIGNALS = {
    "us citizen", "u.s. citizen", "united states citizen",
    "canadian citizen", "uk citizen", "british citizen",
    "security clearance", "clearance required", "secret clearance",
    "top secret", "nato clearance",
    "authorized to work in the us", "authorized to work in the united states",
    "must be eligible to work in", "work authorization required",
    "right to work in the uk", "right to work in the us",
    "sponsorship not available", "we do not sponsor",
    "no sponsorship", "cannot sponsor",
    "only open to residents of", "must reside in",
    "must be based in", "must be located in",
}

# Markers that mean the job is genuinely open to anyone worldwide
WORLDWIDE_MARKERS = {
    "worldwide", "anywhere", "global", "international",
    "all countries", "remote (worldwide)", "no restriction",
    "open to all", "location independent",
}

# Generic location strings with no geographic meaning
GENERIC_REMOTE = {"remote", "work from home", "wfh", "telecommute", "distributed"}

# Known country names — used to detect "specific country list that excludes the user"
# We check whether the location string contains 2+ of these AND none match the user's terms.
# We keep this to major English-language terms only; it doesn't need to be exhaustive.
_COUNTRY_NAMES = {
    "afghanistan","albania","algeria","angola","argentina","armenia","australia",
    "austria","azerbaijan","bangladesh","belarus","belgium","bolivia","brazil",
    "bulgaria","cambodia","cameroon","canada","chile","china","colombia","croatia",
    "cuba","czech","denmark","ecuador","egypt","estonia","ethiopia","finland",
    "france","georgia","germany","ghana","greece","guatemala","hungary","india",
    "indonesia","iran","iraq","ireland","israel","italy","japan","jordan","kazakhstan",
    "kenya","korea","latvia","lebanon","libya","lithuania","malaysia","mexico",
    "morocco","myanmar","nepal","netherlands","new zealand","nigeria","norway",
    "pakistan","peru","philippines","poland","portugal","romania","russia",
    "saudi arabia","senegal","serbia","singapore","slovakia","south africa",
    "spain","sri lanka","sudan","sweden","switzerland","syria","taiwan","tanzania",
    "thailand","tunisia","turkey","ukraine","united arab emirates","uae",
    "united kingdom","united states","uruguay","uzbekistan","venezuela","vietnam",
    "zambia","zimbabwe","uk","usa","us",
}


def _any_citizenship_wall(job):
    """Returns True if the job description contains hard citizenship/sponsorship language."""
    full_text = (
        (job.get("location") or "") + " " +
        (job.get("description") or "")[:600]
    ).lower()
    return any(signal in full_text for signal in CITIZENSHIP_SIGNALS)


def _geo_score(job, location_terms, loc_l, is_worldwide, is_generic):
    """
    Returns a float multiplier (0.0 to 1.0) representing how accessible
    this job is from the user's location. Used as a secondary sort key
    so that closer or more accessible jobs float to the top of each bucket.

    Tiers:
      1.00  user's exact city or country
      0.88  truly worldwide remote (blank, generic, or explicit worldwide marker)
      0.80  remote=True from API with a foreign location label (e.g. "Remote, US")
      0.65  remote job with a single named foreign city/company location
      0.42  remote but restricted to a specific list of countries that excludes the user
      0.30  hard citizenship or work authorisation wall detected in description
      0.48  specific foreign country, on-site (no remote flag)
    """
    full_text = (
        (job.get("location") or "") + " " +
        (job.get("description") or "")[:600]
    ).lower()

    # Tier 1.00: exact city/country match
    if location_terms and any(t in loc_l for t in location_terms):
        return 1.00

    # Tier 0.30: citizenship / no-sponsorship hard wall in description
    if any(signal in full_text for signal in CITIZENSHIP_SIGNALS):
        return 0.30

    # Tier 0.88: truly worldwide or generic/blank location
    if is_worldwide or is_generic:
        return 0.88

    # Tier 0.80: job has remote=True flag from API even though location has a place name
    # e.g. "Remote, US" — technically accessible, just labeled with a base country
    if job.get("remote", False):
        # But check if it's a multi-country restricted list
        country_hits = sum(1 for c in _COUNTRY_NAMES if c in loc_l)
        if country_hits >= 2:
            return 0.42
        return 0.80

    # Tier 0.42: location names 2+ specific countries, none of which is the user's
    # e.g. "Australia, Canada, New Zealand, United Kingdom, United States"
    country_hits = sum(1 for c in _COUNTRY_NAMES if c in loc_l)
    if country_hits >= 2:
        return 0.42

    # Tier 0.65: non-remote but single named foreign location
    return 0.48


SIM_MATCH_THRESHOLD = 0.32
SIM_INTEREST_THRESHOLD = 0.20


def _is_internship(job, job_text):
    """
    Broader internship detection — checks the structured job_type field
    AND scans the title and first 400 chars of description for intern keywords.
    Many APIs don't tag the job_type field correctly, so text scanning is essential.
    """
    if job.get("job_type") == "internship":
        return True
    haystack = (
        (job.get("title") or "") + " " +
        (job.get("description") or "")[:400]
    ).lower()
    return bool(re.search(r"\bintern(ship)?\b", haystack))


def categorize_jobs(jobs, cv_text, cv_profile, prefs):
    """
    prefs:
      salary_min_usd : float | None  candidate's minimum desired pay, USD/year
      salary_max_usd : float | None  candidate's maximum expected pay, USD/year
      location_terms : list[str]     lowercased city/country keywords
      job_type        : "internship" | "any"

    Returns dict: {"matches": [...], "might_interest": [...], "consider": [...]}
    """
    cv_blob = " ".join([
        cv_profile.get("summary", ""),
        " ".join(cv_profile.get("skills", [])),
        " ".join(cv_profile.get("target_roles", [])),
        cv_text[:2000],
    ])
    cv_emb = embed(cv_blob)

    job_texts = [f"{j['title']} {' '.join(j.get('tags') or [])} {j['description'][:800]}" for j in jobs]
    job_embs = embed_many(job_texts)

    location_terms = prefs.get("location_terms") or []
    salary_min_usd = prefs.get("salary_min_usd")
    salary_max_usd = prefs.get("salary_max_usd")
    wants_internship = prefs.get("job_type") == "internship"

    buckets = {"matches": [], "might_interest": [], "consider": []}

    for idx, job in enumerate(jobs):
        emb = job_embs[idx] if idx < len(job_embs) else embed(job_texts[idx])
        sim = cosine_sim(cv_emb, emb)
        job["similarity"] = round(max(sim, 0.0), 3)

        job["salary_min_usd"] = to_usd_per_year(job.get("salary_min"), job.get("salary_currency"), job.get("salary_period"))
        job["salary_max_usd"] = to_usd_per_year(job.get("salary_max"), job.get("salary_currency"), job.get("salary_period"))

        # --- location analysis
        loc_l = (job.get("location") or "").lower()
        is_remote = job.get("remote", False)

        if location_terms:
            location_hit = any(term in loc_l for term in location_terms)
            is_worldwide = any(m in loc_l for m in WORLDWIDE_MARKERS)

            # is_generic: blank, a known plain remote word, OR starts with
            # "remote" / "fully remote" / "100% remote" / "remote-first" etc.
            # This catches "Remote, US", "Remote - Europe", "100% Remote",
            # "Fully Remote", "Remote-first" which are all open to anyone.
            _loc_stripped = loc_l.strip()
            is_generic = (
                _loc_stripped == ""
                or _loc_stripped in GENERIC_REMOTE
                or _loc_stripped.startswith("remote")
                or _loc_stripped.startswith("fully remote")
                or bool(re.match(r"^\d+%\s*remote", _loc_stripped))
            )

            # Also treat "remote" appearing anywhere + no citizenship wall as accessible.
            # e.g. "United States (Remote)" — user in Pakistan can still apply.
            # BUT exclude multi-country restricted lists like "Australia, Canada, UK, US"
            _country_hits = sum(1 for c in _COUNTRY_NAMES if c in loc_l)
            _is_country_restricted_list = _country_hits >= 2 and not location_hit

            # location_match = True when:
            #   (a) user's city/country in location string, OR
            #   (b) worldwide/global marker, OR
            #   (c) blank or starts with "remote" or is known generic remote string, OR
            #   (d) remote=True flag set by the API (job is remote regardless of location label)
            #       UNLESS it's a country-restricted list or has a citizenship wall
            location_match = (
                location_hit
                or is_worldwide
                or is_generic
                or (job.get("remote", False) and not _any_citizenship_wall(job) and not _is_country_restricted_list)
            )
        else:
            location_hit = False
            is_worldwide = False
            is_generic = True
            location_match = True  # no preference set

        # --- geo proximity score (used for sorting within buckets)
        geo = _geo_score(job, location_terms, loc_l, is_worldwide, is_generic)
        job["geo_score"] = geo

        # composite score drives ordering within each bucket
        job["composite_score"] = round(sim * geo, 4)

        # --- salary (only penalise when both sides have real numbers)
        salary_known = job.get("salary_max_usd") is not None
        salary_ok = True
        if salary_known and salary_min_usd is not None:
            salary_ok = job["salary_max_usd"] >= salary_min_usd * 0.7
        if salary_known and salary_max_usd is not None and job.get("salary_min_usd") is not None:
            salary_ok = salary_ok and job["salary_min_usd"] <= salary_max_usd * 1.5

        # --- job type (broader detection + hard gate on Matches only)
        is_internship_listing = _is_internship(job, job_texts[idx])
        # Non-internship jobs when internship is selected still appear in
        # Might Interest You — they are NOT dropped from results entirely.
        type_ok = (not wants_internship) or is_internship_listing

        # --- reasons shown to the user
        reasons = []
        overlap = _skill_overlap(cv_profile.get("skills", []), job_texts[idx])
        if overlap:
            reasons.append(f"Matches your skills: {', '.join(overlap[:5])}")
        if location_hit:
            reasons.append("Located where you want to work")
        if is_remote:
            reasons.append("Fully remote")
        if salary_known and salary_min_usd and job["salary_max_usd"] >= salary_min_usd:
            reasons.append("Pays within or above your target range")
        if wants_internship and is_internship_listing:
            reasons.append("Listed as an internship")
        if wants_internship and not is_internship_listing:
            reasons.append("Not an internship listing but shown in case it interests you")
        if geo == 0.30:
            reasons.append("⚠️ May require citizenship or work authorisation in another country")
        job["match_reasons"] = reasons

        # --- categorise
        # When type_ok is False (user wants internship, job is not one),
        # cap at might_interest regardless of similarity score.
        if sim >= SIM_MATCH_THRESHOLD and location_match and salary_ok and type_ok:
            bucket = "matches"
        elif sim >= SIM_INTEREST_THRESHOLD or (not type_ok and sim >= SIM_MATCH_THRESHOLD):
            bucket = "might_interest"
        else:
            bucket = "consider"

        buckets[bucket].append(job)

    # Sort each bucket by composite_score (sim * geo) descending.
    # This means: Lahore internship > worldwide remote > foreign-accessible > restricted.
    for key in buckets:
        buckets[key].sort(key=lambda j: j["composite_score"], reverse=True)

    return buckets