"""
job_sources.py
================
Normalised integrations for every job board API Jobee talks to.

Every fetch_* function returns a list of dicts with this common shape:

{
    "id": str,                 # unique id, used for de-duplication
    "title": str,
    "company": str,
    "location": str,
    "remote": bool,
    "description": str,        # plain text, html stripped
    "url": str,
    "salary_min": float | None, # in salary_currency, per salary_period
    "salary_max": float | None,
    "salary_currency": str | None,
    "salary_period": str | None,   # "year" | "month" | "hour"
    "job_type": str,            # "internship" | "full_time" | "part_time"
                                 # | "contract" | "other"
    "posted_date": str | None,
    "source": str,
    "tags": list[str],
}

Free, no-key sources: Remotive, Arbeitnow, RemoteOK, Jobicy, Himalayas
Optional, key-based sources: Adzuna, JSearch (RapidAPI)
"""

import re
import html
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger("job_sources")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; JobeeRadar/1.0)"}
TIMEOUT = 15


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean_html(raw):
    """Strip tags and collapse whitespace from an HTML/description blob."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _classify_job_type(title, raw_type, description=""):
    """Best-effort mapping to: internship | full_time | part_time | contract | other"""
    title_l = (title or "").lower()
    raw_l = (raw_type or "").lower()
    desc_l = (description or "")[:600].lower()

    if "intern" in raw_l or re.search(r"\bintern(ship)?\b", title_l):
        return "internship"
    if "part" in raw_l:
        return "part_time"
    if any(k in raw_l for k in ("contract", "freelance", "temporary", "temp")):
        return "contract"
    if "full" in raw_l:
        return "full_time"
    if re.search(r"\bintern(ship)?\b", desc_l):
        return "internship"
    return "other"


def _parse_salary_string(text):
    """Pull min/max numbers + currency out of free-text salary strings like
    '$50,000 - $70,000' or '£60k - £90k a year'."""
    if not text:
        return None, None, None

    clean = text.replace(",", "")
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([kK])?", clean)
    nums = []
    for num, k in matches:
        val = float(num)
        if k:
            val *= 1000
        if val >= 100:  # ignore stray small numbers (e.g. "40 hours")
            nums.append(val)

    currency = "USD"
    if "£" in text or "GBP" in text:
        currency = "GBP"
    elif "€" in text or "EUR" in text:
        currency = "EUR"
    elif "PKR" in text or "Rs" in text:
        currency = "PKR"
    elif "₹" in text or "INR" in text:
        currency = "INR"

    period = "year"
    if re.search(r"hour|/hr|hourly", text, re.I):
        period = "hour"
    elif re.search(r"month|/mo|monthly", text, re.I):
        period = "month"

    if not nums:
        return None, None, currency
    if len(nums) == 1:
        return nums[0], nums[0], currency
    return min(nums), max(nums), currency


def _safe_get(url, **kwargs):
    kwargs.setdefault("headers", HEADERS)
    kwargs.setdefault("timeout", TIMEOUT)
    r = requests.get(url, **kwargs)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# 1. Remotive — free, no key
# ---------------------------------------------------------------------------

def fetch_remotive(query="", limit=100):
    out = []
    try:
        params = {"limit": limit}
        if query:
            params["search"] = query
        data = _safe_get("https://remotive.com/api/remote-jobs", params=params).json()
        for j in data.get("jobs", []):
            salary_min, salary_max, currency = _parse_salary_string(j.get("salary", ""))
            out.append({
                "id": f"remotive-{j.get('id')}",
                "title": (j.get("title") or "").strip(),
                "company": (j.get("company_name") or "").strip(),
                "location": j.get("candidate_required_location") or "Remote",
                "remote": True,
                "description": _clean_html(j.get("description", ""))[:4000],
                "url": j.get("url", ""),
                "salary_min": salary_min,
                "salary_max": salary_max,
                "salary_currency": currency,
                "salary_period": "year",
                "job_type": _classify_job_type(j.get("title"), j.get("job_type"), j.get("description")),
                "posted_date": j.get("publication_date"),
                "source": "Remotive",
                "tags": (j.get("tags") or [])[:8],
            })
    except Exception as e:
        logger.warning("Remotive fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# 2. Arbeitnow — free, no key (no real text search, filter client side)
# ---------------------------------------------------------------------------

def fetch_arbeitnow(query="", max_pages=2):
    out = []
    terms = [t.lower() for t in re.split(r"[,\s]+", query) if t]
    try:
        for page in range(1, max_pages + 1):
            data = _safe_get(
                "https://www.arbeitnow.com/api/job-board-api", params={"page": page}
            ).json()
            jobs = data.get("data", [])
            if not jobs:
                break
            for j in jobs:
                haystack = f"{j.get('title','')} {' '.join(j.get('tags') or [])} {j.get('description','')}".lower()
                if terms and not any(t in haystack for t in terms):
                    continue
                job_types = j.get("job_types") or []
                created = j.get("created_at")
                posted = None
                if created:
                    try:
                        posted = datetime.fromtimestamp(int(created), tz=timezone.utc).isoformat()
                    except (ValueError, TypeError):
                        posted = None
                out.append({
                    "id": f"arbeitnow-{j.get('slug')}",
                    "title": (j.get("title") or "").strip(),
                    "company": (j.get("company_name") or "").strip(),
                    "location": j.get("location") or ("Remote" if j.get("remote") else ""),
                    "remote": bool(j.get("remote")),
                    "description": _clean_html(j.get("description", ""))[:4000],
                    "url": j.get("url", ""),
                    "salary_min": None,
                    "salary_max": None,
                    "salary_currency": None,
                    "salary_period": None,
                    "job_type": _classify_job_type(j.get("title"), " ".join(job_types), j.get("description")),
                    "posted_date": posted,
                    "source": "Arbeitnow",
                    "tags": (j.get("tags") or [])[:8],
                })
            if not (data.get("links") or {}).get("next"):
                break
    except Exception as e:
        logger.warning("Arbeitnow fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# 3. RemoteOK — free, no key (no text search, filter client side)
# ---------------------------------------------------------------------------

def fetch_remoteok(query="", limit=200):
    out = []
    terms = [t.lower() for t in re.split(r"[,\s]+", query) if t]
    try:
        data = _safe_get("https://remoteok.com/api").json()
        for j in data[1:]:  # first element is a legal notice, not a job
            if not isinstance(j, dict) or "position" not in j:
                continue
            haystack = f"{j.get('position','')} {' '.join(j.get('tags') or [])} {j.get('description','')}".lower()
            if terms and not any(t in haystack for t in terms):
                continue
            out.append({
                "id": f"remoteok-{j.get('id')}",
                "title": (j.get("position") or "").strip(),
                "company": (j.get("company") or "").strip(),
                "location": j.get("location") or "Remote",
                "remote": True,
                "description": _clean_html(j.get("description", ""))[:4000],
                "url": j.get("url") or f"https://remoteok.com/remote-jobs/{j.get('id')}",
                "salary_min": float(j["salary_min"]) if j.get("salary_min") else None,
                "salary_max": float(j["salary_max"]) if j.get("salary_max") else None,
                "salary_currency": "USD",
                "salary_period": "year",
                "job_type": _classify_job_type(j.get("position"), "", j.get("description")),
                "posted_date": j.get("date"),
                "source": "RemoteOK",
                "tags": (j.get("tags") or [])[:8],
            })
            if len(out) >= limit:
                break
    except Exception as e:
        logger.warning("RemoteOK fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# 4. Jobicy — free, no key
# ---------------------------------------------------------------------------

JOBICY_GEO_MAP = {
    "usa": "usa", "united states": "usa", "us": "usa",
    "uk": "uk", "united kingdom": "uk", "britain": "uk",
    "canada": "canada",
    "australia": "australia",
    "europe": "europe", "eu": "europe",
    "asia": "asia",
}


def guess_jobicy_geo(country):
    if not country:
        return None
    return JOBICY_GEO_MAP.get(country.strip().lower())


def fetch_jobicy(query="", geo=None, count=50):
    out = []
    try:
        params = {"count": count}
        if query:
            params["tag"] = query.split(",")[0].split()[0]  # jobicy tags are single keywords
        if geo:
            params["geo"] = geo
        data = _safe_get("https://jobicy.com/api/v2/remote-jobs", params=params).json()
        for j in data.get("jobs", []):
            out.append({
                "id": f"jobicy-{j.get('id')}",
                "title": (j.get("jobTitle") or "").strip(),
                "company": (j.get("companyName") or "").strip(),
                "location": j.get("jobGeo") or "Remote",
                "remote": True,
                "description": _clean_html(j.get("jobDescription") or j.get("jobExcerpt") or "")[:4000],
                "url": j.get("url", ""),
                "salary_min": float(j["annualSalaryMin"]) if j.get("annualSalaryMin") else None,
                "salary_max": float(j["annualSalaryMax"]) if j.get("annualSalaryMax") else None,
                "salary_currency": j.get("salaryCurrency") or "USD",
                "salary_period": "year",
                "job_type": _classify_job_type(
                    j.get("jobTitle"), " ".join(j.get("jobType") or []), j.get("jobDescription")
                ),
                "posted_date": j.get("pubDate"),
                "source": "Jobicy",
                "tags": (j.get("jobIndustry") or [])[:8],
            })
    except Exception as e:
        logger.warning("Jobicy fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# 5. Himalayas — free, no key. Supports employment_type=Intern for internships
# ---------------------------------------------------------------------------

def fetch_himalayas(query="", employment_type=None, limit=100):
    out = []
    try:
        params = {}
        if query:
            params["q"] = query
        if employment_type:
            params["employment_type"] = employment_type
        data = _safe_get("https://himalayas.app/jobs/api/search", params=params).json()
        for j in (data.get("jobs") or [])[:limit]:
            company = j.get("companyName")
            if not company and isinstance(j.get("company"), dict):
                company = j["company"].get("name")
            restrictions = j.get("locationRestrictions") or []
            out.append({
                "id": f"himalayas-{j.get('guid') or j.get('id') or j.get('title')}",
                "title": (j.get("title") or "").strip(),
                "company": (company or "").strip(),
                "location": ", ".join(restrictions) if restrictions else "Remote (worldwide)",
                "remote": True,
                "description": _clean_html(j.get("description") or j.get("excerpt") or "")[:4000],
                "url": j.get("applicationLink") or j.get("guid") or "",
                "salary_min": float(j["salaryMin"]) if j.get("salaryMin") else None,
                "salary_max": float(j["salaryMax"]) if j.get("salaryMax") else None,
                "salary_currency": j.get("salaryCurrency") or "USD",
                "salary_period": "year",
                "job_type": _classify_job_type(
                    j.get("title"), j.get("employmentType", ""), j.get("description")
                ),
                "posted_date": j.get("pubDate"),
                "source": "Himalayas",
                "tags": (j.get("categories") or [])[:8],
            })
    except Exception as e:
        logger.warning("Himalayas fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# 6. Adzuna — optional, needs app_id + app_key (free signup)
#    Covers: us gb at au be br ca ch de es fr in it mx nl nz pl sg za
# ---------------------------------------------------------------------------

ADZUNA_COUNTRIES = {
    "us", "gb", "at", "au", "be", "br", "ca", "ch", "de", "es",
    "fr", "in", "it", "mx", "nl", "nz", "pl", "sg", "za",
}

ADZUNA_COUNTRY_NAMES = {
    "united states": "us", "usa": "us", "us": "us",
    "united kingdom": "gb", "uk": "gb", "britain": "gb",
    "austria": "at", "australia": "au", "belgium": "be", "brazil": "br",
    "canada": "ca", "switzerland": "ch", "germany": "de", "spain": "es",
    "france": "fr", "india": "in", "italy": "it", "mexico": "mx",
    "netherlands": "nl", "new zealand": "nz", "poland": "pl",
    "singapore": "sg", "south africa": "za",
}


def guess_adzuna_country(country):
    if not country:
        return None
    code = ADZUNA_COUNTRY_NAMES.get(country.strip().lower())
    if code in ADZUNA_COUNTRIES:
        return code
    return None


def fetch_adzuna(query, country_code, location="", app_id=None, app_key=None,
                 results_per_page=50, salary_min=None):
    out = []
    if not app_id or not app_key or country_code not in ADZUNA_COUNTRIES:
        return out
    try:
        params = {
            "app_id": app_id, "app_key": app_key,
            "what": query, "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        if location:
            params["where"] = location
        if salary_min:
            params["salary_min"] = int(salary_min)
        data = _safe_get(
            f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1", params=params
        ).json()
        for j in data.get("results", []):
            currency = "GBP" if country_code == "gb" else "EUR" if country_code in (
                "de", "fr", "es", "it", "nl", "at", "be") else "USD"
            out.append({
                "id": f"adzuna-{j.get('id')}",
                "title": (j.get("title") or "").strip(),
                "company": ((j.get("company") or {}).get("display_name") or "").strip(),
                "location": (j.get("location") or {}).get("display_name", ""),
                "remote": False,
                "description": _clean_html(j.get("description", ""))[:4000],
                "url": j.get("redirect_url", ""),
                "salary_min": j.get("salary_min"),
                "salary_max": j.get("salary_max"),
                "salary_currency": currency,
                "salary_period": "year",
                "job_type": _classify_job_type(j.get("title"), j.get("contract_time", ""), j.get("description")),
                "posted_date": j.get("created"),
                "source": "Adzuna",
                "tags": [(j.get("category") or {}).get("label", "")] if (j.get("category") or {}).get("label") else [],
            })
    except Exception as e:
        logger.warning("Adzuna fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# 7. JSearch (RapidAPI) — optional, needs a RapidAPI key.
#    Aggregates Google for Jobs -> LinkedIn, Indeed, Glassdoor and more.
#    Best source for country-specific listings (e.g. Pakistan).
# ---------------------------------------------------------------------------

def fetch_jsearch(query, location="", api_key=None, num_pages=2, employment_types=None):
    out = []
    if not api_key:
        return out
    try:
        q = f"{query} in {location}" if location else query
        params = {"query": q, "page": "1", "num_pages": str(num_pages), "date_posted": "month"}
        if employment_types:
            params["employment_types"] = employment_types
        headers = {**HEADERS, "X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
        resp = requests.get("https://jsearch.p.rapidapi.com/search", params=params, headers=headers, timeout=TIMEOUT)
        if resp.status_code == 429:
            logger.warning("JSearch fetch failed: 429 Too Many Requests (monthly quota hit)")
            return out
        resp.raise_for_status()
        data = resp.json()
        for j in data.get("data", []):
            loc_parts = [p for p in [j.get("job_city"), j.get("job_country")] if p]
            location_str = ", ".join(loc_parts) or ("Remote" if j.get("job_is_remote") else "")
            out.append({
                "id": f"jsearch-{j.get('job_id')}",
                "title": (j.get("job_title") or "").strip(),
                "company": (j.get("employer_name") or "").strip(),
                "location": location_str,
                "remote": bool(j.get("job_is_remote")),
                "description": _clean_html(j.get("job_description", ""))[:4000],
                "url": j.get("job_apply_link") or j.get("job_google_link") or "",
                "salary_min": j.get("job_min_salary"),
                "salary_max": j.get("job_max_salary"),
                "salary_currency": j.get("job_salary_currency"),
                "salary_period": (j.get("job_salary_period") or "year").lower(),
                "job_type": _classify_job_type(
                    j.get("job_title"), j.get("job_employment_type", ""), j.get("job_description")
                ),
                "posted_date": j.get("job_posted_at_datetime_utc"),
                "source": "JSearch",
                "tags": [],
            })
    except Exception as e:
        logger.warning("JSearch fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_SOURCE_NAMES = ["Remotive", "Arbeitnow", "RemoteOK", "Jobicy", "Himalayas", "Adzuna", "JSearch"]


def fetch_all_jobs(queries, prefs=None, api_keys=None):
    """
    queries: list[str] search terms (job titles / keywords)
    prefs: {
        "location": str,            # free text, e.g. "Lahore, Pakistan"
        "adzuna_country": str|None, # 2-letter Adzuna code, or None to skip
        "jobicy_geo": str|None,
        "himalayas_employment_type": str|None,  # e.g. "Intern"
        "salary_min": float|None,
    }
    api_keys: {
        "adzuna_app_id": str, "adzuna_app_key": str, "jsearch_key": str,
    }

    Returns (jobs: list[dict], source_counts: dict[str, int], errors: list[str])
    """
    prefs = prefs or {}
    api_keys = api_keys or {}

    jobs_for = {}
    futures = {}

    # JSearch has a tight free-tier quota (200 req/month).
    # Only send the first query to avoid burning through it in one search.
    jsearch_queries = queries[:1]

    # When the user wants internships, sources without a native internship filter
    # (Remotive, Arbeitnow, RemoteOK, Jobicy) get augmented queries like
    # "AI Engineer intern" so they return relevant internship listings too.
    wants_internship = bool(prefs.get("himalayas_employment_type"))  # set when internship selected
    def _intern_q(q):
        ql = q.lower()
        if wants_internship and "intern" not in ql:
            return q + " intern"
        return q

    with ThreadPoolExecutor(max_workers=16) as pool:
        for q in queries:
            iq = _intern_q(q)
            futures[pool.submit(fetch_remotive, iq)] = "Remotive"
            futures[pool.submit(fetch_arbeitnow, iq)] = "Arbeitnow"
            futures[pool.submit(fetch_remoteok, iq)] = "RemoteOK"

            # Jobicy requires a tag of at least 3 chars and chokes on short
            # acronyms like "AI". Walk every word in the query and pick the
            # first one that is long enough; fall back to the full query only
            # if nothing qualifies (avoids the 400 that "AI" causes).
            words = [w for w in re.split(r"[\s,]+", iq) if len(w) >= 3]
            jobicy_tag = words[0] if words else ""
            if jobicy_tag:
                futures[pool.submit(fetch_jobicy, jobicy_tag, prefs.get("jobicy_geo"))] = "Jobicy"

            futures[pool.submit(fetch_himalayas, q, prefs.get("himalayas_employment_type"))] = "Himalayas"

            if prefs.get("adzuna_country") and api_keys.get("adzuna_app_id") and api_keys.get("adzuna_app_key"):
                futures[pool.submit(
                    fetch_adzuna, iq, prefs["adzuna_country"], prefs.get("location", ""),
                    api_keys["adzuna_app_id"], api_keys["adzuna_app_key"],
                    50, prefs.get("salary_min"),
                )] = "Adzuna"

        # JSearch: one query only to protect free-tier quota
        if api_keys.get("jsearch_key"):
            for q in jsearch_queries:
                iq = _intern_q(q)
                futures[pool.submit(fetch_jsearch, iq, prefs.get("location", ""), api_keys["jsearch_key"])] = "JSearch"

        errors = []
        results = []
        for fut in as_completed(futures):
            source = futures[fut]
            try:
                results.extend(fut.result())
            except Exception as e:
                errors.append(f"{source}: {e}")

    # de-duplicate by URL (fallback to id)
    seen = set()
    deduped = []
    source_counts = {name: 0 for name in ALL_SOURCE_NAMES}
    for j in results:
        key = j.get("url") or j.get("id")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(j)
        source_counts[j["source"]] = source_counts.get(j["source"], 0) + 1

    return deduped, source_counts, errors
