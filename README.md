# Jobee 🛰️
### AI Job & Internship Radar

Jobee is a Streamlit app that takes your CV, figures out who you are as a candidate, searches seven job boards at once, and sorts every result into three buckets: jobs that genuinely match you, jobs worth a look, and everything else. Nothing is ever thrown away.

---

## What it does

You upload your CV (PDF, DOCX, or TXT). Jobee extracts your skills, experience level, target roles, and location using a local Ollama model. It then fires off parallel searches across up to seven job sources, runs embedding based similarity scoring against your full profile, applies a geographic proximity multiplier so jobs in your city float to the top, and optionally runs a second AI pass to correct borderline placements.

Every job lands in exactly one of three tabs:

**Matches**: strong semantic similarity, location accessible, salary in range, and role type correct.

**Might Interest You**: good CV overlap but something did not fully line up, such as a foreign location, mismatched pay, or a full time role when you asked for internships.

**Worth Considering**: everything else the search turned up, sorted so the most relevant ones appear first.

---

## Sources

| Source | Key needed | Notes |
|---|---|---|
| Remotive | No | Remote jobs only |
| Arbeitnow | No | Broad international listings |
| RemoteOK | No | Remote tech jobs |
| Jobicy | No | Remote jobs with geo filtering |
| Himalayas | No | Strong internship support |
| Adzuna | Yes | Country specific listings |
| JSearch | Yes | Aggregates LinkedIn, Indeed, Glassdoor via RapidAPI |

The five free sources work with no setup at all. Adzuna and JSearch unlock the Deep Search toggle and pull in significantly more country specific results, including local Pakistan listings when you set your location.

---

## Setup

### Prerequisites

Python 3.10 or newer and an Ollama instance with the `gemma4:cloud` model (or whichever model you configure).

### Install dependencies

```bash
pip install streamlit ollama python-dotenv pymupdf python-docx \
    sentence-transformers requests numpy
```

### Environment variables

Create a `.env` file in the project root. The free sources work without any keys. Add the optional ones to unlock Deep Search.

```env
# Required: your Ollama Cloud credentials
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_MODEL=gemma4:cloud

# Optional: enables Adzuna in Deep Search
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key

# Optional: enables JSearch in Deep Search (RapidAPI key)
JSEARCH_API_KEY=your_rapidapi_key
```

You can get an Adzuna key for free at [developer.adzuna.com](https://developer.adzuna.com). JSearch is available on RapidAPI and has a free tier of 200 requests per month.

### Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to use it

1. **Upload your CV** in PDF, DOCX, or TXT format. Jobee parses it and shows you the profile it extracted, including your detected skills, experience level, and suggested search roles. You can edit the keyword field to add or remove terms before searching.

2. **Set your preferences.** Enter your city and country so geographic scoring works correctly. Pick Internship or Any for role type. If you want full time roles, you can optionally set a pay range and currency.

3. **Click Scan the job universe.** Jobee searches all sources in parallel. With the free sources this typically returns 300 to 400 jobs in about 15 seconds.

4. **Browse the results.** The Matches tab shows your strongest fits sorted by a composite score that combines semantic similarity with geographic proximity. Jobs in your city or country always appear above worldwide remote, which appears above country restricted listings.

---

## How the matching works

### Embedding similarity

Jobee uses `all-MiniLM-L6-v2` from sentence-transformers to embed your full CV profile (summary, skills, target roles, and CV text) as a single vector. Every job title, tags, and description up to 800 characters is embedded the same way. Cosine similarity between the two vectors drives the base score.

Jobs above 0.32 similarity that also pass location, salary, and type checks land in Matches. Jobs above 0.20 land in Might Interest You. Everything else goes to Worth Considering.

### Geographic scoring

Each job gets a geo multiplier that adjusts its position within its bucket:

| Score | What it means |
|---|---|
| 1.00 | Your city or country in the location string |
| 0.88 | Genuinely worldwide remote, blank location, or plain "Remote" |
| 0.80 | Remote flagged by the API but listed with a foreign base location |
| 0.42 | Restricted to a specific list of countries that does not include yours |
| 0.48 | Single foreign country, on site only |
| 0.30 | Hard citizenship or no sponsorship wall detected in the description |

The composite score is `similarity × geo_multiplier`. This means a Lahore job at 62% similarity (composite 0.620) outranks a Canadian remote job at 67% similarity (composite 0.536).

### AI reranking

After embedding scoring, Jobee selects the jobs that sit closest to the bucket thresholds (within 0.07 similarity of either cutoff, up to 20 jobs) and sends them to the LLM with your full profile. The model confirms or corrects each placement and adds a short reason. If this step fails for any reason, the embedding only result is used unchanged.

### Internship detection

When you select Internship mode, Jobee does two things. First, it appends "intern" to search queries sent to sources without a native internship filter (Remotive, Arbeitnow, RemoteOK, Jobicy, Adzuna, JSearch) so those sources actually return intern listings. Himalayas has a native employment type filter and uses that directly. Second, the matcher scans the job title and first 400 characters of the description with a regex so internships that are not tagged correctly by the API are still detected.

Non-internship jobs that score highly are not dropped. They appear in Might Interest You with a note explaining they are full time roles shown in case they are still of interest.

---

## Project structure

```
app.py          Streamlit UI, search orchestration, result rendering
cv_parser.py    PDF/DOCX/TXT extraction and LLM profile extraction
job_sources.py  All seven job board integrations and the fetch orchestrator
matcher.py      Embedding scoring, geo proximity, salary checks, bucket logic
reranker.py     AI borderline reranking pass
```

---

## Contributing

Pull requests are welcome. The most useful things to add would be more job sources, better salary parsing for non-English listings, and saved search history.

---

## Credits

Built by Hassaan Raza.

CV parsing via PyMuPDF and python-docx. Embedding matching via sentence-transformers. Profile extraction and reranking via Ollama Cloud. Job data from Remotive, Arbeitnow, RemoteOK, Jobicy, Himalayas, Adzuna, and JSearch.

Pay comparisons use approximate currency conversion rates for sorting purposes only. Always confirm salary details directly with the employer.
