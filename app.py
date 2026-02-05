import os
import uuid
import re
import html
import logging
from datetime import datetime, timedelta, timezone

import requests
from dateutil import parser as dateparser
from bs4 import BeautifulSoup
import feedparser
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    abort,
    Response,
)
import pandas as pd
from dotenv import load_dotenv

# ------------------ CONFIG ------------------

load_dotenv()

# Constants
REMOTEOK_API_URL = "https://remoteok.com/api"
LAST_24_HOURS = timedelta(hours=24)
API_TIMEOUT_SECONDS = 30  # Timeout for external API calls
MAX_REMOTEOK_JOBS = 50  # Maximum number of jobs to fetch from RemoteOK
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# Preloaded sample jobs from portal exports
PRELOADED_JOBS = [
    {
        "Job Title": "Tech Lead Full-Stack Rails Engineer",
        "Company": "Mitre Media",
        "Description": "Architect and implement LLM-powered web applications on a Rails 8 microservices platform. Lead full-stack delivery across Dividend.com and MutualFunds.com with a focus on AI-driven investing experiences.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-01-14 20:46",
        "Job Link": "https://remotive.com/remote-jobs/software-development/tech-lead-full-stack-rails-engineer-2069746",
    },
    {
        "Job Title": "Tech Lead Databricks Data Engineer",
        "Company": "Mitre Media",
        "Description": "Own the data backbone for AI-driven fintech products. Build Databricks ETL pipelines, optimize cloud data platforms, and mentor engineers in a remote-first team.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-01-14 20:46",
        "Job Link": "https://remotive.com/remote-jobs/software-development/tech-lead-databricks-data-engineer-2069747",
    },
    {
        "Job Title": "Software Engineer C++ (Senior)",
        "Company": "Apexver",
        "Description": "Lead design and optimization of low-latency trading systems. Build high-performance C++ services, mentor engineers, and drive architecture for reliability and speed.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-01-14 13:45",
        "Job Link": "https://remotive.com/remote-jobs/software-development/software-engineer-c-senior-2069728",
    },
    {
        "Job Title": "Senior Front-End Developer – Analytics & UX Focused",
        "Company": "Actionable.co",
        "Description": "Own front-end development of analytics dashboards and UX workflows. Build high-impact, accessible experiences across web and mobile in a remote-first team.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-01-13 18:29",
        "Job Link": "https://remotive.com/remote-jobs/software-development/senior-front-end-developer-analytics-ux-focused-remote-2088537",
    },
    {
        "Job Title": "WordPress Developer",
        "Company": "Uncanny Owl",
        "Description": "Build and maintain WordPress plugin integrations at scale. Own features end-to-end, work with APIs, and deliver high-quality remote work for a global product.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-02-04 21:57",
        "Job Link": "https://weworkremotely.com/remote-jobs/uncanny-owl-wordpress-developer",
    },
    {
        "Job Title": "Senior Web Developer",
        "Company": "Zipdev",
        "Description": "Own high-performance marketing sites using React/Next.js or Astro. Build SEO-optimized, accessible experiences with SSR/SSG and headless CMS platforms.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-02-04 17:40",
        "Job Link": "https://weworkremotely.com/remote-jobs/zipdev-senior-web-developer",
    },
    {
        "Job Title": "ASP.NET Developer (C#, MVC, SQL Server, JavaScript)",
        "Company": "Linkage Web Development Solutions",
        "Description": "Maintain and improve ASP.NET WebForms/MVC applications for U.S. clients. Work remotely with SQL Server and MySQL systems.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-02-04 17:40",
        "Job Link": "https://weworkremotely.com/remote-jobs/linkage-web-development-asp-net-developer-c-mvc-sql-server-javascript",
    },
    {
        "Job Title": "Senior Software Engineer (Full Stack/DevOps) – India",
        "Company": "Aspire",
        "Description": "Drive backend, infrastructure, and developer tooling improvements across a distributed engineering team. Focus on reliability, CI/CD, and cloud operations.",
        "Location Type": "remote",
        "Posted At (UTC)": "2026-02-04 17:40",
        "Job Link": "https://weworkremotely.com/remote-jobs/aspire-senior-software-engineer-full-stack-devops-india",
    },
]

# Environment variables
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

app = Flask(__name__)

# Secret key from environment variable (required in production)
if FLASK_SECRET_KEY:
    app.secret_key = FLASK_SECRET_KEY
else:
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    if not app.debug:
        app.logger.warning("Using default secret key. Set FLASK_SECRET_KEY in production!")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


# ------------------ UTILS ------------------

def strip_html(text):
    """
    Remove HTML tags and decode HTML entities from text.
    Returns clean plain text.
    """
    if not text:
        return ""
    
    # Remove HTML tags using regex
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_description(raw_desc):
    """
    Normalize raw job descriptions to clean plain text.
    """
    cleaned = strip_html(raw_desc)
    return cleaned or "See original job post for full role details."


def dummy_scrape_jobs(prefs):
    """
    Placeholder for real scraping.
    For MVP you can:
    - Call public job sources (RemoteOK, Wellfound, company career pages)
    - Filter by last 24 hours, sector, skills etc.

    For now, we simulate 5 jobs that 'match' the preferences.
    """
    skills = [s.strip() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = prefs.get("sector")
    location_type = prefs.get("job_location")

    now = datetime.now(timezone.utc)
    jobs = []

    sample_raw_jobs = [
        {
            "job_title": "Python Backend Developer",
            "company": "TechNova Solutions",
            "raw_desc": "Work on REST APIs, microservices, and integration with cloud platforms.",
            "location_type": "remote",
            "posted_at": now - timedelta(hours=5),
            "link": "https://example.com/job/python-backend-developer",
            "sector": "IT"
        },
        {
            "job_title": "Full Stack Engineer",
            "company": "NextGen Labs",
            "raw_desc": "React + Flask stack, build dashboards and internal tools.",
            "location_type": "hybrid",
            "posted_at": now - timedelta(hours=10),
            "link": "https://example.com/job/full-stack-engineer",
            "sector": "IT"
        },
        {
            "job_title": "Data Engineer",
            "company": "DataBridge Analytics",
            "raw_desc": "ETL pipelines, SQL, cloud warehouses.",
            "location_type": "onsite",
            "posted_at": now - timedelta(hours=20),
            "link": "https://example.com/job/data-engineer",
            "sector": "Engineering"
        },
        {
            "job_title": "Healthcare Data Analyst",
            "company": "MediInsight",
            "raw_desc": "Analyze patient data, reporting, dashboards.",
            "location_type": "remote",
            "posted_at": now - timedelta(hours=8),
            "link": "https://example.com/job/healthcare-data-analyst",
            "sector": "Healthcare"
        },
        {
            "job_title": "DevOps Engineer",
            "company": "CloudMatrix",
            "raw_desc": "CI/CD, Kubernetes, monitoring for SaaS products.",
            "location_type": "hybrid",
            "posted_at": now - timedelta(hours=3),
            "link": "https://example.com/job/devops-engineer",
            "sector": "IT"
        },
    ]

    min_expected_salary = prefs.get("min_salary")  # currently unused in dummy data

    for j in sample_raw_jobs:
        # filter by last 24 hours
        if j["posted_at"] < now - LAST_24_HOURS:
            continue

        if sector and sector != "any" and j["sector"].lower() != sector.lower():
            continue

        if location_type and location_type != "any" and j["location_type"].lower() != location_type.lower():
            continue

        jobs.append({
            "Job Title": j["job_title"],
            "Company": j["company"],
            "Description": normalize_description(j["raw_desc"]),
            "Location Type": j["location_type"],
            "Posted At (UTC)": j["posted_at"].strftime("%Y-%m-%d %H:%M"),
            "Job Link": j["link"],
        })

    return jobs

def fetch_remoteok_jobs(prefs):
    """
    Fetch real remote jobs from RemoteOK API.
    Returns up to MAX_REMOTEOK_JOBS (50) jobs matching the preferences.

    NOTE (legal):
    RemoteOK requires that you:
    - Mention RemoteOK as a source
    - Link directly to the RemoteOK job URL (no redirects)
    """

    try:
        resp = requests.get(
            REMOTEOK_API_URL,
            headers={"User-Agent": "JobFetchMVP/1.0 (https://yourdomain.com)"},
            timeout=API_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.Timeout as e:
        logger.error(f"RemoteOK API request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"RemoteOK API request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"RemoteOK fetch error: {e}", exc_info=True)
        return []

    if not isinstance(data, list) or len(data) <= 1:
        return []

    # First element is "legal", rest are jobs
    jobs_raw = data[1:]

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    results = []

    for j in jobs_raw:
        # Limit to first MAX_REMOTEOK_JOBS jobs
        if len(results) >= MAX_REMOTEOK_JOBS:
            break

        date_str = j.get("date")
        try:
            dt = dateparser.parse(date_str) if date_str else None
            # Ensure datetime is timezone-aware (convert to UTC if naive)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
        except Exception:
            dt = None

        position = j.get("position") or ""
        company = j.get("company") or ""
        raw_desc = strip_html(j.get("description") or "")  # Clean HTML from RemoteOK description
        tags = j.get("tags") or []
        url = j.get("url") or ""

        # RemoteOK is remote-only, so:
        # - If user chose "onsite" or "hybrid", we skip; dummy/other scrapers will handle.
        if job_location in ["onsite", "hybrid"]:
            continue

        # Simple sector filter: only IT / Engineering for now (or "any")
        if sector not in ["", "any"]:
            if sector not in ["it", "engineering"]:
                # we only treat RemoteOK as IT/Engineering remote source
                continue

        # Simple skill filter (look into title, tags, description)
        text_blob = (position + " " + raw_desc + " " + " ".join(tags)).lower()
        if skills and not any(s in text_blob for s in skills):
            continue

        results.append({
            "Job Title": position,
            "Company": company,
            "Description": normalize_description(raw_desc),
            "Location Type": "remote",  # RemoteOK is remote
            "Posted At (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
            "Job Link": url,        # DIRECT RemoteOK link
        })

    return results

# Wellfound API is no longer available (returns 404)
# Function removed - API endpoint no longer exists

def fetch_jobicy_jobs(prefs):
    """
    Fetch remote jobs from Jobicy API.
    Returns up to 50 jobs matching preferences.
    """
    url = "https://jobicy.com/api/v2/remote-jobs"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LetsJobifyBot/1.0; +https://letsjobify.com)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout as e:
        logger.error(f"Jobicy API request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"Jobicy API request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Jobicy fetch error: {e}", exc_info=True)
        return []

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    # Jobicy is remote-only
    if job_location not in ["remote", "any"]:
        return []

    results = []
    jobs_list = data.get("jobs", [])[:50]  # Limit to 50

    for j in jobs_list:
        title = j.get("jobTitle") or ""
        company = j.get("companyName") or ""
        raw_desc = strip_html(j.get("jobDescription", ""))
        url = j.get("url") or ""

        if not title or not company:
            continue

        # Skill match filter
        text_blob = (title + " " + raw_desc).lower()
        if skills and not any(s in text_blob for s in skills):
            continue

        # Sector filter
        if sector not in ["", "any"]:
            if sector not in ["it", "engineering"]:
                continue

        results.append({
            "Job Title": title,
            "Company": company,
            "Description": normalize_description(raw_desc),
            "Location Type": "remote",
            "Posted At (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "Job Link": url,
        })

    return results

def fetch_remotive_jobs(prefs):
    """
    Fetch remote jobs from Remotive API.
    Returns up to 50 jobs matching preferences.
    """
    url = "https://remotive.com/api/remote-jobs"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LetsJobifyBot/1.0; +https://letsjobify.com)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout as e:
        logger.error(f"Remotive API request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"Remotive API request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Remotive fetch error: {e}", exc_info=True)
        return []

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    # Remotive is remote-only
    if job_location not in ["remote", "any"]:
        return []

    results = []
    jobs_list = data.get("jobs", [])[:50]  # Limit to 50

    for j in jobs_list:
        title = j.get("title") or ""
        company = j.get("company_name") or ""
        raw_desc = strip_html(j.get("description", ""))
        url = j.get("url") or ""
        publication_date = j.get("publication_date")

        if not title or not company:
            continue

        # Parse date
        dt = None
        if publication_date:
            try:
                dt = dateparser.parse(publication_date)
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
            except Exception:
                dt = None

        # Skill match filter
        text_blob = (title + " " + raw_desc).lower()
        if skills and not any(s in text_blob for s in skills):
            continue

        # Sector filter
        if sector not in ["", "any"]:
            if sector not in ["it", "engineering"]:
                continue

        results.append({
            "Job Title": title,
            "Company": company,
            "Description": normalize_description(raw_desc),
            "Location Type": "remote",
            "Posted At (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
            "Job Link": url,
        })

    return results

def fetch_weworkremotely_jobs(prefs):
    """
    Fetch remote jobs from WeWorkRemotely RSS feed.
    Returns up to 50 jobs matching preferences.
    """
    url = "https://weworkremotely.com/remote-jobs.rss"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LetsJobifyBot/1.0; +https://letsjobify.com)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except requests.Timeout as e:
        logger.error(f"WeWorkRemotely RSS request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"WeWorkRemotely RSS request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"WeWorkRemotely fetch error: {e}", exc_info=True)
        return []

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    # WeWorkRemotely is remote-only
    if job_location not in ["remote", "any"]:
        return []

    results = []
    entries = feed.entries[:50]  # Limit to 50

    for entry in entries:
        title = entry.get("title", "").strip()
        company = entry.get("author", "").strip()
        raw_desc = strip_html(entry.get("summary", ""))
        url = entry.get("link", "")
        published = entry.get("published")

        if not title:
            continue

        # Parse date
        dt = None
        if published:
            try:
                dt = dateparser.parse(published)
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
            except Exception:
                dt = None

        # Skill match filter
        text_blob = (title + " " + raw_desc).lower()
        if skills and not any(s in text_blob for s in skills):
            continue

        # Sector filter
        if sector not in ["", "any"]:
            if sector not in ["it", "engineering"]:
                continue

        results.append({
            "Job Title": title,
            "Company": company or "Unknown",
            "Description": normalize_description(raw_desc),
            "Location Type": "remote",
            "Posted At (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
            "Job Link": url,
        })

    return results

def fetch_eu_remote_jobs(prefs):
    """
    Fetch remote jobs from EU Remote Jobs RSS feed.
    Returns up to 50 jobs matching preferences.
    """
    url = "https://euremotejobs.com/feed/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LetsJobifyBot/1.0; +https://letsjobify.com)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except requests.Timeout as e:
        logger.error(f"EU Remote Jobs RSS request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"EU Remote Jobs RSS request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"EU Remote Jobs fetch error: {e}", exc_info=True)
        return []

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    # EU Remote Jobs is remote-only
    if job_location not in ["remote", "any"]:
        return []

    results = []
    entries = feed.entries[:50]  # Limit to 50

    for entry in entries:
        title = entry.get("title", "").strip()
        company = entry.get("author", "").strip()
        raw_desc = strip_html(entry.get("summary", ""))
        url = entry.get("link", "")
        published = entry.get("published")

        if not title:
            continue

        # Parse date
        dt = None
        if published:
            try:
                dt = dateparser.parse(published)
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
            except Exception:
                dt = None

        # Skill match filter
        text_blob = (title + " " + raw_desc).lower()
        if skills and not any(s in text_blob for s in skills):
            continue

        # Sector filter
        if sector not in ["", "any"]:
            if sector not in ["it", "engineering"]:
                continue

        results.append({
            "Job Title": title,
            "Company": company or "Unknown",
            "Description": normalize_description(raw_desc),
            "Location Type": "remote",
            "Posted At (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
            "Job Link": url,
        })

    return results

def fetch_himalayas_jobs(prefs):
    """
    Fetch remote jobs from Himalayas.app by scraping.
    Returns up to 50 jobs matching preferences.
    Note: Web scraping may be fragile if site structure changes.
    """
    url = "https://himalayas.app/jobs?remote=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except requests.Timeout as e:
        logger.error(f"Himalayas request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"Himalayas request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Himalayas fetch error: {e}", exc_info=True)
        return []

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    # Himalayas is remote-only
    if job_location not in ["remote", "any"]:
        return []

    results = []
    try:
        cards = soup.select(".job-card")[:50]  # Limit to 50
    except Exception as e:
        logger.error(f"Himalayas parsing error: {e}", exc_info=True)
        return []

    for c in cards:
        try:
            title_elem = c.select_one(".title")
            company_elem = c.select_one(".company")
            link_elem = c.select_one("a")

            title = title_elem.text.strip() if title_elem else ""
            company = company_elem.text.strip() if company_elem else ""
            link = link_elem.get("href", "") if link_elem else ""

            if not title:
                continue

            url = f"https://himalayas.app{link}" if link.startswith("/") else link

            # Skill match filter
            text_blob = (title + " " + company).lower()
            if skills and not any(s in text_blob for s in skills):
                continue

            # Sector filter
            if sector not in ["", "any"]:
                if sector not in ["it", "engineering"]:
                    continue

            results.append({
                "Job Title": title,
                "Company": company or "Unknown",
                "Description": "See original job post for full role details.",
                "Location Type": "remote",
                "Posted At (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "Job Link": url,
            })
        except Exception as e:
            logger.debug(f"Error processing Himalayas job card: {e}")
            continue

    return results

def fetch_remote_co_jobs(prefs):
    """
    Fetch remote jobs from Remote.co by scraping.
    Returns up to 50 jobs matching preferences.
    Note: Web scraping may be fragile if site structure changes.
    """
    url = "https://remote.co/remote-jobs/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except requests.Timeout as e:
        logger.error(f"Remote.co request timeout: {e}", exc_info=True)
        return []
    except requests.RequestException as e:
        logger.error(f"Remote.co request error: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Remote.co fetch error: {e}", exc_info=True)
        return []

    skills = [s.strip().lower() for s in prefs.get("skills", "").split(",") if s.strip()]
    sector = (prefs.get("sector") or "").lower()
    job_location = (prefs.get("job_location") or "remote").lower()

    # Remote.co is remote-only
    if job_location not in ["remote", "any"]:
        return []

    results = []
    try:
        jobs = soup.select("div.job_listing")[:50]  # Limit to 50
    except Exception as e:
        logger.error(f"Remote.co parsing error: {e}", exc_info=True)
        return []

    for j in jobs:
        try:
            title_elem = j.select_one("a div")
            link_elem = j.select_one("a")

            title = title_elem.text.strip() if title_elem else ""
            link = link_elem.get("href", "") if link_elem else ""

            if not title:
                continue

            url = link if link.startswith("http") else f"https://remote.co{link}"

            # Skill match filter
            text_blob = title.lower()
            if skills and not any(s in text_blob for s in skills):
                continue

            # Sector filter
            if sector not in ["", "any"]:
                if sector not in ["it", "engineering"]:
                    continue

            results.append({
                "Job Title": title,
                "Company": "Unknown",
                "Description": "See original job post for full role details.",
                "Location Type": "remote",
                "Posted At (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "Job Link": url,
            })
        except Exception as e:
            logger.debug(f"Error processing Remote.co job listing: {e}")
            continue

    return results

def fetch_all_jobs(prefs):
    """
    Combine multiple job sources:
      - RemoteOK (global remote IT/Engineering)
      - Jobicy (remote jobs)
      - Remotive (remote jobs)
      - WeWorkRemotely (remote jobs RSS)
      - EU Remote Jobs (remote jobs RSS)
      - Dummy/placeholder jobs (for India/local/onsite/hybrid) as fallback
    """
    jobs = []

    # 1. Try RemoteOK (global remote IT/Engineering)
    try:
        remote_jobs = fetch_remoteok_jobs(prefs)
        jobs.extend(remote_jobs)
        logger.info(f"Fetched {len(remote_jobs)} jobs from RemoteOK")
    except Exception as e:
        logger.error(f"Error in fetch_remoteok_jobs: {e}", exc_info=True)

    # 2. Try Jobicy (remote jobs API)
    try:
        jobicy_jobs = fetch_jobicy_jobs(prefs)
        jobs.extend(jobicy_jobs)
        logger.info(f"Fetched {len(jobicy_jobs)} jobs from Jobicy")
    except Exception as e:
        logger.error(f"Error in fetch_jobicy_jobs: {e}", exc_info=True)

    # 3. Try Remotive (remote jobs API)
    try:
        remotive_jobs = fetch_remotive_jobs(prefs)
        jobs.extend(remotive_jobs)
        logger.info(f"Fetched {len(remotive_jobs)} jobs from Remotive")
    except Exception as e:
        logger.error(f"Error in fetch_remotive_jobs: {e}", exc_info=True)

    # 4. Try WeWorkRemotely (RSS feed)
    try:
        wwr_jobs = fetch_weworkremotely_jobs(prefs)
        jobs.extend(wwr_jobs)
        logger.info(f"Fetched {len(wwr_jobs)} jobs from WeWorkRemotely")
    except Exception as e:
        logger.error(f"Error in fetch_weworkremotely_jobs: {e}", exc_info=True)

    # 5. Try EU Remote Jobs (RSS feed)
    try:
        eu_jobs = fetch_eu_remote_jobs(prefs)
        jobs.extend(eu_jobs)
        logger.info(f"Fetched {len(eu_jobs)} jobs from EU Remote Jobs")
    except Exception as e:
        logger.error(f"Error in fetch_eu_remote_jobs: {e}", exc_info=True)

    # 6. Try Himalayas (web scraping)
    try:
        himalayas_jobs = fetch_himalayas_jobs(prefs)
        jobs.extend(himalayas_jobs)
        logger.info(f"Fetched {len(himalayas_jobs)} jobs from Himalayas")
    except Exception as e:
        logger.error(f"Error in fetch_himalayas_jobs: {e}", exc_info=True)

    # 7. Try Remote.co (web scraping)
    try:
        remote_co_jobs = fetch_remote_co_jobs(prefs)
        jobs.extend(remote_co_jobs)
        logger.info(f"Fetched {len(remote_co_jobs)} jobs from Remote.co")
    except Exception as e:
        logger.error(f"Error in fetch_remote_co_jobs: {e}", exc_info=True)

    # 8. For onsite/hybrid / India-specific, keep dummy for now
    #    or if other sources returned nothing, still give user something.
    if not jobs or (prefs.get("job_location") in ["onsite", "hybrid"]):
        try:
            dummy_jobs = dummy_scrape_jobs(prefs)
            jobs.extend(dummy_jobs)
            logger.info(f"Fetched {len(dummy_jobs)} dummy jobs")
        except Exception as e:
            logger.error(f"Error in dummy_scrape_jobs: {e}", exc_info=True)

    return jobs


def generate_excel(jobs):
    """
    Take a list of dict jobs and generate an Excel file.
    Returns file_id (UUID string).
    """
    if not jobs:
        # still generate an empty file with headers
        jobs = [{
            "Job Title": "",
            "Company": "",
            "Description": "",
            "Location Type": "",
            "Posted At (UTC)": "",
            "Job Link": "",
        }]

    df = pd.DataFrame(jobs)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}.xlsx"
    filepath = os.path.join(GENERATED_DIR, filename)

    df.to_excel(filepath, index=False)

    return file_id


# ------------------ ROUTES ------------------

@app.route("/", methods=["GET"])
def index():
    base_url = request.url_root.rstrip('/')
    sample_jobs = PRELOADED_JOBS[:8]
    return render_template(
        "index.html",
        base_url=base_url,
        sample_jobs=sample_jobs,
    )


@app.route("/favicon.ico")
def favicon():
    """Serve favicon from static folder"""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.png', mimetype='image/png')


@app.route("/privacy-policy", methods=["GET"])
def privacy_policy():
    """Privacy policy page"""
    base_url = request.url_root.rstrip('/')
    return render_template("privacy_policy.html", base_url=base_url)

@app.route("/blog", methods=["GET"])
def blog():
    """Blog landing page"""
    base_url = request.url_root.rstrip('/')
    return render_template("blog.html", base_url=base_url)


@app.route("/sitemap.xml", methods=["GET"])
def sitemap():
    """Generate sitemap.xml for SEO"""
    base_url = request.url_root.rstrip('/')
    sitemap_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{base_url}/</loc>
    <lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{base_url}/blog</loc>
    <lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.7</priority>
  </url>
  <url>
    <loc>{base_url}/privacy-policy</loc>
    <lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>"""
    return Response(sitemap_content, mimetype='application/xml')


@app.route("/robots.txt", methods=["GET"])
def robots():
    """Generate robots.txt for search engines"""
    base_url = request.url_root.rstrip('/')
    robots_content = f"""User-agent: *
Allow: /
Disallow: /admin
Disallow: /login
Disallow: /dashboard
Disallow: /download/
Disallow: /generated/
Disallow: /generate_jobs

Sitemap: {base_url}/sitemap.xml
"""
    return Response(robots_content, mimetype='text/plain')


@app.route("/llms.txt", methods=["GET"])
def llms_txt():
    """Serve llms.txt file for LLMs"""
    llms_path = os.path.join(app.root_path, 'llms.txt')
    if os.path.exists(llms_path):
        with open(llms_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content, mimetype='text/plain')
    else:
        abort(404)


def validate_preferences(data):
    """
    Validate and sanitize user preferences.
    Returns tuple: (is_valid, error_message, sanitized_prefs)
    """
    job_location = (data.get("job_location") or "").strip()
    sector = (data.get("sector") or "").strip()
    skills = (data.get("skills") or "").strip()
    min_salary = data.get("min_salary")

    # Validate required fields
    if not job_location:
        return False, "job_location is required", None
    if not sector:
        return False, "sector is required", None
    if not skills:
        return False, "skills are required", None

    # Validate job_location
    valid_locations = ["remote", "onsite", "hybrid", "any"]
    if job_location.lower() not in valid_locations:
        return False, f"job_location must be one of: {', '.join(valid_locations)}", None

    # Validate sector (allow any custom sector for flexibility)
    if len(sector) > 100:
        return False, "sector must be less than 100 characters", None

    # Validate skills
    if len(skills) > 500:
        return False, "skills must be less than 500 characters", None

    # Validate min_salary
    if min_salary is not None:
        try:
            min_salary = int(min_salary)
            if min_salary < 0:
                return False, "min_salary must be non-negative", None
            if min_salary > 100000000:  # Reasonable upper limit
                return False, "min_salary is too high", None
        except (ValueError, TypeError):
            return False, "min_salary must be a valid number", None

    return True, None, {
        "job_location": job_location,
        "sector": sector,
        "skills": skills,
        "min_salary": min_salary,
    }


@app.route("/generate_jobs", methods=["POST"])
def generate_jobs():
    """
    Generate job matches without payment.
    """
    data = request.json or {}

    is_valid, error_msg, prefs = validate_preferences(data)
    if not is_valid:
        app.logger.warning(f"Invalid preferences provided: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400

    try:
        app.logger.info("Starting job fetch for free request")
        jobs = fetch_all_jobs(prefs)
        file_id = generate_excel(jobs)
        download_url = f"/download/{file_id}"
        return jsonify({"success": True, "download_url": download_url})
    except requests.RequestException as e:
        app.logger.error(f"Network error fetching jobs: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to fetch job listings from external sources. Please try again later."
        }), 500
    except pd.errors.ExcelWriterError as e:
        app.logger.error(f"Excel generation error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to generate Excel file. Please contact support."
        }), 500
    except Exception as e:
        app.logger.error(f"Unexpected error generating jobs: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while generating your job listings. Please contact support."
        }), 500



@app.route("/download/<file_id>", methods=["GET"])
def download_file(file_id):
    """
    Download the generated Excel file.
    Validates file_id to prevent path traversal attacks.
    """
    # Validate file_id format (UUID)
    if not UUID_PATTERN.match(file_id):
        app.logger.warning(f"Invalid file_id format attempted: {file_id}")
        abort(400, description="Invalid file ID format")
    
    filename = f"{file_id}.xlsx"
    filepath = os.path.join(GENERATED_DIR, filename)
    
    # Additional security: ensure the resolved path is within GENERATED_DIR
    # This prevents directory traversal attacks
    resolved_path = os.path.abspath(filepath)
    resolved_dir = os.path.abspath(GENERATED_DIR)
    
    if not resolved_path.startswith(resolved_dir):
        app.logger.warning(f"Path traversal attempt detected: {file_id}")
        abort(400, description="Invalid file path")
    
    if not os.path.exists(filepath):
        app.logger.warning(f"File not found: {file_id}")
        abort(404, description="File not found")
    
    app.logger.info(f"File downloaded: {file_id}")
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # For development only
    app.run(debug=True)
