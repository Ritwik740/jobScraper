import os
import uuid
import re
import html
import logging
from datetime import datetime, timedelta, timezone

import mysql.connector
from mysql.connector import Error as MySQLError
import requests
from dateutil import parser as dateparser
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    abort,
    Response,
)
import razorpay
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------

load_dotenv()

# Constants
REMOTEOK_API_URL = "https://remoteok.com/api"
LAST_24_HOURS = timedelta(hours=24)
DEFAULT_PAYMENT_AMOUNT = 10000  # in paise
API_TIMEOUT_SECONDS = 30  # Timeout for external API calls
MAX_REMOTEOK_JOBS = 50  # Maximum number of jobs to fetch from RemoteOK
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# Environment variables
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
PAYMENT_AMOUNT = int(os.getenv("PAYMENT_AMOUNT", str(DEFAULT_PAYMENT_AMOUNT)))  # in paise
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")

if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET and GEMINI_API_KEY):
    raise RuntimeError("Please set RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, GEMINI_API_KEY in .env")

if not all([DB_USER, DB_NAME, DB_PASS, DB_HOST]):
    raise RuntimeError("Please set DB_USER, DB_NAME, DB_PASS, DB_HOST in .env for MySQL connectivity")

genai.configure(api_key=GEMINI_API_KEY)

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

# Razorpay client
rz_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# In-memory store for preferences and generated file mapping
# order_id -> {"prefs": {...}, "file_id": str or None, "created_at": datetime}
ORDERS = {}

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


# ------------------ DB UTILS ------------------

DB_CONFIG = {
    "user": DB_USER,
    "password": DB_PASS,
    "host": DB_HOST,
    "database": DB_NAME,
    "autocommit": True,
}


def get_db_connection():
    """
    Get a new MySQL connection using environment configuration.
    """
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except MySQLError as e:
        logger.error(f"Error connecting to MySQL: {e}", exc_info=True)
        raise


def init_db():
    """
    Create required tables if they do not exist.
    Currently manages the 'transactions' table used to store payment records.
    """
    create_transactions_table_sql = """
        CREATE TABLE IF NOT EXISTS transactions (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            razorpay_order_id VARCHAR(64) NOT NULL,
            razorpay_payment_id VARCHAR(64) NOT NULL,
            amount INT NOT NULL,
            currency VARCHAR(8) NOT NULL,
            status VARCHAR(32) NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(create_transactions_table_sql)
    except MySQLError as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()

def save_transaction(razorpay_order_id, razorpay_payment_id, amount, currency, status):
    """
    Persist a payment transaction record in the database.
    """
    sql = """
        INSERT INTO transactions (
            razorpay_order_id,
            razorpay_payment_id,
            amount,
            currency,
            status
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    params = (
        razorpay_order_id,
        razorpay_payment_id,
        amount,
        currency,
        status,
    )

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
    except MySQLError as e:
        logger.error(
            f"Error saving transaction {razorpay_payment_id} for order {razorpay_order_id}: {e}",
            exc_info=True,
        )
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


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


def gemini_enhance_description(job_title, company, raw_desc, skills, sector):
    """
    Use Gemini to produce a short, user-friendly job description and optional match score.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        You are helping a job seeker.

        Job Title: {job_title}
        Company: {company}
        Sector: {sector}
        Raw Description: {raw_desc}
        Candidate Skills: {', '.join(skills)}

        1) Rewrite a concise 2–3 line description suited for a candidate.
        2) Give a relevance score from 0 to 100 based on how well the job matches the skills.

        Respond in this exact format:
        DESCRIPTION: <your short description>
        SCORE: <number>
        """
        res = model.generate_content(prompt)
        text = (res.text or "").strip()

        desc = ""
        score = 0

        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("DESCRIPTION:"):
                desc = line.split("DESCRIPTION:", 1)[1].strip()
            elif line.upper().startswith("SCORE:"):
                try:
                    score = int(line.split("SCORE:", 1)[1].strip())
                except ValueError:
                    score = 0

        if not desc:
            desc = raw_desc
        
        # Strip HTML from the description
        desc = strip_html(desc)

        return desc, score
    except Exception as e:
        logger.error(f"Gemini API error: {e}", exc_info=True)
        return raw_desc, 0


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

        enhanced_desc, score = gemini_enhance_description(
            job_title=j["job_title"],
            company=j["company"],
            raw_desc=j["raw_desc"],
            skills=skills,
            sector=sector,
        )

        jobs.append({
            "Job Title": j["job_title"],
            "Company": j["company"],
            "Description": enhanced_desc,
            "Location Type": j["location_type"],
            "Posted At (UTC)": j["posted_at"].strftime("%Y-%m-%d %H:%M"),
            "Job Link": j["link"],
            "Match Score": score,
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

        enhanced_desc, score = gemini_enhance_description(
            job_title=position,
            company=company,
            raw_desc=raw_desc,
            skills=skills,
            sector=sector or "IT",
        )

        results.append({
            "Job Title": position,
            "Company": company,
            "Description": enhanced_desc,
            "Location Type": "remote",  # RemoteOK is remote
            "Posted At (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt else "",
            "Job Link": url,        # DIRECT RemoteOK link
            "Match Score": score,
        })

    return results

def fetch_all_jobs(prefs):
    """
    Combine:
      - Real global remote jobs from RemoteOK
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

    # 2. For onsite/hybrid / India-specific, keep dummy for now
    #    or if RemoteOK returned nothing, still give user something.
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
            "Match Score": "",
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
    return render_template(
        "index.html",
        razorpay_key_id=RAZORPAY_KEY_ID,
        payment_amount=PAYMENT_AMOUNT,
        base_url=base_url
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
Allow: /privacy-policy
Disallow: /download/
Disallow: /generated/
Disallow: /verify_payment
Disallow: /create_order

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


@app.route("/create_order", methods=["POST"])
def create_order():
    """
    Step 1: Receive job preferences + create Razorpay order.
    """
    data = request.json or {}

    # Validate preferences
    is_valid, error_msg, prefs = validate_preferences(data)
    if not is_valid:
        app.logger.warning(f"Invalid preferences provided: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 400

    # Prepare order
    order_data = {
        "amount": PAYMENT_AMOUNT,
        "currency": "INR",
        "payment_capture": 1,
    }

    try:
        order = rz_client.order.create(data=order_data)
        order_id = order["id"]
    except Exception as e:
        app.logger.error(f"Error creating Razorpay order: {e}", exc_info=True)
        return jsonify(
            {
                "success": False,
                "error": "Failed to create payment order. Please try again.",
            }
        ), 500

    # Store preferences in memory mapped to order_id
    ORDERS[order_id] = {
        "prefs": prefs,
        "file_id": None,
        "created_at": datetime.now(timezone.utc),
    }

    app.logger.info(
        f"Created order {order_id} for preferences: {prefs['sector']}, {prefs['job_location']}"
    )

    return jsonify(
        {
            "order_id": order_id,
            "amount": PAYMENT_AMOUNT,
            "currency": "INR",
        }
    )


@app.route("/verify_payment", methods=["POST"])
def verify_payment():
    """
    Step 2: Frontend sends payment_id + order_id + signature for verification.
    If valid → run scraper → generate Excel → return download URL.
    """
    data = request.json or {}

    razorpay_payment_id = data.get("razorpay_payment_id")
    razorpay_order_id = data.get("razorpay_order_id")
    razorpay_signature = data.get("razorpay_signature")

    if not all([razorpay_payment_id, razorpay_order_id, razorpay_signature]):
        return jsonify({"success": False, "error": "Missing payment fields"}), 400

    # Verify signature
    params_dict = {
        'razorpay_order_id': razorpay_order_id,
        'razorpay_payment_id': razorpay_payment_id,
        'razorpay_signature': razorpay_signature
    }

    try:
        rz_client.utility.verify_payment_signature(params_dict)
    except razorpay.errors.SignatureVerificationError as e:
        app.logger.warning(f"Payment verification failed for order {razorpay_order_id}: {e}")
        return jsonify({"success": False, "error": "Payment verification failed"}), 400
    except Exception as e:
        app.logger.error(f"Error verifying payment signature: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Payment verification error"}), 500

    # Persist successful transaction details in DB (best-effort, do not block flow)
    try:
        save_transaction(
            razorpay_order_id=razorpay_order_id,
            razorpay_payment_id=razorpay_payment_id,
            amount=PAYMENT_AMOUNT,
            currency="INR",
            status="success",
        )
    except MySQLError:
        # Log and continue; we don't want to break the user flow if logging fails
        app.logger.error(
            f"Failed to persist transaction for order {razorpay_order_id}",
            exc_info=True,
        )

    # Fetch preferences from in-memory store
    order_data = ORDERS.get(razorpay_order_id)
    if not order_data:
        return jsonify({"success": False, "error": "Order not found"}), 404

    prefs = order_data["prefs"]

    # Scrape jobs + generate Excel with comprehensive error handling
    try:
        app.logger.info(f"Starting job fetch for order {razorpay_order_id}")
        jobs = fetch_all_jobs(prefs)
        
        if not jobs:
            app.logger.warning(f"No jobs found for order {razorpay_order_id} with preferences: {prefs}")
            # Still generate an empty file so user gets something
        
        app.logger.info(f"Fetched {len(jobs)} jobs, generating Excel file...")
        file_id = generate_excel(jobs)

        # Store file_id back in memory
        order_data["file_id"] = file_id

        app.logger.info(
            f"Successfully generated file {file_id} for order {razorpay_order_id}"
        )

        download_url = f"/download/{file_id}"

        return jsonify({"success": True, "download_url": download_url})
    except requests.RequestException as e:
        app.logger.error(f"Network error fetching jobs for order {razorpay_order_id}: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to fetch job listings from external sources. Please try again later."
        }), 500
    except pd.errors.ExcelWriterError as e:
        app.logger.error(f"Excel generation error for order {razorpay_order_id}: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to generate Excel file. Please contact support."
        }), 500
    except Exception as e:
        app.logger.error(f"Unexpected error generating jobs for order {razorpay_order_id}: {e}", exc_info=True)
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
    # Initialize DB schema (creates tables if they don't exist)
    init_db()


    app.run()
