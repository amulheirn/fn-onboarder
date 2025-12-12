from __future__ import annotations

import csv
import io
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")

DEFAULT_BASE_URL = os.environ.get("DEFAULT_BASE_URL", "https://fwd.app")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "20"))
NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", "fn-onboarder/1.0 (https://fwd.app)")

COUNTRY_NAME_TO_CODE = {
    "GERMANY": "DE",
    "DEUTSCHLAND": "DE",
    "FRANCE": "FR",
    "SPAIN": "ES",
    "ITALY": "IT",
    "UNITED STATES": "US",
    "USA": "US",
    "UNITED STATES OF AMERICA": "US",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "GREAT BRITAIN": "GB",
    "ENGLAND": "GB",
    "WALES": "GB",
    "SCOTLAND": "GB",
    "AUSTRIA": "AT",
    "BELGIUM": "BE",
    "BULGARIA": "BG",
    "CROATIA": "HR",
    "CYPRUS": "CY",
    "CZECHIA": "CZ",
    "CZECH REPUBLIC": "CZ",
    "DENMARK": "DK",
    "ESTONIA": "EE",
    "FINLAND": "FI",
    "GREECE": "GR",
    "HUNGARY": "HU",
    "ICELAND": "IS",
    "IRELAND": "IE",
    "LITHUANIA": "LT",
    "LATVIA": "LV",
    "LUXEMBOURG": "LU",
    "MALTA": "MT",
    "NETHERLANDS": "NL",
    "HOLLAND": "NL",
    "NORWAY": "NO",
    "POLAND": "PL",
    "PORTUGAL": "PT",
    "ROMANIA": "RO",
    "SLOVAKIA": "SK",
    "SLOVENIA": "SI",
    "SWEDEN": "SE",
    "SWITZERLAND": "CH",
    "SUISSE": "CH",
    "SCHWEIZ": "CH",
}

ALLOWED_COUNTRY_CODES: Set[str] = set(COUNTRY_NAME_TO_CODE.values())
# Explicitly allow the codes used in the default-country dropdown
ALLOWED_COUNTRY_CODES.update(
    {
        "AT",
        "BE",
        "BG",
        "CH",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GB",
        "GR",
        "HR",
        "HU",
        "IE",
        "IS",
        "IT",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "NO",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
        "US",
    }
)

ISO3_TO_ISO2 = {
    "DEU": "DE",
    "FRA": "FR",
    "ESP": "ES",
    "ITA": "IT",
    "GBR": "GB",
    "USA": "US",
    "AUS": "AU",
    "CAN": "CA",
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "HRV": "HR",
    "CYP": "CY",
    "CZE": "CZ",
    "DNK": "DK",
    "EST": "EE",
    "FIN": "FI",
    "GRC": "GR",
    "HUN": "HU",
    "ISL": "IS",
    "IRL": "IE",
    "LTU": "LT",
    "LVA": "LV",
    "LUX": "LU",
    "MLT": "MT",
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
    "CHE": "CH",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_config() -> Dict[str, str]:
    """Return config from the signed session cookie."""
    return {
        "base_url": session.get("base_url") or DEFAULT_BASE_URL,
        "api_key": session.get("api_key") or "",
        "api_secret": session.get("api_secret") or "",
        "network_id": session.get("network_id") or "",
        "network_name": session.get("network_name") or "",
        "verify_ssl": session.get("verify_ssl", True),
        "collector_online": bool(session.get("collector_online", False)),
    }


def save_config(form_data: Dict[str, str]) -> None:
    session["base_url"] = form_data.get("base_url") or DEFAULT_BASE_URL
    session["api_key"] = (form_data.get("api_key") or "").strip()
    session["api_secret"] = (form_data.get("api_secret") or "").strip()
    session["network_id"] = (form_data.get("network_id") or session.get("network_id") or "").strip()
    session["network_name"] = (form_data.get("network_name") or session.get("network_name") or "").strip()
    session["collector_online"] = False
    disable_ssl = parse_bool(form_data.get("disable_ssl_verify"))
    session["verify_ssl"] = False if disable_ssl else True
    session.permanent = False


def build_auth(config: Dict[str, str]) -> Optional[Tuple[str, str]]:
    if config.get("api_key") and config.get("api_secret"):
        return (config["api_key"], config["api_secret"])
    return None


def api_request(
    method: str, path: str, config: Dict[str, str], *, json_body: Optional[Dict[str, Any]] = None
) -> requests.Response:
    base_url = (config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}{path}"
    resp = requests.request(
        method=method.upper(),
        url=url,
        json=json_body,
        auth=build_auth(config),
        verify=bool(config.get("verify_ssl", True)),
        timeout=REQUEST_TIMEOUT,
    )
    return resp


def update_network_name(config: Dict[str, str]) -> Optional[str]:
    """Attempt to resolve and store a human-readable network name."""
    network_id = (config.get("network_id") or "").strip()
    if not network_id:
        session["network_name"] = ""
        return None
    if not (config.get("api_key") and config.get("api_secret")):
        session["network_name"] = ""
        return None

    # First try direct lookup
    try:
        resp = api_request("get", f"/api/networks/{network_id}", config)
        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
                name = data.get("name") or data.get("displayName") or data.get("id")
                session["network_name"] = name or ""
                return name
            except ValueError:
                pass
    except requests.RequestException:
        session["network_name"] = ""
        return None

    # Fallback: list networks and match id
    try:
        resp = api_request("get", "/api/networks", config)
        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and str(item.get("id")) == network_id:
                            name = item.get("name") or item.get("displayName") or item.get("id")
                            session["network_name"] = name or ""
                            return name
            except ValueError:
                pass
    except requests.RequestException:
        session["network_name"] = ""
        return None

    session["network_name"] = ""
    return None


def update_collector_status(config: Dict[str, str]) -> bool:
    """Check collector status and store the flag in session."""
    network_id = (config.get("network_id") or "").strip()
    if not network_id or not (config.get("api_key") and config.get("api_secret")):
        session["collector_online"] = False
        session["collector_status"] = ""
        return False

    try:
        resp = api_request("get", f"/api/networks/{network_id}/collector/status", config)
        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
                busy_status = (data.get("busyStatus") or "").upper()
                is_set = True
                if "isSet" in data:
                    is_set = bool(data.get("isSet"))
                session["collector_status"] = busy_status
                online = is_set and busy_status != "OFFLINE"
            except ValueError:
                # Treat non-JSON 2xx as online if body is truthy
                online = True
                session["collector_status"] = ""
            session["collector_online"] = online
            return online
    except requests.RequestException:
        session["collector_online"] = False
        session["collector_status"] = ""
        return False

    session["collector_online"] = False
    session["collector_status"] = ""
    return False


@app.get("/api/networks/list")
def list_networks():
    config = current_config()
    try:
        resp = api_request("get", "/api/networks", config)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        data = resp.json()
    except ValueError:
        data = resp.text
    if 200 <= resp.status_code < 300:
        return jsonify({"ok": True, "networks": data})
    return jsonify({"ok": False, "status": resp.status_code, "response": data}), 400


def fetch_existing_location_ids(config: Dict[str, str]) -> Set[str]:
    network_id = (config.get("network_id") or "").strip()
    if not network_id:
        return set()
    try:
        resp = api_request("get", f"/api/networks/{network_id}/locations", config)
        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
                if isinstance(data, list):
                    return {str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")}
            except ValueError:
                return set()
    except requests.RequestException:
        return set()
    return set()


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_csv_text(csv_text: str, has_headers: bool = True) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    cleaned = csv_text.lstrip("\ufeff").strip()
    if not cleaned:
        return [], [], []

    preview = cleaned[:2048]

    def detect_delimiter(sample: str) -> Optional[str]:
        lines = sample.splitlines()[:5]
        candidates = [",", ";", "\t", "|"]
        best_delim = None
        best_score = -1
        for delim in candidates:
            counts = [line.count(delim) for line in lines if line.strip()]
            if not counts:
                continue
            score = sum(counts)
            if score > best_score:
                best_score = score
                best_delim = delim
        return best_delim if best_score > 0 else None

    detected_delim = detect_delimiter(preview)
    if detected_delim:
        dialect = csv.excel
        dialect.delimiter = detected_delim  # type: ignore[attr-defined]
    else:
        try:
            dialect = csv.Sniffer().sniff(preview)
        except csv.Error:
            dialect = csv.excel

    def sanitize_header(name: str, seen: Set[str], fallback: str) -> str:
        base = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip().lstrip("\ufeff")).strip("_")
        base = base or fallback
        candidate = base
        counter = 1
        while candidate in seen:
            counter += 1
            candidate = f"{base}_{counter}"
        seen.add(candidate)
        return candidate

    if has_headers:
        reader = csv.DictReader(io.StringIO(cleaned), dialect=dialect)
        raw_headers = [h or "" for h in (reader.fieldnames or [])]
        seen: Set[str] = set()
        headers = [sanitize_header(h, seen, f"col{i+1}") for i, h in enumerate(raw_headers)]
        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            if not raw_row or not any((value or "").strip() for value in raw_row.values()):
                continue
            row_dict: Dict[str, str] = {}
            for raw_h, safe_h in zip(raw_headers, headers):
                row_dict[safe_h] = raw_row.get(raw_h, "")
            rows.append(row_dict)
        header_labels = [h.strip().lstrip("\ufeff") or headers[idx] for idx, h in enumerate(raw_headers)]
        return headers, rows, header_labels

    reader = csv.reader(io.StringIO(cleaned), dialect=dialect)
    raw_rows = [row for row in reader]
    if not raw_rows:
        return [], [], []
    headers = [f"col{i+1}" for i in range(len(raw_rows[0]))]
    rows: List[Dict[str, str]] = []
    for raw in raw_rows:
        # Pad/truncate to header length
        padded = list(raw) + [""] * (len(headers) - len(raw))
        row_dict = {h: padded[i] if i < len(padded) else "" for i, h in enumerate(headers)}
        if any((v or "").strip() for v in row_dict.values()):
            rows.append(row_dict)
    return headers, rows, headers


def rows_to_csv(rows: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """Serialize rows to CSV with provided headers (or keys of the first row)."""
    if not rows:
        return ""
    fieldnames = headers if headers else list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({h: row.get(h, "") for h in fieldnames})
    return output.getvalue()


def slugify(text: str, fallback: str = "") -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    trimmed = "-".join(filter(None, safe.split("-")))
    if trimmed:
        return trimmed[:64]
    return fallback or "site"


def normalize_country_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    code = str(value).strip().upper()
    if not code:
        return None
    if code in COUNTRY_NAME_TO_CODE:
        mapped = COUNTRY_NAME_TO_CODE[code]
        return mapped if mapped in ALLOWED_COUNTRY_CODES else None
    if len(code) == 2 and code.isalpha():
        candidate = "GB" if code == "UK" else code
        return candidate if candidate in ALLOWED_COUNTRY_CODES else None
    if len(code) == 3 and code.isalpha():
        mapped = ISO3_TO_ISO2.get(code)
        if mapped and mapped in ALLOWED_COUNTRY_CODES:
            return mapped
    return None


def geocode_address(address: str, country_hint: Optional[str]) -> Tuple[float, float]:
    query = (address or "").strip()
    if not query:
        raise ValueError("Address is empty")
    params = {"q": query, "format": "jsonv2", "limit": 1}
    if country_hint:
        params["countrycodes"] = country_hint.lower()
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise ValueError(f"Geocoding request failed: {exc}")

    if not 200 <= resp.status_code < 300:
        raise ValueError(f"Geocoding failed (HTTP {resp.status_code})")

    try:
        data = resp.json()
    except ValueError:
        raise ValueError("Geocoding response was not valid JSON")

    if not isinstance(data, list) or not data:
        raise ValueError(f"No geocoding result for address '{address}'")

    first = data[0] or {}
    try:
        lat_val = float(first.get("lat"))
        lng_val = float(first.get("lon"))
    except (TypeError, ValueError, KeyError):
        raise ValueError("Geocoding result missing lat/lon")

    return round(lat_val, 6), round(lng_val, 6)


def prepare_location_payloads(
    csv_text: str,
    mapping: Dict[str, Optional[str]],
    auto_site_code: bool = True,
    existing_ids: Optional[Set[str]] = None,
    has_headers: bool = True,
    city_overrides: Optional[Set[str]] = None,
    default_country: Optional[str] = None,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Dict[str, Any]], int]:
    """Convert CSV rows to location payloads and collect per-row errors."""
    headers, rows, _header_labels = parse_csv_text(csv_text, has_headers)
    payloads: List[Tuple[int, Dict[str, Any]]] = []
    errors: List[Dict[str, Any]] = []
    used_ids: Set[str] = set(existing_ids or set())
    prefix_counters: Dict[str, int] = {}
    override_set: Set[str] = {c.strip().lower() for c in (city_overrides or set()) if c and c.strip()}
    default_country_code = normalize_country_code(default_country)
    seen_rows: Set[Tuple[str, str]] = set()

    def split_prefix_num(value: str) -> Tuple[str, Optional[int]]:
        digits = ""
        for ch in reversed(value):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        if digits:
            try:
                return value[: len(value) - len(digits)], int(digits)
            except ValueError:
                return value, None
        return value, None

    for eid in list(used_ids):
        prefix, num = split_prefix_num(str(eid))
        if num is not None:
            prefix_counters[prefix] = max(prefix_counters.get(prefix, 0), num)

    def derive_prefix(name: str, address: str) -> str:
        source = name or address or "LOC"
        letters = "".join(ch for ch in source if ch.isalpha())
        prefix = (letters[:3] or "LOC").upper()
        if len(prefix) < 3:
            prefix = (prefix + "XXX")[:3]
        # keep prefix short enough to allow numeric suffix under 16 chars total
        return prefix[:12]

    def next_site_code(name: str, address: str) -> str:
        prefix = derive_prefix(name, address)
        seq = prefix_counters.get(prefix, 0) + 1
        candidate = f"{prefix}{seq}"
        # ensure uniqueness and length under 16
        while candidate in used_ids or len(candidate) >= 16:
            seq += 1
            candidate = f"{prefix}{seq}"
        prefix_counters[prefix] = seq
        used_ids.add(candidate)
        return candidate

    def strip_building_marker(addr: str) -> str:
        """Remove building markers like 'Geb.FE10' and anything after for geocoding only."""
        return re.sub(r"\s*Geb\..*$", "", addr, flags=re.IGNORECASE).strip()

    def normalize_address(addr: str) -> str:
        """Normalize supported formats into street, city, country ordering for geocoding."""
        addr = (addr or "").strip()
        if not addr:
            return ""
        addr = strip_building_marker(addr)
        # Already comma-separated; leave as-is
        if "," in addr:
            return addr
        # Tokenize: expected COUNTRY CITY STREET...
        tokens = addr.split()
        if len(tokens) >= 3 and len(tokens[0]) in (2, 3) and tokens[0].isalpha():
            country = tokens[0].strip()
            country_code = normalize_country_code(country)
            if country_code:
                rest_tokens = tokens[1:]
                # Try override matches (longest first)
                city = rest_tokens[0].strip()
                street_tokens = rest_tokens[1:]
                if override_set:
                    sorted_overrides = sorted(override_set, key=lambda x: len(x.split()), reverse=True)
                    for ov in sorted_overrides:
                        ov_parts = ov.split()
                        if len(ov_parts) <= len(rest_tokens) and [p.lower() for p in rest_tokens[: len(ov_parts)]] == ov_parts:
                            city = " ".join(rest_tokens[: len(ov_parts)])
                            street_tokens = rest_tokens[len(ov_parts):]
                            break
                street = " ".join(street_tokens).strip()
                if street:
                    return f"{street}, {city}, {country}"
        # Regex fallback for unicode city names
        match = re.match(r"^\s*(?P<country>[A-Za-z]{2,3})\s+(?P<city>[^,]+?)\s+(?P<street>.+)$", addr, flags=re.UNICODE)
        if match:
            country_raw = match.group("country").strip()
            country_code = normalize_country_code(country_raw)
            if country_code:
                city = match.group("city").strip()
                street = match.group("street").strip()
                return f"{street}, {city}, {country_code}"
        return addr

    def extract_country_code(raw_addr: str, normalized: str) -> Optional[str]:
        """Best-effort extraction of a country code from raw/normalized address."""
        for candidate in [raw_addr, normalized]:
            text = (candidate or "").strip()
            if not text:
                continue
            # Comma-separated: look at last segment
            if "," in text:
                last = text.split(",")[-1].strip()
                token = (last.split()[:1] or [""])[0].upper()
                if len(token) in (2, 3) and token.isalpha():
                    code = normalize_country_code(token)
                    if code:
                        return code
                name = last.upper() or ""
                mapped = normalize_country_code(name)
                if mapped:
                    return mapped
            # Space-separated first token might be country code
            parts = text.split()
            if parts and len(parts[0]) in (2, 3) and parts[0].isalpha():
                code = normalize_country_code(parts[0])
                if code:
                    return code
            # Name lookup
            name = parts[-1].upper() if parts else ""
            mapped = normalize_country_code(name)
            if mapped:
                return mapped
        return None

    for index, row in enumerate(rows, start=1):
        try:
            row_name = (row.get(mapping["name"], "") or "").strip()
            row_address_raw = (row.get(mapping["address"], "") or "").strip()
            row_address = normalize_address(row_address_raw)
            dedupe_key = (row_name.lower(), row_address.lower())
            if dedupe_key in seen_rows:
                errors.append({"row": index, "ok": False, "error": "Duplicate name+address; skipping row"})
                continue
            seen_rows.add(dedupe_key)
            explicit_country = extract_country_code(row_address_raw, row_address)
            country_code = explicit_country or default_country_code
            site_code_value = None
            if mapping.get("siteCode"):
                site_code_value = (row.get(mapping["siteCode"], "") or "").strip()
                if site_code_value and len(site_code_value) >= 16:
                    raise ValueError("site-code must be fewer than 16 characters")

            if not row_name or not row_address:
                raise ValueError("Required fields (name, address) must be present per row")

            if site_code_value:
                final_id = site_code_value
                if final_id in used_ids:
                    errors.append({"row": index, "ok": False, "error": f"Duplicate location id/site-code '{final_id}' found; skipping row"})
                    continue
                used_ids.add(final_id)
            else:
                final_id = next_site_code(row_name, row_address)

            geocode_query = row_address
            if not explicit_country and country_code:
                geocode_query = f"{row_address}, {country_code}"
            lat, lng = geocode_address(geocode_query, country_code)

            location_payload: Dict[str, Any] = {
                "id": final_id,
                # Preserve the original user-provided string for display; normalized version is only for geocoding.
                "name": row_name or row_address_raw or final_id,
                "lat": lat,
                "lng": lng,
            }
            if auto_site_code or site_code_value:
                location_payload["siteCode"] = final_id

            payloads.append((index, location_payload))
        except Exception as exc:
            errors.append({"row": index, "ok": False, "error": str(exc)})

    return payloads, errors, len(rows)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> str:
    return render_template("index.html", config=current_config())


@app.post("/config")
def update_config_route():
    if request.is_json:
        payload = request.get_json(force=True)
        save_config(payload)
        cfg = current_config()
        update_network_name(cfg)
        update_collector_status(cfg)
        return jsonify({"ok": True, "config": current_config()})

    save_config(request.form)
    cfg = current_config()
    update_network_name(cfg)
    update_collector_status(cfg)
    return redirect(url_for("index"))


@app.post("/api/test-version")
def test_version():
    config = current_config()
    update_network_name(config)
    update_collector_status(config)
    try:
        resp = api_request("get", "/api/version", config)
    except requests.RequestException as exc:  # network/timeout errors
        session["last_api_ok"] = False
        return jsonify({"ok": False, "error": str(exc)}), 502

    try:
        body = resp.json()
    except ValueError:
        body = resp.text

    success = 200 <= resp.status_code < 300
    session["last_api_ok"] = success
    return jsonify({
        "ok": success,
        "status": resp.status_code,
        "body": body,
        "url": f"{(config.get('base_url') or DEFAULT_BASE_URL).rstrip('/')}/api/version",
        "collector_online": bool(session.get("collector_online", False)),
        "network_name": session.get("network_name", ""),
    }), (200 if success else 400)


@app.post("/api/collector-status")
def collector_status():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required before checking collectors."}), 400
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/collector/status", config)
    except requests.RequestException as exc:
        session["collector_online"] = False
        session["collector_status"] = ""
        return jsonify({"ok": False, "error": str(exc), "collector_online": False}), 502

    try:
        body = resp.json()
    except ValueError:
        body = resp.text

    busy_status = ""
    is_set = True
    if isinstance(body, dict):
        busy_status = (body.get("busyStatus") or "").upper()
        is_set = bool(body.get("isSet")) if "isSet" in body else True

    online = is_set and busy_status != "OFFLINE" and 200 <= resp.status_code < 300
    session["collector_online"] = online
    session["collector_status"] = busy_status
    return jsonify(
        {
            "ok": online,
            "collector_online": online,
            "busyStatus": busy_status,
            "isSet": is_set,
            "response": body,
            "status": resp.status_code,
        }
    ), (200 if online else 400)


@app.get("/credentials")
def credentials_page():
    return render_template("credentials.html", config=current_config())


@app.post("/api/credentials")
def create_credentials():
    config = current_config()
    update_collector_status(config)
    if not session.get("collector_online"):
        return jsonify({"ok": False, "error": "No collectors online. Check collector status before submitting credentials."}), 400
    data = request.get_json(force=True) if request.is_json else request.form
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400

    mode = (data.get("mode") or "").lower()
    if mode not in {"cli", "http"}:
        return jsonify({"ok": False, "error": "mode must be cli or http"}), 400

    name = (data.get("name") or "").strip()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    if not name or not username or not password:
        return jsonify({"ok": False, "error": "name, username, and password are required"}), 400

    base_payload: Dict[str, Any] = {
        "type": "LOGIN",
        "name": name,
        "username": username,
        "password": password,
    }

    if mode == "cli":
        target_path = f"/api/networks/{config['network_id']}/cli-credentials"
        priv_password_id = (data.get("privilegedModePasswordId") or "").strip() or None
        privilege_level = parse_int(data.get("privilegeLevel"))
        auto_associate = parse_bool(data.get("autoAssociate"))
        if priv_password_id:
            base_payload["privilegedModePasswordId"] = priv_password_id
        if privilege_level is not None:
            base_payload["privilegeLevel"] = privilege_level
        if auto_associate is not None:
            base_payload["autoAssociate"] = auto_associate
    else:
        target_path = f"/api/networks/{config['network_id']}/http-credentials"
        login_type = (data.get("loginType") or "LOCAL").strip().upper() or "LOCAL"
        base_payload["loginType"] = login_type
        auto_associate = parse_bool(data.get("autoAssociate"))
        if auto_associate is not None:
            base_payload["autoAssociate"] = auto_associate

    try:
        resp = api_request("post", target_path, config, json_body=base_payload)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502

    response_body: Any
    try:
        response_body = resp.json()
    except ValueError:
        response_body = resp.text

    if 200 <= resp.status_code < 300:
        return jsonify({"ok": True, "status": resp.status_code, "response": response_body})

    return (
        jsonify(
            {
                "ok": False,
                "status": resp.status_code,
                "response": response_body,
                "error": "API rejected the request",
            }
        ),
        400,
    )


@app.get("/locations")
def locations_page():
    return render_template("locations.html", config=current_config())


@app.post("/api/locations/preview")
def preview_locations():
    data = request.get_json(force=True)
    csv_text = data.get("csvText") or ""
    has_headers = bool(data.get("hasHeaders", True))
    include_full = bool(data.get("full", False))
    headers, rows, header_labels = parse_csv_text(csv_text, has_headers)
    sample_rows = rows[:5]
    resp: Dict[str, Any] = {"headers": headers, "headerLabels": header_labels, "sample": sample_rows}
    if include_full:
        resp["rows"] = rows
    return jsonify(resp)


@app.post("/api/locations/geocode")
def geocode_locations():
    config = current_config()
    payload = request.get_json(force=True)
    csv_text = payload.get("csvText") or ""
    rows_payload = payload.get("rows")
    mapping: Dict[str, Optional[str]] = payload.get("mapping") or {}
    auto_site_code = bool(payload.get("autoGenerateSiteCode", True))
    has_headers = bool(payload.get("hasHeaders", True))
    city_overrides = set(payload.get("cityOverrides") or [])
    default_country = payload.get("defaultCountry")

    required_fields = ["name", "address"]
    missing_fields = [field for field in required_fields if not mapping.get(field)]
    if missing_fields:
        return jsonify({"ok": False, "error": f"Missing mappings for: {', '.join(missing_fields)}"}), 400

    if isinstance(rows_payload, list) and rows_payload:
        normalized_rows: List[Dict[str, Any]] = []
        for item in rows_payload:
            if isinstance(item, dict):
                normalized_rows.append({k: (v if v is not None else "") for k, v in item.items()})
        if normalized_rows:
            csv_text = rows_to_csv(normalized_rows)
            has_headers = True

    existing_ids = fetch_existing_location_ids(config)
    payloads, errors, total = prepare_location_payloads(
        csv_text, mapping, auto_site_code, existing_ids, has_headers, city_overrides, default_country
    )
    locations = [{"row": row_num, "location": loc} for row_num, loc in payloads]
    return jsonify(
        {
            "ok": True,
            "total": total,
            "geocoded": len(payloads),
            "locations": locations,
            "errors": errors,
            "existingIdsChecked": len(existing_ids),
        }
    )


@app.get("/api/locations/list")
def list_locations():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/locations", config)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    if 200 <= resp.status_code < 300:
        return jsonify({"ok": True, "locations": data})
    return jsonify({"ok": False, "status": resp.status_code, "response": data}), 400


@app.post("/api/locations")
def upload_locations():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400

    payload = request.get_json(force=True)
    csv_text = payload.get("csvText") or ""
    rows_payload = payload.get("rows")
    mapping: Dict[str, Optional[str]] = payload.get("mapping") or {}
    auto_site_code = bool(payload.get("autoGenerateSiteCode", True))
    has_headers = bool(payload.get("hasHeaders", True))
    city_overrides = set(payload.get("cityOverrides") or [])
    default_country = payload.get("defaultCountry")

    required_fields = ["name", "address"]
    missing_fields = [field for field in required_fields if not mapping.get(field)]
    if missing_fields:
        return jsonify({"ok": False, "error": f"Missing mappings for: {', '.join(missing_fields)}"}), 400

    existing_ids = fetch_existing_location_ids(config)
    if isinstance(rows_payload, list) and rows_payload:
        normalized_rows: List[Dict[str, Any]] = []
        for item in rows_payload:
            if isinstance(item, dict):
                normalized_rows.append({k: (v if v is not None else "") for k, v in item.items()})
        if normalized_rows:
            csv_text = rows_to_csv(normalized_rows)
            has_headers = True

    payloads, prep_errors, total = prepare_location_payloads(
        csv_text, mapping, auto_site_code, existing_ids, has_headers, city_overrides, default_country
    )
    results: List[Dict[str, Any]] = list(prep_errors)
    posted = 0

    for row_num, location_payload in payloads:
        loc_id = str(location_payload.get("id") or location_payload.get("siteCode") or "").strip()
        if loc_id:
            if not hasattr(upload_locations, "_posted_ids"):
                upload_locations._posted_ids = set()  # type: ignore[attr-defined]
            posted_ids_set = upload_locations._posted_ids  # type: ignore[attr-defined]
            if loc_id in posted_ids_set:
                results.append(
                    {
                        "row": row_num,
                        "ok": False,
                        "error": f"Duplicate location id '{loc_id}' in request; skipped",
                    }
                )
                continue
            posted_ids_set.add(loc_id)
        try:
            resp = api_request(
                "post", f"/api/networks/{config['network_id']}/locations", config, json_body=location_payload
            )
        except requests.RequestException as exc:
            results.append({"row": row_num, "ok": False, "error": str(exc)})
            continue

        try:
            resp_body = resp.json()
        except ValueError:
            resp_body = resp.text

        if 200 <= resp.status_code < 300:
            posted += 1
            results.append({"row": row_num, "ok": True, "response": resp_body})
        else:
            results.append(
                {
                    "row": row_num,
                    "ok": False,
                    "status": resp.status_code,
                    "response": resp_body,
                }
            )

    return jsonify({"ok": True, "total": total, "posted": posted, "results": results})


@app.post("/api/locations/delete")
def delete_locations():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400

    data = request.get_json(force=True)
    ids = data.get("ids") or []
    if not isinstance(ids, list) or not ids:
        return jsonify({"ok": False, "error": "Provide a list of ids to delete"}), 400

    results: List[Dict[str, Any]] = []
    deleted = 0
    for loc_id in ids:
        loc_id = str(loc_id).strip()
        if not loc_id:
            continue
        try:
            resp = api_request("delete", f"/api/networks/{config['network_id']}/locations/{loc_id}", config)
        except requests.RequestException as exc:
            results.append({"id": loc_id, "ok": False, "error": str(exc)})
            continue
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if 200 <= resp.status_code < 300:
            deleted += 1
            results.append({"id": loc_id, "ok": True, "response": body})
        else:
            results.append({"id": loc_id, "ok": False, "status": resp.status_code, "response": body})

    return jsonify({"ok": True, "deleted": deleted, "total": len(ids), "results": results})


@app.post("/api/locations/update")
def update_locations():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400

    data = request.get_json(force=True)
    updates = data.get("updates") or []
    if not isinstance(updates, list) or not updates:
        return jsonify({"ok": False, "error": "Provide updates as a list of {id, lat, lng}"}), 400

    results: List[Dict[str, Any]] = []
    updated = 0
    for item in updates:
        path_id = str(item.get("pathId") or item.get("id") or "").strip()
        lat_val = parse_float(item.get("lat"))
        lng_val = parse_float(item.get("lng"))
        if not path_id or lat_val is None or lng_val is None:
            results.append({"id": path_id or "(missing)", "ok": False, "error": "id, lat, lng are required"})
            continue
        payload: Dict[str, Any] = {"lat": lat_val, "lng": lng_val}
        # Optional fields if present
        for key in ("id", "name", "city", "adminDivision", "country"):
            if item.get(key) is not None:
                payload[key] = item.get(key)
        try:
            resp = api_request(
                "patch",
                f"/api/networks/{config['network_id']}/locations/{path_id}",
                config,
                json_body=payload,
            )
        except requests.RequestException as exc:
            results.append({"id": path_id, "ok": False, "error": str(exc)})
            continue
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if 200 <= resp.status_code < 300:
            updated += 1
            results.append({"id": path_id, "ok": True, "response": body, "status": resp.status_code})
        else:
            results.append({"id": path_id, "ok": False, "status": resp.status_code, "response": body})

    success = updated == len(updates)
    status = 200 if success else 400
    return jsonify({"ok": success, "updated": updated, "total": len(updates), "results": results}), status


@app.get("/devices")
def devices_page():
    return render_template("devices.html", config=current_config())


DEVICE_FIELDS: List[Dict[str, Any]] = [
    {"name": "name", "label": "Name", "type": "str", "required": True},
    {"name": "type", "label": "Type", "type": "str", "required": False},
    {"name": "host", "label": "Host", "type": "str", "required": True},
    {"name": "port", "label": "Port", "type": "int", "required": False},
    {"name": "cliCredentialId", "label": "CLI Credential Id", "type": "str", "required": False},
    {"name": "cliCredential2Id", "label": "CLI Credential 2 Id", "type": "str", "required": False},
    {"name": "cliCredential3Id", "label": "CLI Credential 3 Id", "type": "str", "required": False},
    {"name": "httpCredentialId", "label": "HTTP Credential Id", "type": "str", "required": False},
    {"name": "snmpCredentialId", "label": "SNMP Credential Id", "type": "str", "required": False},
    {"name": "jumpServerId", "label": "Jump Server Id", "type": "str", "required": False},
    {"name": "disableIpv6Collection", "label": "Disable IPv6 Collection", "type": "bool", "required": False},
    {"name": "fullCollectionLog", "label": "Full Collection Log", "type": "bool", "required": False},
    {"name": "paginationMode", "label": "Pagination Mode", "type": "str", "required": False},
    {"name": "terminalWidth", "label": "Terminal Width", "type": "int", "required": False},
    {"name": "largeRtt", "label": "Large RTT", "type": "bool", "required": False},
    {"name": "minBytesReadPerSec", "label": "Min Bytes Read Per Sec", "type": "bool", "required": False},
    {"name": "maxConnectionsPerMin", "label": "Max Connections Per Min", "type": "int", "required": False},
    {"name": "useFileTransferForLargeOutput", "label": "Use File Transfer For Large Output", "type": "bool", "required": False},
    {"name": "bmpStationHost", "label": "BMP Station Host", "type": "str", "required": False},
    {"name": "bgpAndBmpFailureHandling", "label": "BGP/BMP Failure Handling", "type": "str", "required": False},
    {"name": "collectBgpAdvertisements", "label": "Collect BGP Advertisements", "type": "bool", "required": False},
    {"name": "bgpTableType", "label": "BGP Table Type", "type": "str", "required": False},
    {"name": "bgpPeerType", "label": "BGP Peer Type", "type": "str", "required": False},
    {"name": "bgpSubnetsToCollect", "label": "BGP Subnets To Collect", "type": "str", "required": False},
    {"name": "prompt", "label": "Prompt", "type": "str", "required": False},
    {"name": "promptResponse", "label": "Prompt Response", "type": "str", "required": False},
    {"name": "collect", "label": "Collect", "type": "bool", "required": False},
    {"name": "note", "label": "Note", "type": "str", "required": False},
    {"name": "enableSnmpCollection", "label": "Enable SNMP Collection", "type": "bool", "required": False},
    {"name": "highCpuUsage", "label": "High CPU Usage", "type": "float", "required": False},
    {"name": "highMemoryUsage", "label": "High Memory Usage", "type": "float", "required": False},
    {"name": "highInputUtilization", "label": "High Input Utilization", "type": "float", "required": False},
    {"name": "highOutputUtilization", "label": "High Output Utilization", "type": "float", "required": False},
    {"name": "mediumCpuUsage", "label": "Medium CPU Usage", "type": "float", "required": False},
    {"name": "mediumMemoryUsage", "label": "Medium Memory Usage", "type": "float", "required": False},
    {"name": "mediumInputUtilization", "label": "Medium Input Utilization", "type": "float", "required": False},
    {"name": "mediumOutputUtilization", "label": "Medium Output Utilization", "type": "float", "required": False},
    {"name": "cloudLocationId", "label": "Cloud Location Id", "type": "str", "required": False},
    {"name": "cloudSetupId", "label": "Cloud Setup Id", "type": "str", "required": False},
    {"name": "cloudDiscoverySource", "label": "Cloud Discovery Source", "type": "str", "required": False},
]


@app.post("/api/devices/preview")
def preview_devices():
    data = request.get_json(force=True)
    csv_text = data.get("csvText") or ""
    has_headers = bool(data.get("hasHeaders", True))
    include_full = bool(data.get("full", False))
    headers, rows, header_labels = parse_csv_text(csv_text, has_headers)
    resp: Dict[str, Any] = {"headers": headers, "headerLabels": header_labels, "sample": rows[:5], "fields": DEVICE_FIELDS}
    if include_full:
        resp["rows"] = rows
    return jsonify(resp)


@app.post("/api/devices")
def upload_devices():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400

    data = request.get_json(force=True)
    csv_text = data.get("csvText") or ""
    mapping: Dict[str, Optional[str]] = data.get("mapping") or {}
    has_headers = bool(data.get("hasHeaders", True))

    # Validate required mappings
    missing = [field["name"] for field in DEVICE_FIELDS if field.get("required") and not mapping.get(field["name"])]
    if missing:
        return jsonify({"ok": False, "error": f"Missing mappings for: {', '.join(missing)}"}), 400

    headers, rows, _header_labels = parse_csv_text(csv_text, has_headers)
    results: List[Dict[str, Any]] = []
    posted = 0

    for index, row in enumerate(rows, start=1):
        try:
            device_payload: Dict[str, Any] = {}
            for field in DEVICE_FIELDS:
                column = mapping.get(field["name"])
                if not column:
                    continue
                raw_value = row.get(column, "")
                value: Any = raw_value
                if field["type"] == "int":
                    value = parse_int(raw_value)
                elif field["type"] == "float":
                    value = parse_float(raw_value)
                elif field["type"] == "bool":
                    parsed_bool = parse_bool(raw_value)
                    value = parsed_bool if parsed_bool is not None else False
                else:
                    value = (raw_value or "").strip()

                if value in (None, ""):
                    continue
                device_payload[field["name"]] = value

            if not device_payload.get("name") or not device_payload.get("host"):
                raise ValueError("name and host are required for every device")

            try:
                resp = api_request(
                    "post",
                    f"/api/networks/{config['network_id']}/classic-devices",
                    config,
                    json_body=device_payload,
                )
            except requests.RequestException as exc:
                results.append({"row": index, "ok": False, "error": str(exc)})
                continue

            try:
                resp_body = resp.json()
            except ValueError:
                resp_body = resp.text

            if 200 <= resp.status_code < 300:
                posted += 1
                results.append({"row": index, "ok": True, "response": resp_body})
            else:
                results.append(
                    {
                        "row": index,
                        "ok": False,
                        "status": resp.status_code,
                        "response": resp_body,
                    }
                )
        except Exception as exc:
            results.append({"row": index, "ok": False, "error": str(exc)})

    return jsonify({"ok": True, "total": len(rows), "posted": posted, "results": results})


@app.get("/device-location")
def device_location_page():
    return render_template("device_location.html", config=current_config())


@app.post("/api/device-location/preview")
def device_location_preview():
    data = request.get_json(force=True)
    csv_text = data.get("csvText") or ""
    has_headers = bool(data.get("hasHeaders", True))
    include_full = bool(data.get("full", False))
    headers, rows, header_labels = parse_csv_text(csv_text, has_headers)
    resp: Dict[str, Any] = {"headers": headers, "headerLabels": header_labels, "sample": rows[:5]}
    if include_full:
        resp["rows"] = rows
    return jsonify(resp)


def fetch_locations_map(config: Dict[str, str]) -> Dict[str, str]:
    locs = fetch_existing_location_ids(config)
    # fetch_existing_location_ids returns set of IDs; we need names too
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/locations", config)
        data = resp.json()
        if isinstance(data, list):
            return {str(item.get("name") or "").lower(): str(item.get("id")) for item in data if item.get("id")}
    except Exception:
        return {}
    return {}


def fetch_device_name_set(config: Dict[str, str]) -> Set[str]:
    if not config.get("network_id"):
        return set()
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/devices", config)
        data = resp.json()
        if 200 <= resp.status_code < 300 and isinstance(data, list):
            return {str(item.get("name")).strip().lower() for item in data if isinstance(item, dict) and item.get("name")}
    except Exception:
        return set()
    return set()


@app.post("/api/device-location/apply")
def device_location_apply():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400

    data = request.get_json(force=True)
    csv_text = data.get("csvText") or ""
    mapping = data.get("mapping") or {}
    use_name = bool(data.get("useName"))
    has_headers = bool(data.get("hasHeaders", True))
    skip_devices = {str(d).strip().lower() for d in (data.get("skipDevices") or []) if str(d).strip()}
    skip_locations = {str(d).strip().lower() for d in (data.get("skipLocations") or []) if str(d).strip()}
    network_devices = fetch_device_name_set(config)

    device_col = mapping.get("device")
    loc_col = mapping.get("location")
    if not device_col or not loc_col:
        return jsonify({"ok": False, "error": "device and location columns are required"}), 400

    rows_payload = data.get("rows")
    headers: List[str] = []
    rows: List[Dict[str, Any]] = []
    if isinstance(rows_payload, list) and rows_payload:
        for item in rows_payload:
            if isinstance(item, dict):
                rows.append({k: (v if v is not None else "") for k, v in item.items()})
        # derive headers from first row
        headers = list(rows[0].keys()) if rows else []
    else:
        headers, rows, _header_labels = parse_csv_text(csv_text, has_headers)
    if not headers or not rows:
        return jsonify({"ok": False, "error": "No CSV rows detected"}), 400

    name_to_id = {}
    if use_name:
        name_to_id = fetch_locations_map(config)

    payload = {}
    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        device_name = (row.get(device_col, "") or "").strip()
        location_value = (row.get(loc_col, "") or "").strip()
        if device_name.lower() in skip_devices:
            results.append({"row": idx, "ok": False, "device": device_name, "error": "Skipped (device not found in network)"})
            continue
        if network_devices and device_name.strip().lower() not in network_devices:
            results.append({"row": idx, "ok": False, "device": device_name, "error": "Skipped (device not found in network)"})
            continue
        if not device_name or not location_value:
            results.append(
                {
                    "row": idx,
                    "ok": False,
                    "error": "Missing device or location value",
                    "device": device_name,
                    "location": location_value,
                }
            )
            continue
        location_id = location_value
        if use_name:
            location_id = name_to_id.get(location_value.lower())
            if not location_id:
                if location_value.lower() in skip_locations:
                    results.append(
                        {
                            "row": idx,
                            "ok": False,
                            "error": f"Skipped (location '{location_value}' not found in network)",
                            "device": device_name,
                            "location": location_value,
                        }
                    )
                    continue
                else:
                    results.append(
                        {
                            "row": idx,
                            "ok": False,
                            "error": f"Location name '{location_value}' not found",
                            "device": device_name,
                            "location": location_value,
                        }
                    )
                continue
        payload[device_name] = location_id

    if not payload:
        return jsonify({"ok": False, "error": "No valid rows to send", "results": results}), 400

    try:
        resp = api_request("patch", f"/api/networks/{config['network_id']}/atlas", config, json_body=payload)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc), "results": results}), 502

    try:
        resp_body = resp.json()
    except ValueError:
        resp_body = resp.text

    success = 200 <= resp.status_code < 300
    results.append({"ok": success, "status": resp.status_code, "response": resp_body})
    return jsonify(
        {
            "ok": success,
            "payload": payload,
            "results": results,
            "status": resp.status_code,
            "attempted": len(payload),
        }
    )


# ---------------------------------------------------------------------------
# Tag maker
# ---------------------------------------------------------------------------


@app.get("/tags")
def tags_page():
    return render_template("tags.html", config=current_config())


@app.get("/api/device-names")
def list_device_names():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/devices", config)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    if 200 <= resp.status_code < 300 and isinstance(data, list):
        names = [item.get("name") for item in data if isinstance(item, dict) and item.get("name")]
        return jsonify({"ok": True, "names": names})
    return jsonify({"ok": False, "status": resp.status_code, "response": data}), 400


@app.post("/api/device-tags")
def create_device_tags():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    payload = request.get_json(force=True)
    tags = payload.get("tags") or []
    if not isinstance(tags, list) or not tags:
        return jsonify({"ok": False, "error": "Provide tags as a list of {name, color?}"}), 400
    results: List[Dict[str, Any]] = []
    created = 0
    for tag in tags:
        tag_name = (tag.get("name") or "").strip()
        tag_color = (tag.get("color") or "").strip() or None
        if not tag_name:
            results.append({"ok": False, "error": "Missing tag name"})
            continue
        body: Dict[str, Any] = {"name": tag_name}
        if tag_color:
            body["color"] = tag_color
        try:
            resp = api_request("post", f"/api/networks/{config['network_id']}/device-tags", config, json_body=body)
        except requests.RequestException as exc:
            results.append({"ok": False, "name": tag_name, "error": str(exc)})
            continue
        try:
            resp_body = resp.json()
        except ValueError:
            resp_body = resp.text
        if 200 <= resp.status_code < 300:
            created += 1
            results.append({"ok": True, "name": tag_name, "status": resp.status_code, "response": resp_body})
        else:
            results.append(
                {"ok": False, "name": tag_name, "status": resp.status_code, "response": resp_body}
            )
    return jsonify({"ok": created == len(results), "created": created, "total": len(results), "results": results})


@app.get("/api/device-tags")
def list_device_tags():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/device-tags", config)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    if 200 <= resp.status_code < 300:
        # Some deployments return list directly, others wrap inside objects
        tags = data
        if isinstance(data, dict):
            if "tags" in data and isinstance(data["tags"], list):
                tags = data["tags"]
            elif "items" in data:
                tags = data.get("items")
        return jsonify({"ok": True, "tags": tags})
    return jsonify({"ok": False, "status": resp.status_code, "response": data}), 400

@app.delete("/api/device-tags/<tag_name>")
def delete_device_tag(tag_name: str):
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    tag = (tag_name or "").strip()
    if not tag:
        return jsonify({"ok": False, "error": "tagName is required"}), 400
    try:
        resp = api_request("delete", f"/api/networks/{config['network_id']}/device-tags/{tag}", config)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    if 200 <= resp.status_code < 300:
        return jsonify({"ok": True, "status": resp.status_code, "response": body})
    return jsonify({"ok": False, "status": resp.status_code, "response": body}), 400


@app.patch("/api/device-tags/<tag_name>")
def patch_device_tag(tag_name: str):
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    current = (tag_name or "").strip()
    if not current:
        return jsonify({"ok": False, "error": "tagName is required"}), 400
    payload = request.get_json(force=True) or {}
    try:
        resp = api_request(
            "patch",
            f"/api/networks/{config['network_id']}/device-tags/{current}",
            config,
            json_body=payload,
        )
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    if 200 <= resp.status_code < 300:
        return jsonify({"ok": True, "status": resp.status_code, "response": body})
    return jsonify({"ok": False, "status": resp.status_code, "response": body}), 400

@app.get("/api/snapshots/latest")
def latest_snapshot():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    try:
        resp = api_request("get", f"/api/networks/{config['network_id']}/snapshots/latestProcessed", config)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    try:
        data = resp.json()
    except ValueError:
        data = resp.text
    if 200 <= resp.status_code < 300:
        snapshot_id = data.get("id") if isinstance(data, dict) else None
        return jsonify({"ok": True, "snapshotId": snapshot_id, "response": data})
    return jsonify({"ok": False, "status": resp.status_code, "response": data}), 400


@app.post("/api/device-tags/apply")
def apply_device_tags():
    config = current_config()
    if not config.get("network_id"):
        return jsonify({"ok": False, "error": "Network ID is required. Set it on the home page."}), 400
    payload = request.get_json(force=True)
    groups = payload.get("groups") or []
    validate = payload.get("validateDevices", True)
    snapshot_id = (payload.get("snapshotId") or "").strip()
    if not isinstance(groups, list) or not groups:
        return jsonify({"ok": False, "error": "groups must be a non-empty list"}), 400

    snapshot_part = f"&snapshotId={snapshot_id}" if snapshot_id else ""
    results: List[Dict[str, Any]] = []
    all_ok = True
    for group in groups:
        tag_name = (group.get("tagName") or "").strip()
        devices = group.get("devices") or []
        if not tag_name or not isinstance(devices, list) or not devices:
            all_ok = False
            results.append({"ok": False, "tag": tag_name or "(missing)", "error": "tagName and devices are required"})
            continue
        try:
            resp = api_request(
                "post",
                f"/api/networks/{config['network_id']}/device-tags/{tag_name}?action=addTo&validateDevices={'true' if validate else 'false'}{snapshot_part}",
                config,
                json_body={"devices": devices},
            )
        except requests.RequestException as exc:
            all_ok = False
            results.append({"ok": False, "tag": tag_name, "error": str(exc)})
            continue
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if 200 <= resp.status_code < 300:
            results.append({"ok": True, "tag": tag_name, "status": resp.status_code, "response": body})
        else:
            all_ok = False
            results.append({"ok": False, "tag": tag_name, "status": resp.status_code, "response": body})

    return jsonify({"ok": all_ok, "results": results})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", "5001")))
