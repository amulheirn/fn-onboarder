from __future__ import annotations

import csv
import hashlib
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


def parse_csv_text(csv_text: str, has_headers: bool = True) -> Tuple[List[str], List[Dict[str, str]]]:
    cleaned = csv_text.strip()
    if not cleaned:
        return [], []

    preview = cleaned[:1024]
    try:
        dialect = csv.Sniffer().sniff(preview)
    except csv.Error:
        dialect = csv.excel

    if has_headers:
        reader = csv.DictReader(io.StringIO(cleaned), dialect=dialect)
        headers = reader.fieldnames or []
        rows = [row for row in reader if row and any((value or "").strip() for value in row.values())]
        return headers, rows

    reader = csv.reader(io.StringIO(cleaned), dialect=dialect)
    raw_rows = [row for row in reader]
    if not raw_rows:
        return [], []
    headers = [f"col{i+1}" for i in range(len(raw_rows[0]))]
    rows: List[Dict[str, str]] = []
    for raw in raw_rows:
        # Pad/truncate to header length
        padded = list(raw) + [""] * (len(headers) - len(raw))
        row_dict = {h: padded[i] if i < len(padded) else "" for i, h in enumerate(headers)}
        if any((v or "").strip() for v in row_dict.values()):
            rows.append(row_dict)
    return headers, rows


def slugify(text: str, fallback: str = "") -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    trimmed = "-".join(filter(None, safe.split("-")))
    if trimmed:
        return trimmed[:64]
    return fallback or "site"


def pseudo_geocode(address: str) -> Tuple[float, float]:
    """Deterministic pseudo-geocode to avoid external calls in offline mode."""
    digest = hashlib.sha1(address.encode("utf-8", errors="ignore")).digest()
    lat_raw = int.from_bytes(digest[:4], "big")
    lng_raw = int.from_bytes(digest[4:8], "big")
    lat = (lat_raw / (2**32 - 1)) * 180 - 90
    lng = (lng_raw / (2**32 - 1)) * 360 - 180
    return round(lat, 4), round(lng, 4)


def prepare_location_payloads(
    csv_text: str,
    mapping: Dict[str, Optional[str]],
    auto_site_code: bool = True,
    existing_ids: Optional[Set[str]] = None,
    has_headers: bool = True,
    city_overrides: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Dict[str, Any]], int]:
    """Convert CSV rows to location payloads and collect per-row errors."""
    headers, rows = parse_csv_text(csv_text, has_headers)
    payloads: List[Tuple[int, Dict[str, Any]]] = []
    errors: List[Dict[str, Any]] = []
    used_ids: Set[str] = set(existing_ids or set())
    prefix_counters: Dict[str, int] = {}
    override_set: Set[str] = {c.strip().lower() for c in (city_overrides or set()) if c and c.strip()}

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

    COUNTRY_BOUNDS = {
        "DE": ((47.27, 55.09), (5.87, 15.04)),
        "US": ((24.52, 49.38), (-124.77, -66.95)),
        "GB": ((49.8, 60.9), (-8.6, 1.8)),
        "UK": ((49.8, 60.9), (-8.6, 1.8)),
        "FR": ((41.0, 51.1), (-5.1, 9.6)),
        "ES": ((27.6, 43.8), (-18.2, 4.3)),
        "IT": ((36.6, 47.1), (6.6, 18.5)),
        "CA": ((41.7, 83.1), (-141.0, -52.6)),
        "AU": ((-43.7, -10.7), (113.3, 153.6)),
    }

    COUNTRY_NAME_TO_CODE = {
        "GERMANY": "DE",
        "DEUTSCHLAND": "DE",
        "UNITED STATES": "US",
        "USA": "US",
        "UK": "GB",
        "UNITED KINGDOM": "GB",
        "FRANCE": "FR",
        "SPAIN": "ES",
        "ITALY": "IT",
        "CANADA": "CA",
        "AUSTRALIA": "AU",
    }

    def normalize_address(addr: str) -> str:
        """Normalize supported formats into street, city, country ordering for geocoding."""
        addr = (addr or "").strip()
        if not addr:
            return ""
        # Already comma-separated; leave as-is
        if "," in addr:
            return addr
        # Tokenize: expected COUNTRY CITY STREET...
        tokens = addr.split()
        if len(tokens) >= 3 and len(tokens[0]) in (2, 3) and tokens[0].isalpha():
            country = tokens[0].strip()
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
            country = match.group("country").strip()
            city = match.group("city").strip()
            street = match.group("street").strip()
            return f"{street}, {city}, {country}"
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
                    return token
                name = last.upper()
                if name in COUNTRY_NAME_TO_CODE:
                    return COUNTRY_NAME_TO_CODE[name]
            # Space-separated first token might be country code
            parts = text.split()
            if parts and len(parts[0]) in (2, 3) and parts[0].isalpha():
                return parts[0].upper()
            # Name lookup
            name = parts[-1].upper() if parts else ""
            if name in COUNTRY_NAME_TO_CODE:
                return COUNTRY_NAME_TO_CODE[name]
        return None

    def clamp_to_country(lat: float, lng: float, country_code: Optional[str]) -> Tuple[float, float]:
        if not country_code:
            return lat, lng
        bounds = COUNTRY_BOUNDS.get(country_code.upper())
        if not bounds:
            return lat, lng
        (lat_min, lat_max), (lng_min, lng_max) = bounds
        # Normalize raw lat/lng to 0-1 then scale into country bounds
        lat_norm = (lat + 90) / 180.0
        lng_norm = (lng + 180) / 360.0
        lat_clamped = lat_min + (lat_max - lat_min) * max(0.0, min(1.0, lat_norm))
        lng_clamped = lng_min + (lng_max - lng_min) * max(0.0, min(1.0, lng_norm))
        return round(lat_clamped, 4), round(lng_clamped, 4)

    for index, row in enumerate(rows, start=1):
        try:
            row_name = (row.get(mapping["name"], "") or "").strip()
            row_address_raw = (row.get(mapping["address"], "") or "").strip()
            row_address = normalize_address(row_address_raw)
            country_code = extract_country_code(row_address_raw, row_address)
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
                    raise ValueError(f"Duplicate site-code/id detected: {final_id}")
                used_ids.add(final_id)
            else:
                final_id = next_site_code(row_name, row_address)

            lat_raw, lng_raw = pseudo_geocode(row_address)
            lat, lng = clamp_to_country(lat_raw, lng_raw, country_code)

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
    headers, rows = parse_csv_text(csv_text, has_headers)
    sample_rows = rows[:5]
    return jsonify({"headers": headers, "sample": sample_rows})


@app.post("/api/locations/geocode")
def geocode_locations():
    config = current_config()
    payload = request.get_json(force=True)
    csv_text = payload.get("csvText") or ""
    mapping: Dict[str, Optional[str]] = payload.get("mapping") or {}
    auto_site_code = bool(payload.get("autoGenerateSiteCode", True))
    has_headers = bool(payload.get("hasHeaders", True))
    city_overrides = set(payload.get("cityOverrides") or [])

    required_fields = ["name", "address"]
    missing_fields = [field for field in required_fields if not mapping.get(field)]
    if missing_fields:
        return jsonify({"ok": False, "error": f"Missing mappings for: {', '.join(missing_fields)}"}), 400

    existing_ids = fetch_existing_location_ids(config)
    payloads, errors, total = prepare_location_payloads(
        csv_text, mapping, auto_site_code, existing_ids, has_headers, city_overrides
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
    mapping: Dict[str, Optional[str]] = payload.get("mapping") or {}
    auto_site_code = bool(payload.get("autoGenerateSiteCode", True))
    has_headers = bool(payload.get("hasHeaders", True))
    city_overrides = set(payload.get("cityOverrides") or [])

    required_fields = ["name", "address"]
    missing_fields = [field for field in required_fields if not mapping.get(field)]
    if missing_fields:
        return jsonify({"ok": False, "error": f"Missing mappings for: {', '.join(missing_fields)}"}), 400

    existing_ids = fetch_existing_location_ids(config)
    payloads, prep_errors, total = prepare_location_payloads(
        csv_text, mapping, auto_site_code, existing_ids, has_headers, city_overrides
    )
    results: List[Dict[str, Any]] = list(prep_errors)
    posted = 0

    for row_num, location_payload in payloads:
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
    headers, rows = parse_csv_text(csv_text, has_headers)
    return jsonify({"headers": headers, "sample": rows[:5], "fields": DEVICE_FIELDS})


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

    headers, rows = parse_csv_text(csv_text, has_headers)
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
    headers, rows = parse_csv_text(csv_text, has_headers)
    return jsonify({"headers": headers, "sample": rows[:5]})


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

    device_col = mapping.get("device")
    loc_col = mapping.get("location")
    if not device_col or not loc_col:
        return jsonify({"ok": False, "error": "device and location columns are required"}), 400

    headers, rows = parse_csv_text(csv_text, has_headers)
    if not headers:
        return jsonify({"ok": False, "error": "No CSV headers detected"}), 400

    name_to_id = {}
    if use_name:
        name_to_id = fetch_locations_map(config)

    payload = {}
    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        device_name = (row.get(device_col, "") or "").strip()
        location_value = (row.get(loc_col, "") or "").strip()
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


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", "5001")))
