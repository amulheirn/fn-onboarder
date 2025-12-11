<p align="center">
  <img src="./.github/onboarder-logo.png" width="200" alt="fn-onboarder logo" />
</p>

<!-- Optional: a one-line header below the logo -->
<h1 align="center">fn-onboarder</h1>

# FN Onboarder (local Flask UI)

Lightweight Flask app to help load data into Forward Networks via its API. Runs locally, keeps values in the browser session, and offers a dark/light theme plus a debug log.

## Setup
- Python 3.9+
- Install dependencies with: `pip install -r requirements.txt`
- Run: `python app.py` (binds to 127.0.0.1:5001 by default for security; override this with `PORT`).

## Key features
- **Setup**: Set base URL, toggle SSL verify, enter API key/secret, test API (`/api/version`), pick network by friendly name, status dots for API/collector, next-steps flowchart at the bottom.

- **Credentials**: Create CLI/HTTP credentials (mandatory fields highlighted), auto-associate switches, collector check with status dot and banner.   Note - collector must be connected for this to work.

- **Locations**: Create from CSV (wtih header toggle), map columns, geocode the locations. Optional city overrides for multi-word cities. Optionally generate locationIds for Forward if not supplied. Upload to the api. 

NOTE: if an address has commas in it, you must put the address in quotes so that each part of it is not treated as a separate field in the CSV.

View/Edit tab with search, inline edit of id/name/lat/lng, per-row save (PATCH) and bulk save, map preview modal, OpenStreetMap link.

Delete tab with select-all and live search.

- **Classic Devices**: Creates devices in Sources.  
- **Device → Location**: Map devices to location IDs/names (auto-resolve names via GET locations), preview, bulk push (PATCH).
- **Debug log**: Toggle in navbar; logs API responses across pages.


## Notes
- Network ID and base URL persist in session/localStorage; API key/secret are only in the browser session.
- City overrides are used only for parsing multi-word city names for the purpose of geocoding. They are not sent to the API.
- All API calls hit the configured base URL; SSL verification can be disabled for self-signed/dev only.

## Running tests
No automated tests included; run the app and exercise pages in the browser.
