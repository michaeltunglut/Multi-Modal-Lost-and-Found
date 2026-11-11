# Campus Lost & Found

A local-first, working prototype for multimodal lost-and-found reporting and candidate retrieval. It stores reports and image references in SQLite, extracts visual features from uploaded photos, compares report descriptions, and ranks opposite-type reports for human review.

## Run it

Requires Python 3.10+, FastAPI, Uvicorn, python-multipart, and Pillow (installed from `requirements.txt`). No cloud account, model download, or external service is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The FastAPI backend serves the report form at <http://127.0.0.1:8000> and the standalone reports page at <http://127.0.0.1:8000/reports>. The SQLite database and uploaded images are created in `data/`. Set `PORT`, `HOST`, or `LOST_FOUND_DATA` to change the server address or storage directory. Stop the server with Ctrl+C.

## Use

1. Submit one LOST report with a photo, category, title, description, and any optional attributes.
2. Submit a corresponding FOUND report (or browse existing reports).
3. Open the separate **Browse reports** page and select **Find possible matches**. Each candidate shows the combined ranking and component scores.

Both report types require a photo and description. Supported photos are JPEG, PNG, and WebP, up to 10 MB. The matching API searches the opposite report type and returns up to 20 candidates scoring at least 0.60. This cutoff is a tunable retrieval heuristic, not a calibrated probability. Non-owners can submit sightings with a photo, details, location, and contact method; a sighting is stored only if its image/text similarity to the linked lost report is at least 0.60. Sighting details and photos are visible only to that lost-report owner and admins. Reports and images persist locally between restarts.

## What the matching scores mean

- **Visual:** normalized RGB and grayscale histogram intersection plus a 64-bit perceptual hash, combined as 62% / 20% / 18%.
- **Description:** cosine similarity over token and adjacent-token features, with a small transparent synonym map (for example, *rucksack/backpack*, *navy/blue*, and *logo/emblem*).
- **Category:** exact category agreement.
- **Combined:** 55% visual + 37% description + 8% category. Candidates below 0.60 are hidden from report owners; admins can still browse all found reports.

These are deterministic baseline features, not learned or pretrained AI embeddings. Similarity scores are retrieval hints, not probabilities or ownership decisions. Lighting, background, camera angle, vague descriptions, and look-alike items can affect ranking. The included baseline is intentionally self-contained; a later phase can replace the extractors with a pretrained vision-language model and compare image-only, text-only, and combined retrieval on consented labeled examples using Precision@K, Recall@K, and MRR.

## API

- `GET /api/health` — health and extractor information
- `GET /api/reports[?type=LOST|FOUND]` — browse reports
- `POST /api/reports` — submit a lost or found report
- `GET /api/reports/{id}/matches` — owner/admin-only ranked candidates
- `POST /api/reports/{id}/sightings` — submit a sighting, saved only at 60% similarity or higher
- `GET /api/reports/{id}/sightings` — owner/admin-only sightings
- `PATCH /api/sightings/{id}` — owner/admin follow-up actions
- `PATCH /api/matches/{id}` — set `REVIEWED`, `REJECTED`, or `RESOLVED`
- `DELETE /api/reports/{id}` — admin-only report deletion
- `GET /uploads/{filename}` — access-controlled report or sighting image

## Data and privacy

This prototype has basic account, owner, and admin access controls, but it is not production-hardened. Use test data until deployment security, retention, consent, backups, and abuse controls are in place. Found-report and sighting contact details are private to the relevant lost-report owner and admins. Do not upload identity documents or images containing sensitive personal information. The database stores image feature vectors so it can recompute rankings without model downloads. Local database and upload files are excluded from Git.


## Accounts and report privacy

- Anyone can browse lost reports and submit found reports.
- Posting a lost report requires a signed-in account so its matches can be restricted to that account.
- Only the lost-report owner and admins can view found candidates, found-report details, finder contact methods, and found photos for that report. Found reports are not listed publicly. Found-report submissions require a contact method (phone, email, or another way to reach the finder).
- Create accounts or sign in at `/account`. Sessions use HttpOnly, SameSite cookies and password hashes.
- To designate an administrator, set both `ADMIN_EMAIL` and a long random `ADMIN_SETUP_TOKEN` before starting the app. Register an account with that email and enter the setup token in the admin-only field. The email alone never grants admin access. Set `SESSION_COOKIE_SECURE=1` when serving over HTTPS.
