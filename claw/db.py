"""
db.py — Google Cloud database access for CareRelay.

Single entry point: get_db() returns a connected client.
All reads/writes go through here so the rest of the codebase
stays decoupled from the specific GCP database product chosen.

TODO: decide which GCP product to use and fill in the sections marked below.
      Leading candidates:
        - Firestore   (schemaless, real-time, easy Python SDK)
        - Cloud SQL   (Postgres/MySQL if relational schema is preferred)
        - BigQuery    (if analytics / bulk WHOOP history is the primary use case)
"""

import os
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_db():
    """
    Returns a connected GCP database client.

    TODO: replace the body with the real client initialisation, e.g.:
        Firestore  → google.cloud.firestore.Client()
        Cloud SQL  → sqlalchemy.create_engine(os.getenv("DB_URL"))
        BigQuery   → google.cloud.bigquery.Client()

    The caller is responsible for closing / releasing the connection
    if the chosen client requires it.
    """
    # TODO: set GOOGLE_APPLICATION_CREDENTIALS or use Workload Identity
    # TODO: set GCP_PROJECT_ID in .env / Cloud Run env vars
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise EnvironmentError("GCP_PROJECT_ID is not set.")

    # TODO: swap the line below for the real client
    raise NotImplementedError("get_db(): GCP client not yet configured — see TODOs above.")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def read_data(user_id: str) -> dict:
    """
    Fetches the latest WHOOP snapshot for a user from the database.
    Drop-in replacement for synthetic_data.get_data() once the DB is live.

    TODO: implement once get_db() is wired up.
        Firestore example:
            db = get_db()
            docs = (
                db.collection("whoop_snapshots")
                .where("user_id", "==", user_id)
                .order_by("date", direction="DESCENDING")
                .limit(1)
                .stream()
            )
            return next(docs).to_dict()

        Cloud SQL example:
            with get_db().connect() as conn:
                row = conn.execute(
                    "SELECT * FROM whoop_snapshots WHERE user_id=%s ORDER BY date DESC LIMIT 1",
                    (user_id,)
                ).fetchone()
                return dict(row)
    """
    # TODO: remove this fallback once the real read is in place
    raise NotImplementedError(f"read_data(): DB not yet connected — user_id={user_id}")


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def log_anomaly_to_db(data: dict, anomaly_level: int) -> None:
    """
    Persists an anomaly event to the database.

    Args:
        data:          WHOOP snapshot dict (from synthetic_data.get_data()).
        anomaly_level: 1–4 severity received from the backend.

    TODO: implement once get_db() is wired up.
        Firestore example:
            db = get_db()
            db.collection("anomalies").add({
                "user_id":       data.get("user_id"),
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "anomaly_level": anomaly_level,
                "recovery_score": data.get("recovery_score"),
                "hrv":           data.get("hrv"),
                "rhr":           data.get("resting_heart_rate"),
                "day_strain":    data.get("day_strain"),
                "raw_snapshot":  data,
            })

        Cloud SQL example:
            with get_db().connect() as conn:
                conn.execute(
                    "INSERT INTO anomalies (user_id, ts, level, snapshot) VALUES (%s,%s,%s,%s)",
                    (data["user_id"], datetime.now(timezone.utc), anomaly_level, json.dumps(data))
                )
    """
    # TODO: remove this print once the real write is in place
    print(f"  [DB] would write anomaly_level={anomaly_level} for user={data.get('user_id')} — not yet connected")
