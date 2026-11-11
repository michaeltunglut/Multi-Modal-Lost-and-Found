"""Best-effort email delivery backed by a small SQLite outbox."""
from datetime import datetime, timezone
from email.message import EmailMessage
import logging
import smtplib
import ssl
from fastapi import BackgroundTasks

from .config import (
    PUBLIC_BASE_URL,
    SMTP_FROM_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_STARTTLS,
    SMTP_USE_SSL,
    SMTP_USERNAME,
)
from .database import db, now

logger = logging.getLogger(__name__)


def _deliver(notification_id: int) -> None:
    with db() as c:
        row = c.execute(
            "SELECT * FROM email_notifications WHERE id=?", (notification_id,)
        ).fetchone()
    if not row or row["status"] == "SENT":
        return
    if not SMTP_HOST or not SMTP_FROM_EMAIL:
        logger.info("Email notification %s is queued; SMTP is not configured", row["event_key"])
        return

    message = EmailMessage()
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = row["recipient"]
    message["Subject"] = row["subject"]
    message.set_content(row["body"])
    try:
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15, context=ssl.create_default_context()) as smtp:
                if SMTP_USERNAME:
                    smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
                smtp.ehlo()
                if SMTP_STARTTLS:
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()
                if SMTP_USERNAME:
                    smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
                smtp.send_message(message)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        logger.warning("Email notification %s could not be sent: %s", row["event_key"], exc)
        with db() as c:
            c.execute(
                "UPDATE email_notifications SET status='FAILED',last_error=? WHERE id=?",
                (str(exc)[:500], notification_id),
            )
        return

    sent_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as c:
        c.execute(
            "UPDATE email_notifications SET status='SENT',last_error='',sent_at=? WHERE id=?",
            (sent_at, notification_id),
        )


def send_pending_notifications() -> None:
    """Drain queued messages on startup after SMTP is configured."""
    with db() as c:
        ids = [row["id"] for row in c.execute(
            "SELECT id FROM email_notifications WHERE status='PENDING' ORDER BY id"
        ).fetchall()]
    for notification_id in ids:
        _deliver(notification_id)


def _queue_and_send(
    event_key: str, event_type: str, recipient: str, subject: str, body: str,
    background_tasks: BackgroundTasks | None = None,
) -> None:
    """Persist each event once, then try delivery without failing report submission."""
    with db() as c:
        c.execute(
            """INSERT OR IGNORE INTO email_notifications
               (event_key,event_type,recipient,subject,body,status,created_at)
               VALUES(?,?,?,?,?,'PENDING',?)""",
            (event_key, event_type, recipient, subject, body, now()),
        )
        row = c.execute(
            "SELECT id,status FROM email_notifications WHERE event_key=?", (event_key,)
        ).fetchone()
    if not row or row["status"] == "SENT":
        return
    if background_tasks is not None:
        background_tasks.add_task(_deliver, row["id"])
    else:
        _deliver(row["id"])


def notify_match_owner(
    lost_report_id: int, found_report_id: int, score: float,
    background_tasks: BackgroundTasks | None = None,
) -> None:
    with db() as c:
        row = c.execute(
            """SELECT u.email,l.title FROM reports l
               JOIN users u ON u.id=l.reporter_id
               WHERE l.id=? AND l.report_type='LOST'""",
            (lost_report_id,),
        ).fetchone()
    if not row:
        return
    url = f"{PUBLIC_BASE_URL}/reports?report={lost_report_id}"
    subject = f"Possible match for your lost item: {row['title']}"
    body = (
        f"A found report may match your lost item, ‘{row['title']}’.\n\n"
        f"Similarity score: {score:.0%}. This is a search suggestion, not proof of ownership.\n"
        "Sign in to review the report and decide whether to contact the finder:\n"
        f"{url}\n"
    )
    _queue_and_send(
        f"match:{lost_report_id}:{found_report_id}", "MATCH", row["email"], subject, body,
        background_tasks,
    )


def notify_sighting_owner(
    lost_report_id: int, sighting_id: int, score: float,
    background_tasks: BackgroundTasks | None = None,
) -> None:
    with db() as c:
        row = c.execute(
            """SELECT u.email,l.title FROM reports l
               JOIN users u ON u.id=l.reporter_id
               WHERE l.id=? AND l.report_type='LOST'""",
            (lost_report_id,),
        ).fetchone()
    if not row:
        return
    url = f"{PUBLIC_BASE_URL}/reports?report={lost_report_id}&view=sightings"
    subject = f"A sighting was reported for your lost item: {row['title']}"
    body = (
        f"Someone submitted a sighting for your lost item, ‘{row['title']}’.\n\n"
        f"Similarity score: {score:.0%}. The sighting is awaiting your review.\n"
        "Sign in to view its details and contact the finder if appropriate:\n"
        f"{url}\n"
    )
    _queue_and_send(
        f"sighting:{sighting_id}", "SIGHTING", row["email"], subject, body,
        background_tasks,
    )
