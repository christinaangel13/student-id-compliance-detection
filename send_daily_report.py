"""
Sends an automatic end-of-day email report to a teacher, containing a single
PDF with:
  - A summary table of today's violations (timestamp, student, status)
  - Each violation's evidence screenshot embedded directly in the PDF

Set up to run once a day via Windows Task Scheduler (see README section),
or automatically when main.py is quit.

Before using, fill in the CONFIG section below with your email details.

Requires: pip install reportlab
"""

import os
import csv
import smtplib
from datetime import date
from email.message import EmailMessage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

# ---------------------------------------------------------------------------
# CONFIG — fill these in
# ---------------------------------------------------------------------------
SENDER_EMAIL = "your_project_gmail@gmail.com"      # the Gmail account sending the report
SENDER_APP_PASSWORD = "xxxx xxxx xxxx xxxx"        # Gmail App Password (NOT your normal password)
TEACHER_EMAIL = "teacher_email@example.com"        # who receives the report

LOG_FILE = "violation_log.csv"
REPORTS_DIR = "daily_reports"   # where today's PDF is staged before emailing


def get_today_violations():
    """Read violation_log.csv and return only rows from today."""
    today_str = date.today().strftime("%Y-%m-%d")
    todays_rows = []

    if not os.path.exists(LOG_FILE):
        return None, todays_rows

    with open(LOG_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0].startswith(today_str):
                todays_rows.append(row)

    return header, todays_rows


def build_pdf_report(header, rows, today_str):
    """Build a single PDF containing a per-student summary table and every
    evidence screenshot, grouped by student and ordered by time within each
    group. Returns the PDF file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORTS_DIR, f"compliance_report_{today_str}.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal_style = styles["Normal"]
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9.5, leading=13)
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9, textColor=colors.grey
    )

    # --- Group all rows by student name, preserving first-seen order,
    #     and sort each student's own violations by time ---
    grouped = {}
    order = []
    for r in rows:
        timestamp, name = r[0], r[1]
        screenshot_path = r[3] if len(r) > 3 else None
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append((timestamp, screenshot_path))

    for name in order:
        grouped[name].sort(key=lambda t: t[0])  # sort each student's entries by time

    story = []

    # --- Header ---
    story.append(Paragraph("Student ID Compliance Report", title_style))
    story.append(Paragraph(f"Date: {today_str}", normal_style))
    story.append(Paragraph(f"Total violations recorded: {len(rows)}", normal_style))
    story.append(Paragraph(f"Students involved: {len(order)}", normal_style))
    story.append(Spacer(1, 16))

    # --- Summary table: one row per student, all their violation times listed together ---
    table_header = ["Student Name", "Times Violated", "Violation Timestamps", "Status"]
    table_data = [table_header]
    for name in order:
        entries = grouped[name]
        times_list = "<br/>".join(t for t, _ in entries)
        table_data.append([
            Paragraph(name, cell_style),
            Paragraph(str(len(entries)), cell_style),
            Paragraph(times_list, cell_style),
            Paragraph("NO_ID", cell_style),
        ])

    tbl = Table(table_data, colWidths=[1.6 * inch, 1.0 * inch, 2.5 * inch, 0.9 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3D1327")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6EFE9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # --- Evidence pages: grouped by student, each student's shots in time order ---
    story.append(Paragraph("Evidence Screenshots", styles["Heading1"]))
    story.append(Spacer(1, 10))

    for name in order:
        entries = grouped[name]
        story.append(Paragraph(f"<b>{name}</b> &nbsp;&mdash;&nbsp; {len(entries)} violation(s)",
                                styles["Heading3"]))
        story.append(Spacer(1, 6))

        for timestamp, screenshot_path in entries:
            story.append(Paragraph(timestamp, caption_style))
            story.append(Spacer(1, 3))

            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    img = Image(screenshot_path, width=3.0 * inch, height=3.0 * inch, kind="proportional")
                    story.append(img)
                except Exception:
                    story.append(Paragraph("(Could not load screenshot)", caption_style))
            else:
                story.append(Paragraph("(Screenshot file not found)", caption_style))

            story.append(Spacer(1, 14))

        story.append(Spacer(1, 10))

    doc.build(story)
    return pdf_path


def send_email(pdf_path, violation_count, today_str):
    msg = EmailMessage()
    msg["Subject"] = f"Student ID Compliance Report — {today_str}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = TEACHER_EMAIL
    msg.set_content(
        f"Attached is the student ID compliance report for {today_str}.\n\n"
        f"Total violations recorded today: {violation_count}\n\n"
        f"The attached PDF contains a summary table and evidence screenshots "
        f"for every violation recorded.\n\n"
        f"This is an automated message from the Student Compliance Detection System."
    )

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf",
                            filename=os.path.basename(pdf_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        smtp.send_message(msg)

    print(f"[INFO] Report emailed to {TEACHER_EMAIL} successfully.")


def build_today_report():
    """Kept for compatibility with main.py's import — builds today's PDF
    and returns (pdf_path, violation_count), or (None, None) if nothing to send."""
    today_str = date.today().strftime("%Y-%m-%d")
    header, rows = get_today_violations()

    if not rows:
        print("[INFO] No violations recorded today — no email sent.")
        return None, None

    pdf_path = build_pdf_report(header, rows, today_str)
    return pdf_path, len(rows)


def main():
    today_str = date.today().strftime("%Y-%m-%d")
    pdf_path, violation_count = build_today_report()
    if pdf_path:
        send_email(pdf_path, violation_count, today_str)


if __name__ == "__main__":
    main()
