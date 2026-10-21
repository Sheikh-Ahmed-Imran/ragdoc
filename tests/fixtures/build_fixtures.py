"""Construct the T-009 validation PDFs.

Fixture construction lives here, deliberately outside the ``ragdoc`` package and
never imports it: the PDFs must not be built by (or agree by construction with)
the attribution code (ADR-003). Ground truth is asserted in ``manifest.json``,
never generated.

Run ``python tests/fixtures/build_fixtures.py`` to (re)write the PDFs next to
this file. Only PyMuPDF is needed. Each line is placed explicitly (no automatic
wrapping), so the extracted text is exactly the lines below joined by newlines;
where the answer sits relative to the 500-char / 50-overlap chunk grid is
therefore a property of this file, not of luck.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
WRAP = 78
LINES_PER_PAGE = 62

#: Filler placed before the unembedded-heading section so the heading ends just
#: before a chunk boundary (multiple of the 450-char stride). See that fixture.
HEADING_PAD = (
    "Notes. The cafeteria on the ground floor opens at eight, the car "
    "park gate is locked at seven in the evening, and visitors must sign in "
    "at the front desk before they are given a pass and escorted to a "
    "room by the person they have come to see. Lost passes are reported to "
    "the front desk, which issues a temporary one for the day."
)


def _para(text: str) -> list[str]:
    return textwrap.wrap(text, WRAP)


def _write_pdf(path: Path, lines: list[str]) -> None:
    doc = pymupdf.open()
    for i in range(0, len(lines), LINES_PER_PAGE):
        page = doc.new_page()
        y = 50
        for line in lines[i : i + LINES_PER_PAGE]:
            if line:
                page.insert_text((50, y), line, fontsize=9, fontname="helv")
            y += 12
    doc.save(path)
    doc.close()


# --------------------------------------------------------------------------- #
# 1. split-list: a five-item list that a chunk boundary cuts.
# --------------------------------------------------------------------------- #

SPLIT_ITEMS = [
    "1. Collect the signed supplier agreement and file a copy with procurement;",
    "2. Verify the supplier tax registration against the national register;",
    "3. Obtain two written trade references dated within the last year;",
    "4. Complete the security questionnaire and have the risk team approve it;",
    "5. Create the supplier record in the ledger and issue the vendor number.",
]
SPLIT_QUESTION = "What are the five steps for onboarding a new supplier?"
SPLIT_ANSWER = " ".join(SPLIT_ITEMS)


def split_list_lines() -> list[str]:
    lines: list[str] = []
    lines += _para(
        "Supplier relationships at the company are managed by the procurement "
        "office on behalf of every department that buys goods or services."
    )
    lines += _para(
        "New suppliers must be admitted before any purchase order is raised, "
        "and the office keeps the admission file for the life of the contract."
    )
    lines += _para("The five steps for onboarding a new supplier are listed here:")
    for item in SPLIT_ITEMS:
        lines += textwrap.wrap(item, 70)
    lines += _para(
        "Departments that skip admission are asked to regularise the supplier "
        "within thirty days, and unpaid invoices are held until they do."
    )
    lines += _para(
        "Questions about the admission process can be sent to the procurement "
        "desk, which answers during normal office hours on working days."
    )
    return lines


# --------------------------------------------------------------------------- #
# 2. unembedded-heading: the answer is intact in one chunk, but the heading that
#    matches the question is not part of that chunk's embedded text.
# --------------------------------------------------------------------------- #

HEADING_QUESTION = "What is the data retention policy for archived invoices?"
HEADING_ANSWER = "kept for seven years and then securely destroyed"
HEADING_TEXT = "4.2 Data Retention Policy for Archived Invoices"

_DISTRACTORS = [
    "Invoices received from suppliers are archived by the accounts payable team "
    "once payment is made, and the archive of invoices is searchable by number.",
    "A retention schedule for invoices and data records is maintained by the "
    "finance department, and the schedule lists each class of archived data.",
    "When an archived invoice is requested by an auditor, the data retention "
    "officer retrieves it from the archive and logs the request.",
    "Invoice data is retained in the ledger system and the archive of paid "
    "invoices is reviewed each year under the data retention rules.",
    "Customer invoices are archived in the same way as supplier invoices, and "
    "the data retention rules for archived invoices apply equally to both.",
    "The retention of invoice data after a supplier relationship ends is "
    "governed by the archive policy and reviewed by the compliance team.",
    "Archived invoices are stored in the document archive with their data "
    "retention class recorded so that retention reviews can find them.",
    "Credit notes are archived with the invoices they cancel, and the data "
    "retention period for an archived invoice also covers its credit notes.",
    "The archive of invoices is backed up nightly, and the backup of archived "
    "invoice data follows the same retention policy as the live archive.",
    "Retention of invoice images scanned from paper follows the data retention "
    "policy for archived invoices, and scans are indexed in the archive.",
    "Access to archived invoices is restricted by role, and each access to "
    "archived invoice data is recorded as part of the retention audit trail.",
    "Disposal of data from the invoice archive is approved by the retention "
    "officer, and a disposal log of archived invoices is kept for review.",
    "Invoices that are under dispute are placed on hold in the archive and "
    "are exempt from routine data retention disposal until resolved.",
    "Training on invoice archive handling and the data retention policy is "
    "given to every new member of the accounts payable team on joining.",
    "Reports on archived invoice volumes and data retention compliance are "
    "produced quarterly by finance and shared with the audit committee.",
    "Exceptions to the invoice archive retention rules must be requested in "
    "writing and approved by the head of finance before they take effect.",
]

_MORE_DISTRACTORS = [
    "The data retention policy for archived invoices is reviewed by the "
    "compliance team every year. Each archived invoice carries a retention "
    "date, and invoices past that date are listed for the retention officer.",
    "Archived invoices from overseas suppliers follow the same data retention "
    "policy as domestic invoices. The invoice archive records the country of "
    "each supplier so that local retention rules can be checked on request.",
    "Retention periods for invoice data are set out in the retention "
    "schedule. The schedule says how long archived invoices and the data "
    "attached to them must be kept before the archive may release them.",
    "A data retention policy review of the invoice archive found that "
    "archived invoices were indexed consistently. The review of archived "
    "invoice retention was presented to the audit committee in full.",
    "The archive stores each invoice with its approval data, payment data and "
    "retention class. Archived invoice data can be exported for an auditor, "
    "and the export is subject to the data retention policy.",
    "Duplicate invoices found in the archive are merged under the data "
    "retention policy. The archived invoice that is kept is the earliest one, "
    "and the retention officer records the merge in the invoice archive log.",
    "Staff who need an archived invoice submit a request to the archive. "
    "The request names the invoice number, and the data retention officer "
    "checks that the retention policy allows the invoice to be released.",
    "Legal holds override the data retention policy for archived invoices. "
    "While a hold is in force the invoice archive keeps every affected "
    "invoice, and the retention officer lifts the hold when legal agrees.",
    "Invoice archive storage costs are reported with the data retention "
    "figures. Longer retention of archived invoices raises the cost of the "
    "archive, and finance weighs that cost against the retention policy.",
    "The data retention policy applies to electronic invoices received by "
    "email as well as to paper invoices. Archived invoice emails are kept in "
    "the archive with the same retention class as the invoice itself.",
    "Archive migration projects must preserve invoice data and retention "
    "dates. After a migration the retention officer samples archived invoices "
    "to confirm that the data retention policy is still being applied.",
    "Quarterly checks confirm that archived invoices older than the "
    "retention period have been disposed of. The invoice archive report lists "
    "any data retention exceptions for the audit committee to review.",
]


def _section(n: int, title: str, body: str) -> list[str]:
    return [f"3.{n} {title}"] + _para(body)


def unembedded_heading_lines(pad: str = HEADING_PAD) -> list[str]:
    lines: list[str] = []
    lines += _para(
        "Finance operations handbook. This handbook describes how the finance "
        "department handles the documents and records it receives each day."
    )
    for n, text in enumerate(_DISTRACTORS[:8], start=1):
        lines += _section(n, f"Archive Topic {n}", text)
    lines += _para(pad)
    lines += [HEADING_TEXT]
    # The body deliberately avoids every word of the heading and the question
    # ("data", "retention", "policy", "archived", "invoices"): the only link
    # between this chunk and the question is the heading.
    lines += _para(
        "The finance team handles the paperwork for this section with great "
        "care every month. Material of this kind is " + HEADING_ANSWER + " by "
        "the finance team, following sign-off from the controller and a final "
        "check of the file."
    )
    lines += _para(
        "The cafeteria on the ground floor opens at eight and the car park "
        "gate is locked at seven in the evening on every working day."
    )
    for n, text in enumerate(_DISTRACTORS[8:] + _MORE_DISTRACTORS, start=9):
        lines += _section(n, f"Archive Topic {n}", text)
    return lines


# --------------------------------------------------------------------------- #
# 3. control: the answer survives chunking intact and ranks inside top_k.
# --------------------------------------------------------------------------- #

CONTROL_QUESTION = "How many days of annual leave do full-time employees receive?"
CONTROL_ANSWER = "Full-time employees receive twenty-five days of annual leave"


def control_lines() -> list[str]:
    lines: list[str] = []
    lines += _para(
        "Employee handbook. This document summarises the main terms of "
        "employment that apply to staff working at the company."
    )
    lines += _para(
        "Annual leave. " + CONTROL_ANSWER + " each calendar year, in addition "
        "to public holidays. Leave must be booked with a line manager."
    )
    lines += _para(
        "The office is open from eight until six on working days, and staff "
        "may start and finish earlier or later by agreement with a manager."
    )
    lines += _para(
        "Expenses are reimbursed monthly on submission of receipts, and "
        "travel must be booked through the approved travel agency."
    )
    lines += _para(
        "Equipment such as laptops is issued on the first day of work and "
        "must be returned to the technology desk when employment ends."
    )
    return lines


# --------------------------------------------------------------------------- #
# 4. two-column: loaders disagree about reading order (v1 comparison).
# --------------------------------------------------------------------------- #

TWO_COLUMN_QUESTION = "Within how many days must refund requests be submitted?"
TWO_COLUMN_LEFT = [
    "Refund requests must be submitted within",
    "thirty days of the delivery date and",
    "must include the original order number",
    "so that finance can match the payment.",
]
TWO_COLUMN_RIGHT = [
    "Office opening hours are eight until",
    "six on working days. Visitors sign in",
    "at the front desk and wait to be",
    "escorted to the room by their host.",
]
TWO_COLUMN_ANSWER = " ".join(TWO_COLUMN_LEFT)


def write_two_column(path: Path) -> None:
    """Left column drawn first, then the right column, on the same baselines.

    Content-stream order keeps each column contiguous (PyMuPDF, pypdf); a loader
    that groups text by vertical position joins the columns row by row
    (pdfplumber), which splits the answer sentence.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    for x, column in ((50, TWO_COLUMN_LEFT), (320, TWO_COLUMN_RIGHT)):
        for i, line in enumerate(column):
            page.insert_text((x, 100 + 14 * i), line, fontsize=9, fontname="helv")
    doc.save(path)
    doc.close()


# --------------------------------------------------------------------------- #
# 5. batch document: one PDF carrying several known-cause questions (v2 batch).
#    Batch mode runs one document, so the unembedded-heading, split-list and
#    control sections are joined into a single handbook. The unembedded-heading
#    section comes first so its text keeps the offsets (and the chunk grid
#    alignment) it has in its own fixture. The split list then starts at
#    extracted char 6511 = 14 * 450 + 211: the 500-char chunk starting at 6300
#    ends 211 chars into the list's 360, and the next chunk starts at 6750, after
#    the list's start, so no chunk contains the whole list (boundary cut).
# --------------------------------------------------------------------------- #


def batch_doc_lines() -> list[str]:
    lines = unembedded_heading_lines()
    lines += [""]
    lines += split_list_lines()
    lines += [""]
    lines += control_lines()
    return lines


def main(pad: str = HEADING_PAD) -> None:
    _write_pdf(HERE / "split_list.pdf", split_list_lines())
    _write_pdf(HERE / "unembedded_heading.pdf", unembedded_heading_lines(pad))
    _write_pdf(HERE / "control.pdf", control_lines())
    write_two_column(HERE / "two_column.pdf")
    _write_pdf(HERE / "batch_doc.pdf", batch_doc_lines())


if __name__ == "__main__":
    main()
