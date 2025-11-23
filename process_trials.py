#!/usr/bin/env python3
"""
process_trials.py

Batch-parse ClinicalTrials.gov XML files, extract modeling-friendly fields,
and emit them as NDJSON lines. By default it reads xml paths from data/all_xml
and writes the summary to data/trials_summary.ndjson.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET


def _clean_text(value: str | None) -> str:
    # Normalize whitespace and convert None to empty string
    return value.strip() if value else ""


def _text(root: ET.Element, path: str) -> str:
    # Safe helper for single-element lookups (returns empty string if missing)
    elem = root.find(path)
    return _clean_text(elem.text if elem is not None else None)


def _text_list(root: ET.Element, path: str) -> List[str]:
    # Collect non-empty text elements at the given path
    values: List[str] = []
    for elem in root.findall(path):
        text = _clean_text(elem.text if elem is not None else None)
        if text:
            values.append(text)
    return values


def _parse_outcome_measures(root: ET.Element, tag: str) -> List[str]:
    # Only keep the measure text; additional metadata can be added later if needed
    measures: List[str] = []
    for outcome in root.findall(tag):
        measure = _clean_text(outcome.findtext("measure"))
        if measure:
            measures.append(measure)
    return measures


def _parse_interventions(root: ET.Element) -> List[Dict[str, str]]:
    # Capture type/name/description for each intervention node
    interventions: List[Dict[str, str]] = []
    for intervention in root.findall("intervention"):
        item = {
            "type": _clean_text(intervention.findtext("intervention_type")),
            "name": _clean_text(intervention.findtext("intervention_name")),
            "description": _clean_text(intervention.findtext("description")),
        }
        if any(item.values()):
            interventions.append(item)
    return interventions


def _parse_locations(root: ET.Element) -> Dict[str, object]:
    # Summarize facility count and a deduplicated country list
    countries = set(_text_list(root, "location_countries/country"))
    facilities = []
    for loc in root.findall("location"):
        facility = loc.find("facility")
        if facility is None:
            continue
        name = _clean_text(facility.findtext("name"))
        address = facility.find("address")
        country = _clean_text(address.findtext("country")) if address is not None else ""
        facilities.append({"name": name, "country": country})
        if country:
            countries.add(country)
    return {
        "facility_count": len(facilities),
        "countries": sorted(countries),
    }


def parse_trial(xml_path: Path) -> Dict[str, object]:
    # Parse a single XML file into a flat dictionary suited for ML preprocessing
    tree = ET.parse(xml_path)
    root = tree.getroot()

    record: Dict[str, object] = {
        "xml_path": str(xml_path),
        "nct_id": _text(root, "id_info/nct_id"),
        "org_study_id": _text(root, "id_info/org_study_id"),
        "brief_title": _text(root, "brief_title"),
        "official_title": _text(root, "official_title"),
        "overall_status": _text(root, "overall_status"),
        "why_stopped": _text(root, "why_stopped"),
        "phase": _text(root, "phase"),
        "study_type": _text(root, "study_type"),
        "lead_sponsor": _text(root, "sponsors/lead_sponsor/agency"),
        "collaborators": _text_list(root, "sponsors/collaborator/agency"),
        "primary_completion_date": _text(root, "primary_completion_date"),
        "primary_completion_date_type": root.find("primary_completion_date").get("type", "") if root.find("primary_completion_date") is not None else "",
        "completion_date": _text(root, "completion_date"),
        "completion_date_type": root.find("completion_date").get("type", "") if root.find("completion_date") is not None else "",
        "start_date": _text(root, "start_date"),
        "start_date_type": root.find("start_date").get("type", "") if root.find("start_date") is not None else "",
        "study_first_posted": _text(root, "study_first_posted"),
        "last_update_posted": _text(root, "last_update_posted"),
        "enrollment": _text(root, "enrollment"),
        "enrollment_type": root.find("enrollment").get("type", "") if root.find("enrollment") is not None else "",
        "gender": _text(root, "eligibility/gender"),
        "minimum_age": _text(root, "eligibility/minimum_age"),
        "maximum_age": _text(root, "eligibility/maximum_age"),
        "healthy_volunteers": _text(root, "eligibility/healthy_volunteers"),
        "conditions": _text_list(root, "condition"),
        "condition_mesh_terms": _text_list(root, "condition_browse/mesh_term"),
        "keywords": _text_list(root, "keyword"),
        "interventions": _parse_interventions(root),
        "intervention_mesh_terms": _text_list(root, "intervention_browse/mesh_term"),
        "primary_outcomes": _parse_outcome_measures(root, "primary_outcome"),
        "secondary_outcomes": _parse_outcome_measures(root, "secondary_outcome"),
        "number_of_arms": _text(root, "number_of_arms"),
        "number_of_groups": _text(root, "number_of_groups"),
        "locations": _parse_locations(root),
    }

    return record


def iter_xml_paths(index_file: Path) -> Iterable[Path]:
    # Stream file paths from the index to avoid loading huge lists into memory
    with index_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield Path(line)


def process_trials(index_file: Path, output_file: Path, max_records: int | None = None) -> None:
    # Main loop: read XML paths, parse, and emit NDJSON records
    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    with output_file.open("w", encoding="utf-8") as writer:
        for xml_path in iter_xml_paths(index_file):
            if max_records is not None and processed >= max_records:
                break
            try:
                record = parse_trial(xml_path)
            except Exception as exc:  # noqa: BLE001 - log every parsing error for traceability
                errors += 1
                logging.exception("Failed to parse %s", xml_path)
                continue

            writer.write(json.dumps(record, ensure_ascii=False))
            writer.write("\n")
            processed += 1

            if processed and processed % 1000 == 0:
                # Provide periodic progress feedback for long runs
                logging.info("Parsed %d trial records", processed)

    logging.info("Done. Records: %d, Failures: %d, Output: %s", processed, errors, output_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse ClinicalTrials XML and emit NDJSON summaries.")
    parser.add_argument(
        "--index-file",
        type=Path,
        default=Path("data/all_xml"),
        help="File that lists XML paths, one per line. Defaults to data/all_xml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trials_summary.ndjson"),
        help="NDJSON output path. Defaults to data/trials_summary.ndjson.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on number of records to parse (useful for smoke tests).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    process_trials(args.index_file, args.output, args.max_records)


if __name__ == "__main__":
    main()

