from __future__ import annotations

import csv
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.models.schemas import ExportData, RegionSelection
from app.services.match_service import MatchService


class ExportService:
    def __init__(self, match_service: MatchService) -> None:
        self.match_service = match_service

    def export_match_result(
        self,
        image_id: str,
        x: int,
        y: int,
        top_n: int,
        output_path: str,
        fmt: str,
        include_query_spectrum: bool,
        include_matched_curves: bool,
        ignore_water_bands: bool,
        min_valid_bands: int | None = None,
        selection: RegionSelection | None = None,
    ) -> ExportData:
        result = self.match_service.match_pixel(
            image_id=image_id,
            x=x,
            y=y,
            top_n=top_n,
            ignore_water_bands=ignore_water_bands,
            min_valid_bands=min_valid_bands,
            return_candidate_curves=include_matched_curves,
            selection=selection,
        )

        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower()
        if fmt not in {"csv", "txt"}:
            raise AppError(ErrorCode.EXPORT_FAILED, f"unsupported export format: {fmt}")

        if fmt == "csv":
            rows_written = self._write_csv(out, result, include_query_spectrum, include_matched_curves)
        else:
            rows_written = self._write_txt(out, result, include_query_spectrum, include_matched_curves)

        return ExportData(
            output_path=str(out),
            format=fmt,
            file_size=out.stat().st_size,
            rows_written=rows_written,
        )

    @staticmethod
    def _write_csv(path: Path, result, include_query_spectrum: bool, include_matched_curves: bool) -> int:
        rows = 0
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "type",
                    "rank",
                    "spectrum_id",
                    "name",
                    "class",
                    "sam_score",
                    "pearson_r",
                    "curve",
                ]
            )
            rows += 1
            if include_query_spectrum:
                writer.writerow(
                    [
                        "query",
                        0,
                        "",
                        f"pixel({result.query.x},{result.query.y})",
                        "",
                        "",
                        "",
                        ",".join(f"{v:.6f}" for v in result.query.spectrum),
                    ]
                )
                rows += 1
            for item in result.results:
                curve_str = ""
                if include_matched_curves and item.curve:
                    curve_str = ",".join(f"{v:.6f}" for v in item.curve)
                writer.writerow(
                    [
                        "match",
                        item.rank,
                        item.spectrum_id,
                        item.name,
                        item.class_name or "",
                        f"{item.sam_score:.8f}",
                        f"{item.pearson_r:.8f}" if item.pearson_r is not None else "",
                        curve_str,
                    ]
                )
                rows += 1
        return rows

    @staticmethod
    def _write_txt(path: Path, result, include_query_spectrum: bool, include_matched_curves: bool) -> int:
        lines = []
        lines.append(f"query_pixel=({result.query.x},{result.query.y})")
        lines.append(f"signature_hash={result.match_context.signature_hash}")
        lines.append(f"metric={result.match_context.metric}")
        lines.append(f"bands_used={result.query.bands_used}/{result.query.bands_total}")
        if include_query_spectrum:
            lines.append("query_spectrum=" + ",".join(f"{v:.6f}" for v in result.query.spectrum))
        lines.append("")
        lines.append("matches:")

        for item in result.results:
            line = (
                f"rank={item.rank}\t"
                f"id={item.spectrum_id}\t"
                f"name={item.name}\t"
                f"class={item.class_name or ''}\t"
                f"sam={item.sam_score:.8f}\t"
                f"pearson={'' if item.pearson_r is None else f'{item.pearson_r:.8f}'}"
            )
            lines.append(line)
            if include_matched_curves and item.curve:
                lines.append("curve=" + ",".join(f"{v:.6f}" for v in item.curve))
        path.write_text("\n".join(lines), encoding="utf-8")
        return len(lines)
