"""Render all three full-size measurement tiles for one diagnosed specimen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(directory: Path) -> Path:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8-sig"))
    assets = directory / "assets"
    points = sorted(metrics["points"], key=lambda p: p["measurement_point"])
    assert [p["measurement_point"] for p in points] == ["P1", "P2", "P3"]
    colors = {"tp": (66, 190, 91), "fn": (58, 132, 246), "fp": (239, 73, 73)}
    background, ink = (246, 248, 251), (25, 43, 63)
    tile, gap, margin, row_height = 256, 18, 18, 406
    width = margin * 2 + tile * 5 + gap * 4
    canvas = Image.new("RGB", (width, row_height * 3 + 70), background)
    draw = ImageDraw.Draw(canvas)
    font_dir = Path("C:/Windows/Fonts")
    title_font = ImageFont.truetype(str(font_dir / "malgunbd.ttf"), 26)
    body_font = ImageFont.truetype(str(font_dir / "malgun.ttf"), 21)
    label_font = ImageFont.truetype(str(font_dir / "malgun.ttf"), 18)
    legend_font = ImageFont.truetype(str(font_dir / "malgun.ttf"), 16)
    headings = ["원본 타일", "정답 마스크 · Exposure", "예측 마스크 · Exposure", "마스크끼리 겹침", "원본 위에 겹침"]
    native_panels = []

    for row, point in enumerate(points):
        name = point["measurement_point"]
        raw = np.array(Image.open(assets / f"{name}_raw.png").convert("L"))
        actual = np.array(Image.open(assets / f"{name}_gt.png").convert("L")) > 0
        predicted = np.array(Image.open(assets / f"{name}_pred.png").convert("L")) > 0
        assert raw.shape == actual.shape == predicted.shape == (tile, tile)
        assert int(actual.sum()) == point["true_exposure_pixels"]
        assert int(predicted.sum()) == point["pred_exposure_pixels"]
        actual_pct = 100 * (1 - actual.mean())
        predicted_pct = 100 * (1 - predicted.mean())
        error = actual_pct - predicted_pct
        assert abs(error - point["actual_minus_pred_pp"]) < 1e-10
        masks = {"tp": actual & predicted, "fn": actual & ~predicted, "fp": ~actual & predicted}
        overlap = np.zeros((tile, tile, 3), dtype=np.uint8)
        composite = np.repeat(raw[:, :, None], 3, axis=2)
        for key, mask in masks.items():
            assert int(mask.sum()) == point[key]
            color = np.array(colors[key])
            overlap[mask] = color
            composite[mask] = np.round(0.30 * composite[mask] + 0.70 * color).astype(np.uint8)
        panels = [np.repeat(raw[:, :, None], 3, axis=2),
                  np.repeat((actual.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
                  np.repeat((predicted.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
                  overlap, composite]
        top = row * row_height
        title = f"{metrics['case'].capitalize()} · 시편 {metrics['specimen']} · {name} · {metrics['position']} 방향"
        draw.text((margin, top + 10), title, fill=ink, font=title_font)
        subtitle = f"함침율   정답 {actual_pct:.2f}%  -  예측 {predicted_pct:.2f}%  =  {error:+.2f}%p"
        draw.text((margin, top + 50), subtitle, fill=ink, font=body_font)
        for col, (heading, panel) in enumerate(zip(headings, panels)):
            x, y = margin + col * (tile + gap), top + 124
            assert draw.textlength(heading, font=label_font) <= tile
            draw.text((x, top + 93), heading, fill=ink, font=label_font)
            # Paste the entire original tile at 1:1 pixels, without resizing or subcropping.
            canvas.paste(Image.fromarray(panel), (x, y))
            native_panels.append((x, y, panel))
        if row < 2:
            draw.line((margin, top + 398, width - margin, top + 398), fill=(216, 224, 234))

    legend_y = row_height * 3 + 2
    for x, key, label in [
        (margin, "tp", "초록: 정답·예측 Exposure 일치"),
        (475, "fn", "파랑: 정답만 있음 · 미검출"),
        (932, "fp", "빨강: 예측만 있음 · 과검출"),
    ]:
        draw.rectangle((x, legend_y + 5, x + 16, legend_y + 21), fill=colors[key])
        draw.text((x + 25, legend_y), label, fill=ink, font=legend_font)
    draw.text((margin, legend_y + 34), "각 패널: 256×256 전체 타일 · 저장 이미지에서 1:1 픽셀 · Exposure = 미함침 영역 · 차이 = 정답 - 예측 (%p)", fill=ink, font=legend_font)
    output = directory / "05_p123_full_tiles_overlap.png"
    canvas.save(output)
    saved = np.array(Image.open(output))
    for x, y, panel in native_panels:
        assert np.array_equal(saved[y:y + tile, x:x + tile], panel)
    print(json.dumps({"output": str(output.resolve()), "image_size": list(canvas.size),
                      "verified_native_panels": len(native_panels)}, ensure_ascii=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "analysis/case05_specimen03_diagnosis")
    render(parser.parse_args().analysis_dir)
