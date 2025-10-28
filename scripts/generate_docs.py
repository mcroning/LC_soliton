#!/usr/bin/env python3
# generate_docs.py — build docs/*.pdf with Unicode-safe fonts and robust layout (no inline <font> tags)

import os, glob, argparse
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping


def register_unicode_fonts(out_dir):
    """
    Try to register a primary body font and a monospaced font from docs/fonts/*.ttf.
    If none found, fall back to Helvetica (body) and Courier (mono).
    Returns (body_font_name, mono_font_name).
    """
    fonts_glob = os.path.join(out_dir, "fonts", "*.ttf")
    ttf_paths = sorted(glob.glob(fonts_glob))

    body = "Helvetica"
    mono = "Courier"

    # Prefer a DejaVu or Noto face if present; otherwise first TTF for both roles.
    preferred_order = ["DejaVuSansMono.ttf", "DejaVuSans.ttf", "NotoSansMono-Regular.ttf", "NotoSans-Regular.ttf"]
    preferred_paths = []
    for pref in preferred_order:
        p = os.path.join(out_dir, "fonts", pref)
        if os.path.exists(p):
            preferred_paths.append(p)

    # Build candidate list: preferred first, then all others (deduped)
    seen = set()
    candidates = []
    for p in preferred_paths + ttf_paths:
        if p not in seen:
            seen.add(p)
            candidates.append(p)

    def _register_face(ttf_path):
        face = os.path.splitext(os.path.basename(ttf_path))[0]
        try:
            pdfmetrics.registerFont(TTFont(face, ttf_path))
            # Provide family mappings so ReportLab can resolve bold/italic requests
            addMapping(face, 0, 0, face)  # normal
            addMapping(face, 1, 0, face)  # bold -> same face if no Bold TTF
            addMapping(face, 0, 1, face)  # italic -> same face if no Italic TTF
            addMapping(face, 1, 1, face)  # bold-italic
            return face
        except Exception as e:
            print(f"[docs] Warning: failed to register {ttf_path}: {e}")
            return None

    # Try to get a mono first, then body
    mono_candidates = [p for p in candidates if "Mono" in os.path.basename(p)]
    body_candidates = [p for p in candidates if p not in mono_candidates]

    # Pick mono
    for p in mono_candidates:
        face = _register_face(p)
        if face:
            mono = face
            print(f"[docs] Registered mono font: {face} ({p})")
            break

    # Pick body
    for p in body_candidates:
        face = _register_face(p)
        if face:
            body = face
            print(f"[docs] Registered body font: {face} ({p})")
            break

    if (mono == "Courier") and (body == "Helvetica"):
        print("[docs] No TTF in docs/fonts; using Helvetica/Courier.")

    return body, mono


def build_pdf(path, title, sections, body_font="Helvetica", mono_font="Courier"):
    """
    sections: list of (heading, [block, block, ...])
      where each block is a dict { "type": "text"|"code", "content": "..." }
    """
    doc = SimpleDocTemplate(
        path,
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=title,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontName=body_font, fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="Head2", parent=styles["Heading2"], fontName=body_font, fontSize=13, leading=16, spaceBefore=6))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontName=body_font, fontSize=8.5, leading=12, textColor="#555555"))

    # Preformatted code style (no HTML parsing)
    code_style = ParagraphStyle(
        name="Code",
        parent=styles["BodyText"],
        fontName=mono_font,
        fontSize=9.5,
        leading=12,
    )

    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Tiny"]))
    story.append(Spacer(1, 0.20 * inch))

    for heading, blocks in sections:
        story.append(Paragraph(f"<b>{heading}</b>", styles["Head2"]))
        story.append(Spacer(1, 0.06 * inch))
        for blk in blocks:
            t = blk.get("type", "text")
            c = blk.get("content", "").rstrip("\n")
            if not c:
                continue
            if t == "code":
                story.append(Preformatted(c, code_style))
            else:
                # Regular paragraph text (no inline <font> tags)
                story.append(Paragraph(c.replace("  ", "&nbsp;&nbsp;"), styles["Body"]))
            story.append(Spacer(1, 0.10 * inch))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs", help="Output directory (default: docs)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "fonts"), exist_ok=True)

    body_font, mono_font = register_unicode_fonts(args.out)

    usage_pdf = os.path.join(args.out, "Usage_Guide.pdf")
    dev_pdf   = os.path.join(args.out, "Development_Guide.pdf")
    deriv_pdf = os.path.join(args.out, "LC_PDE_Derivation.pdf")

    # ---------- Usage Guide ----------
    usage_sections = [
        ("Quickstart", [
            {"type": "text", "content": "Install the environment and run examples."},
            {"type": "text", "content": "GPU:"},
            {"type": "code", "content": "conda env create -f environment.yml\nconda activate lc_soliton\nmake install"},
            {"type": "text", "content": "CPU:"},
            {"type": "code", "content": "conda env create -f environment-cpu.yml\nconda activate lc_soliton_cpu\nmake install"},
            {"type": "text", "content": "Example run:"},
            {"type": "code", "content": "python examples/run_theta2d.py --Nx 128 --Ny 128 --xaper 10.0 \\\n  --steps 500 --dt 1e-3 --b 1.0 --bi 0.3 --intensity 1.0 \\\n  --mobility 4.0 --save theta_out.npz"},
        ]),
        ("Physical Time & Mobility", [
            {"type": "text", "content":
             "mobility ≡ (K/γ<sub>1</sub>) · (4/d²) [s⁻¹]. Set dt in seconds."},
            {"type": "text", "content":
             "Example: K = 10 pN, γ<sub>1</sub> = 0.1 Pa·s, d = 10 μm → mobility ≈ 4.0"},
        ]),

    ]
    build_pdf(usage_pdf, "LC Soliton Simulator — Usage Guide", usage_sections, body_font, mono_font)

    # ---------- Development Guide ----------
    dev_sections = [
        ("Repo layout", [
            {"type": "code", "content": "lc_soliton/    # package modules\nexamples/      # runnable examples\nscripts/       # utilities"},
        ]),
        ("Notebooks", [
            {"type": "text", "content": "Import the notebook API and restart the kernel after editable installs."},
            {"type": "code", "content": "from lc_soliton import *\npython -m pip install -e ."},
        ]),
        ("Docs pipeline", [
            {"type": "text", "content": "Build the PDFs with:"},
            {"type": "code", "content": "make docs"},
        ]),
    ]
    build_pdf(dev_pdf, "LC Soliton Simulator — Development Guide", dev_sections, body_font, mono_font)

        # ---------- Governing Equations (Sketch) ----------
    deriv_sections = [
        ("Transient PDE", [
            {"type": "text", "content":
             "Let θ(x,y,t) be the director tilt. A common transient form is:"},
            {"type": "text", "content":
             "(γ<sub>1</sub>/K) ∂θ/∂t = ∇²θ + (ε<sub>0</sub> Δε<sub>RF</sub> E²)/(2K) · sin(2θ) + "
             "(ε<sub>0</sub> n<sub>a</sub>² |E<sub>op</sub>|²)/(4K) · sin(2θ)"},
        ]),
        ("Dimensionless Form", [
            {"type": "text", "content":
             "∂θ/∂t' = ∇²θ + b · sin(2θ) + b<sub>i</sub> I(x,y) · sin(2θ)"},
        ]),
        ("Notes", [
            {"type": "text", "content":
             "For exact LaTeX, keep equations in the GitHub README (MathJax)."},
        ]),
    ]

    build_pdf(deriv_pdf, "LC Soliton Simulator — Governing Equations (Sketch)", deriv_sections, body_font, mono_font)

    print("Wrote:\n  -", usage_pdf, "\n  -", dev_pdf, "\n  -", deriv_pdf)


if __name__ == "__main__":
    main()
