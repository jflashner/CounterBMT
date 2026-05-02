from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = (
    REPO_ROOT
    / "outputs"
    / "waymax_sdc_postsplit_semantics_top1000_prep_local_goldmatch_rerendered_vlmrisk_gpt54_original_copy"
    / "examples"
    / "waymax_scene_01332__sdc_1698__t_000"
)

RISK_STYLES = {
    "low": "risklow",
    "medium": "riskmedium",
    "high": "riskhigh",
    "critical": "riskcritical",
}

RISK_EDGE_COLORS = {
    "low": "risklowcolor",
    "medium": "riskmediumcolor",
    "high": "riskhighcolor",
    "critical": "riskcriticalcolor",
}


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def wrapped_tex(value: Any, width: int = 34, max_lines: int = 3) -> str:
    text = " ".join(str(value or "").split())
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    return r"\\ ".join(latex_escape(line) for line in lines[:max_lines])


def load_slot(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    highlighted = payload.get("highlighted_paths") or []
    if not highlighted:
        raise ValueError(f"No highlighted_paths in {path}")
    slot = dict(highlighted[0])
    slot["scenario_id"] = payload.get("scenario_id", "")
    slot["example_id"] = payload.get("example_id", "")
    slot["sdc_id"] = payload.get("sdc_id", "")
    slot["current_time_index"] = payload.get("current_time_index", 0)
    return slot


def load_slots(example_dir: Path) -> List[Dict[str, Any]]:
    paths = [example_dir / "contract_normalized_gt.json"]
    paths.extend(sorted(example_dir.glob("contract_normalized_alt_*.json")))
    slots = [load_slot(path) for path in paths if path.exists()]
    if len(slots) < 2:
        raise FileNotFoundError(f"Expected GT plus alternatives in {example_dir}")
    return slots


def slot_node(slot: Dict[str, Any], idx: int, y: float) -> str:
    slot_id = str(slot.get("slot_id") or "")
    source = "GT" if slot_id == "gt" else slot_id.replace("_", " ").upper()
    label = str(slot.get("semantic_label") or "unknown").replace("_", " ")
    risk = str(slot.get("risk_level") or "unknown").lower()
    confidence = float(slot.get("confidence") or 0.0)
    style = RISK_STYLES.get(risk, "riskunknown")
    node_name = f"slot{idx}"
    return (
        rf"\node[{style}] ({node_name}) at (7.00,{y:.2f}) "
        rf"{{\footnotesize\textbf{{{latex_escape(source)}}}\\ {latex_escape(label)}\\ "
        rf"\scriptsize risk: {latex_escape(risk)} \quad conf: {confidence:.2f}}};"
    )


def risk_node(
    slot: Dict[str, Any],
    idx: int,
    y: float,
    *,
    x: float = 11.20,
    evidence_font: str = r"\tiny",
    wrap_width: int = 42,
    max_lines: int = 3,
) -> str:
    rationale = slot.get("risk_rationale_short") or slot.get("rationale_short") or ""
    return (
        rf"\node[evidence] (risk{idx}) at ({x:.2f},{y:.2f}) "
        rf"{{{evidence_font} {wrapped_tex(rationale, width=wrap_width, max_lines=max_lines)}}};"
    )


def build_tex(slots: List[Dict[str, Any]], *, large_evidence: bool = False) -> str:
    first = slots[0]
    scenario_id = latex_escape(first.get("scenario_id", ""))
    sdc_id = latex_escape(first.get("sdc_id", ""))
    t_idx = latex_escape(first.get("current_time_index", 0))
    y_values = [5.05, 3.82, 2.59, 1.36]
    evidence_x = 11.55 if large_evidence else 11.20
    outcome_x = 16.45 if large_evidence else 15.45
    panel_right = 17.50 if large_evidence else 16.45
    evidence_width = "5.20cm" if large_evidence else "4.20cm"
    evidence_text_width = "4.90cm" if large_evidence else "3.95cm"
    evidence_min_height = "0.96cm" if large_evidence else "0.82cm"
    evidence_font = r"\footnotesize" if large_evidence else r"\tiny"
    evidence_wrap_width = 37 if large_evidence else 42
    evidence_max_lines = 4 if large_evidence else 3
    outcome_width = "1.75cm" if large_evidence else "1.50cm"
    outcome_height = "0.70cm" if large_evidence else "0.58cm"
    outcome_font = r"\sffamily\scriptsize" if large_evidence else r"\sffamily\tiny"

    slot_nodes = "\n".join(slot_node(slot, i, y_values[i]) for i, slot in enumerate(slots))
    risk_nodes = "\n".join(
        risk_node(
            slot,
            i,
            y_values[i],
            x=evidence_x,
            evidence_font=evidence_font,
            wrap_width=evidence_wrap_width,
            max_lines=evidence_max_lines,
        )
        for i, slot in enumerate(slots)
    )
    out_angles = [32, 8, -8, -32]
    slot_edges = "\n".join(
        rf"\draw[edge] (decision.east) to[out={out_angles[i] if i < len(out_angles) else 0},in=180] (slot{i}.west);"
        for i in range(len(slots))
    )
    risk_edges = "\n".join(
        rf"\draw[edge, draw={{{RISK_EDGE_COLORS.get(str(slot.get('risk_level') or 'unknown').lower(), 'riskunknowncolor')}}}] "
        rf"(slot{i}.east) -- (risk{i}.west);"
        for i, slot in enumerate(slots)
    )
    outcome_edges = "\n".join(
        "\n".join(
            rf"\draw[thinflow] (risk{i}.east) to[out=0,in=180] ({outcome}.west);"
            for outcome in ("collision", "compliance", "progress")
        )
        for i in range(len(slots))
    )

    return rf"""\documentclass[tikz,border=3pt]{{standalone}}
\usetikzlibrary{{arrows.meta,positioning}}
\definecolor{{textcolor}}{{HTML}}{{0F172A}}
\definecolor{{muted}}{{HTML}}{{475569}}
\definecolor{{panel}}{{HTML}}{{F8FAFC}}
\definecolor{{edgecolor}}{{HTML}}{{334155}}
\definecolor{{risklowcolor}}{{HTML}}{{10B981}}
\definecolor{{riskmediumcolor}}{{HTML}}{{F59E0B}}
\definecolor{{riskhighcolor}}{{HTML}}{{F43F5E}}
\definecolor{{riskcriticalcolor}}{{HTML}}{{BE123C}}
\definecolor{{riskunknowncolor}}{{HTML}}{{94A3B8}}
\tikzset{{
  base/.style={{rounded corners=2pt, align=center, draw=edgecolor, line width=0.45pt, inner sep=3.0pt, font=\sffamily\scriptsize, text=textcolor}},
  context/.style={{base, fill=cyan!12, draw=cyan!55!black, minimum width=2.05cm, minimum height=0.72cm}},
  decision/.style={{base, fill=blue!12, draw=blue!70!black, minimum width=2.45cm, minimum height=0.82cm}},
  risklow/.style={{base, fill=risklowcolor, draw=white, text=white, minimum width=2.95cm, minimum height=0.82cm}},
  riskmedium/.style={{base, fill=riskmediumcolor, draw=white, text=white, minimum width=2.95cm, minimum height=0.82cm}},
  riskhigh/.style={{base, fill=riskhighcolor, draw=white, text=white, minimum width=2.95cm, minimum height=0.82cm}},
  riskcritical/.style={{base, fill=riskcriticalcolor, draw=white, text=white, minimum width=2.95cm, minimum height=0.82cm}},
  riskunknown/.style={{base, fill=riskunknowncolor, draw=white, text=white, minimum width=2.95cm, minimum height=0.82cm}},
  evidence/.style={{base, fill=white, draw=gray!25, minimum width={evidence_width}, text width={evidence_text_width}, minimum height={evidence_min_height}}},
  outcome/.style={{base, fill=red!8, draw=red!45, minimum width={outcome_width}, minimum height={outcome_height}, font={outcome_font}}},
  edge/.style={{-Latex, draw=edgecolor, line width=0.55pt}},
  thinflow/.style={{-Latex, draw=edgecolor, opacity=0.42, line width=0.36pt}},
}}
\begin{{document}}
\begin{{tikzpicture}}[x=1cm,y=1cm]
  \fill[panel] (-0.35,0.35) rectangle ({panel_right:.2f},6.45);

  \node[anchor=west, font=\sffamily\bfseries\large, text=textcolor] at (-0.18,6.23)
    {{Expanded CounterDrive DAG for one local decision}};
  \node[anchor=west, font=\sffamily\scriptsize, text=muted] at (-0.18,5.93)
    {{{scenario_id} \quad SDC={sdc_id} \quad t={t_idx} \quad factual branch + {len(slots) - 1} alternatives}};

  \node[context] (scene) at (1.05,4.70) {{Logged scene\\ map + agents\\ history}};
  \node[context] (branch) at (1.05,3.45) {{Local branch\\ geometry\\ valid path family}};
  \node[context] (contract) at (1.05,2.20) {{VLM contract\\ semantic label\\ risk rationale}};

  \node[decision] (decision) at (3.85,3.45) {{\textbf{{Decision}}\\ do(path\_choice=$y$)}};

  \draw[edge] (scene.east) to[out=0,in=155] (decision.west);
  \draw[edge] (branch.east) -- (decision.west);
  \draw[edge] (contract.east) to[out=0,in=205] (decision.west);

{slot_nodes}

{risk_nodes}

{slot_edges}

{risk_edges}

  \node[outcome] (collision) at ({outcome_x:.2f},4.70) {{collision\\ near miss}};
  \node[outcome] (compliance) at ({outcome_x:.2f},3.45) {{signal\\ compliance}};
  \node[outcome] (progress) at ({outcome_x:.2f},2.20) {{progress\\ interaction order}};

{outcome_edges}

  \node[font=\sffamily\scriptsize\bfseries, text=muted] at (1.05,0.65) {{context}};
  \node[font=\sffamily\scriptsize\bfseries, text=muted] at (3.85,0.65) {{intervention}};
  \node[font=\sffamily\scriptsize\bfseries, text=muted] at (7.00,0.65) {{expanded alternatives}};
  \node[font=\sffamily\scriptsize\bfseries, text=muted] at ({evidence_x:.2f},0.65) {{risk evidence}};
  \node[font=\sffamily\scriptsize\bfseries, text=muted] at ({outcome_x:.2f},0.65) {{outcomes}};
\end{{tikzpicture}}
\end{{document}}
"""


def compile_tex(tex_path: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        return
    subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> None:
    slots = load_slots(EXAMPLE_DIR)
    tex_path = FIGURE_DIR / "dag_expanded_maneuvers_example.tex"
    tex_path.write_text(build_tex(slots), encoding="utf-8")
    compile_tex(tex_path)
    print(f"Wrote {tex_path}")
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"Wrote {pdf_path}")

    large_tex_path = FIGURE_DIR / "dag_expanded_maneuvers_example_large_evidence.tex"
    large_tex_path.write_text(build_tex(slots, large_evidence=True), encoding="utf-8")
    compile_tex(large_tex_path)
    print(f"Wrote {large_tex_path}")
    large_pdf_path = large_tex_path.with_suffix(".pdf")
    if large_pdf_path.exists():
        print(f"Wrote {large_pdf_path}")


if __name__ == "__main__":
    main()
