"""Static checks only. This script never invokes a TeX compiler."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ["main.tex", "PAPER_PHAT_HIEN_VAT_CAN.tex", "paper_preamble.tex", "refs.bib", "build.ps1"]

def uncommented(source: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in source.splitlines())

def check() -> list[str]:
    issues: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file(): issues.append(f"Missing file: {name}")
    if issues: return issues
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    paper = (ROOT / "PAPER_PHAT_HIEN_VAT_CAN.tex").read_text(encoding="utf-8")
    preamble = (ROOT / "paper_preamble.tex").read_text(encoding="utf-8")
    for name in re.findall(r"\\input\{([^}]+)\}", uncommented(main + "\n" + paper)):
        candidate = ROOT / name
        if not candidate.suffix: candidate = candidate.with_suffix(".tex")
        if not candidate.is_file(): issues.append(f"Missing input: {name}")
    for path in [ROOT / "main.tex", ROOT / "PAPER_PHAT_HIEN_VAT_CAN.tex", ROOT / "paper_preamble.tex"]:
        code = uncommented(path.read_text(encoding="utf-8"))
        if code.count("{") != code.count("}"): issues.append(f"Unbalanced braces: {path.name}")
        if sorted(re.findall(r"\\begin\{([^}]+)\}", code)) != sorted(re.findall(r"\\end\{([^}]+)\}", code)):
            issues.append(f"Unbalanced environments: {path.name}")
    if "TeX Gyre Termes" not in preamble or "newtxtext" not in preamble:
        issues.append("Missing cloud-safe fonts")
    if r"\begin{abstract}" not in paper or r"\bibliography{refs}" not in paper:
        issues.append("Paper lacks abstract or BibTeX hook")
    if "Kết quả TEST cuối" not in paper or "FINAL\\_LOCKED" not in paper:
        issues.append("Paper does not contain the locked TEST result section")
    if not (ROOT / "refs.bib").read_text(encoding="utf-8").strip(): issues.append("refs.bib is empty")
    combined = main + paper + preamble
    for pattern in [r"[A-Za-z]:\\", r"\\write18", "minted", "pythontex"]:
        if re.search(pattern, combined): issues.append(f"Forbidden Prism construct: {pattern}")
    return issues

if __name__ == "__main__":
    errors = check()
    print(f"STATIC LINT: {'FAIL' if errors else 'PASS'}; checked {len(REQUIRED)} required files; TeX not compiled")
    for error in errors: print("-", error)
    raise SystemExit(bool(errors))
