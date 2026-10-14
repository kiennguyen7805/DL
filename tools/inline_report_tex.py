from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "docs" / "reports" / "BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU"
MASTER = ROOT / "BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU.tex"


def expand(text: str, stack: tuple[Path, ...] = ()) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        path = (ROOT / name).resolve()
        if path in stack:
            raise RuntimeError(f"Vòng lặp input: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        body = path.read_text(encoding="utf-8")
        return (
            f"\n% ===== BEGIN INLINED: {name} =====\n"
            + expand(body, stack + (path,))
            + f"\n% ===== END INLINED: {name} =====\n"
        )

    return re.sub(r"\\input\{([^}]+\.tex)\}", repl, text)


source = MASTER.read_text(encoding="utf-8")
expanded = expand(source, (MASTER.resolve(),))
if re.search(r"\\input\{[^}]+\.tex\}", expanded):
    raise RuntimeError("Vẫn còn input .tex sau khi gộp")
MASTER.write_text(expanded, encoding="utf-8", newline="\n")
MAIN = ROOT / "main.tex"
MAIN.write_text(expanded, encoding="utf-8", newline="\n")
print(MASTER)
print(MAIN)
