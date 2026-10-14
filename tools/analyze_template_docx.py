from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W, "a": "http://schemas.openxmlformats.org/drawingml/2006/main", "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}


def q(local: str) -> str:
    return f"{{{W}}}{local}"


def twips(v):
    return round(int(v) / 20, 2) if v is not None else None


def text_of(el):
    return "".join(el.xpath(".//w:t/text()", namespaces=NS)).strip()


def main(docx_path: str, out_dir: str):
    src = Path(docx_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    doc = Document(src)
    report = {"source": str(src), "paragraphs": len(doc.paragraphs), "tables": len(doc.tables), "sections_python_docx": [], "styles": {}, "xml": {}}

    for s in doc.sections:
        report["sections_python_docx"].append({
            "page_width_pt": round(s.page_width.pt, 2), "page_height_pt": round(s.page_height.pt, 2),
            "top_margin_pt": round(s.top_margin.pt, 2), "bottom_margin_pt": round(s.bottom_margin.pt, 2),
            "left_margin_pt": round(s.left_margin.pt, 2), "right_margin_pt": round(s.right_margin.pt, 2),
            "header_distance_pt": round(s.header_distance.pt, 2), "footer_distance_pt": round(s.footer_distance.pt, 2),
            "start_type": str(s.start_type), "different_first_page_header_footer": s.different_first_page_header_footer,
            "header_text": " | ".join(p.text for p in s.header.paragraphs if p.text.strip()),
            "footer_text": " | ".join(p.text for p in s.footer.paragraphs if p.text.strip()),
        })

    for style in doc.styles:
        if getattr(style, "type", None) is None:
            continue
        pf = getattr(style, "paragraph_format", None)
        font = getattr(style, "font", None)
        report["styles"][style.name] = {
            "type": str(style.type), "base": getattr(getattr(style, "base_style", None), "name", None),
            "font": getattr(font, "name", None), "size_pt": round(font.size.pt, 2) if font and font.size else None,
            "bold": getattr(font, "bold", None), "italic": getattr(font, "italic", None),
            "alignment": str(getattr(pf, "alignment", None)) if pf else None,
            "line_spacing": str(getattr(pf, "line_spacing", None)) if pf else None,
            "space_before_pt": round(pf.space_before.pt, 2) if pf and pf.space_before else None,
            "space_after_pt": round(pf.space_after.pt, 2) if pf and pf.space_after else None,
            "first_line_indent_pt": round(pf.first_line_indent.pt, 2) if pf and pf.first_line_indent else None,
            "left_indent_pt": round(pf.left_indent.pt, 2) if pf and pf.left_indent else None,
        }

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        report["package_parts"] = [{"name": n, "size": z.getinfo(n).file_size} for n in names]
        root = etree.fromstring(z.read("word/document.xml"))
        body = root.find("w:body", NS)
        styles_root = etree.fromstring(z.read("word/styles.xml"))
        settings = etree.fromstring(z.read("word/settings.xml")) if "word/settings.xml" in names else None
        app = etree.fromstring(z.read("docProps/app.xml")) if "docProps/app.xml" in names else None
        core = etree.fromstring(z.read("docProps/core.xml")) if "docProps/core.xml" in names else None

        app_vals = {}
        if app is not None:
            for ch in app:
                app_vals[etree.QName(ch).localname] = ch.text
        report["app_properties"] = app_vals
        report["core_properties"] = {etree.QName(ch).localname: ch.text for ch in core} if core is not None else {}

        defaults = {}
        for tag in ("docDefaults", "latentStyles"):
            els = styles_root.xpath(f"./w:{tag}", namespaces=NS)
            if els:
                defaults[tag] = etree.tostring(els[0], encoding="unicode")
        report["xml"]["style_defaults"] = defaults
        report["xml"]["settings"] = etree.tostring(settings, encoding="unicode") if settings is not None else None

        paragraphs = []
        page = 1
        headings = []
        page_words = Counter()
        style_counts = Counter()
        section_marks = []
        for idx, p in enumerate(body.xpath(".//w:p", namespaces=NS)):
            txt = text_of(p)
            ppr = p.find("w:pPr", NS)
            pstyle = None
            if ppr is not None and ppr.find("w:pStyle", NS) is not None:
                pstyle = ppr.find("w:pStyle", NS).get(q("val"))
            breaks_before = len(p.xpath(".//w:lastRenderedPageBreak", namespaces=NS)) + len(p.xpath(".//w:br[@w:type='page']", namespaces=NS))
            if breaks_before:
                page += breaks_before
            words = len(re.findall(r"\b[\wÀ-ỹĐđ]+\b", txt, flags=re.UNICODE))
            page_words[page] += words
            style_counts[pstyle or "(none)"] += 1
            numpr = p.xpath("./w:pPr/w:numPr", namespaces=NS)
            sectpr = p.xpath("./w:pPr/w:sectPr", namespaces=NS)
            rec = {"index": idx, "page_est": page, "style": pstyle, "text": txt, "words": words, "numbered": bool(numpr), "section_break": bool(sectpr)}
            paragraphs.append(rec)
            if pstyle and ("Heading" in pstyle or pstyle.startswith("Title") or pstyle.startswith("Tieu") or pstyle.startswith("Chng")):
                headings.append(rec)
            if sectpr:
                section_marks.append({"paragraph": idx, "page_est": page, "text": txt})

        sects = []
        for i, sp in enumerate(root.xpath("//w:sectPr", namespaces=NS)):
            pg = sp.find("w:pgSz", NS); mar = sp.find("w:pgMar", NS); typ = sp.find("w:type", NS)
            sects.append({
                "index": i,
                "width_pt": twips(pg.get(q("w"))) if pg is not None else None,
                "height_pt": twips(pg.get(q("h"))) if pg is not None else None,
                "orient": pg.get(q("orient")) if pg is not None else None,
                "margins_pt": {k: twips(mar.get(q(k))) for k in ["top","right","bottom","left","header","footer","gutter"]} if mar is not None else {},
                "type": typ.get(q("val")) if typ is not None else "nextPage",
                "titlePg": sp.find("w:titlePg", NS) is not None,
                "header_refs": [dict(x.attrib) for x in sp.findall("w:headerReference", NS)],
                "footer_refs": [dict(x.attrib) for x in sp.findall("w:footerReference", NS)],
                "page_num_type": dict(sp.find("w:pgNumType", NS).attrib) if sp.find("w:pgNumType", NS) is not None else None,
            })

        headers_footers = {}
        for n in names:
            if re.fullmatch(r"word/(header|footer)\d+\.xml", n):
                rr = etree.fromstring(z.read(n))
                headers_footers[n] = {"text": text_of(rr), "fields": rr.xpath(".//w:instrText/text()", namespaces=NS), "xml": etree.tostring(rr, encoding="unicode")}

        report["xml"].update({
            "paragraphs": paragraphs, "headings": headings, "style_counts": style_counts,
            "page_words_est": dict(page_words), "page_count_est_from_breaks": page,
            "sections": sects, "section_marks": section_marks, "headers_footers": headers_footers,
            "fields": root.xpath(".//w:instrText/text()", namespaces=NS),
            "bookmarks": [x.get(q("name")) for x in root.xpath(".//w:bookmarkStart", namespaces=NS)],
            "drawings": len(root.xpath(".//w:drawing", namespaces=NS)),
            "pictures_vml": len(root.xpath(".//w:pict", namespaces=NS)),
            "tables": len(root.xpath(".//w:tbl", namespaces=NS)),
            "footnotes": "word/footnotes.xml" in names,
            "endnotes": "word/endnotes.xml" in names,
            "comments": "word/comments.xml" in names,
            "numbering_present": "word/numbering.xml" in names,
        })

        # Extract all readable text, with table-cell content included in document order.
        full_lines = []
        for child in body:
            tag = etree.QName(child).localname
            if tag == "p":
                t = text_of(child)
                if t:
                    full_lines.append(t)
            elif tag == "tbl":
                for tr in child.findall("w:tr", NS):
                    cells = [text_of(tc) for tc in tr.findall("w:tc", NS)]
                    full_lines.append(" | ".join(cells))
        (out / "template_text.txt").write_text("\n".join(full_lines), encoding="utf-8")

    (out / "template_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: dict(x)), encoding="utf-8")
    print(json.dumps({"output": str(out), "app_pages": report["app_properties"].get("Pages"), "app_words": report["app_properties"].get("Words"), "estimated_pages": report["xml"]["page_count_est_from_breaks"], "paragraphs": len(report["xml"]["paragraphs"]), "tables": report["xml"]["tables"], "drawings": report["xml"]["drawings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
