#!/usr/bin/env python3
"""Make field-preserving Word copies with individually tracked text edits.

Only document.xml and (when needed) settings.xml are changed. Existing field
instructions, equations, styles, relationships and prior revision elements
remain. Rejection of this run's new revision IDs is checked against the
original visible text before a copy is written.
"""

from __future__ import annotations
import copy
import datetime
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W, "m": M}
XMLSPACE = "{http://www.w3.org/XML/1998/namespace}space"


def q(name):
    return "{" + W + "}" + name


def visible_nodes(paragraph):
    nodes = []
    field_depth = 0
    for node in paragraph.iter():
        if node.tag == q("fldChar"):
            kind = node.get(q("fldCharType"))
            if kind == "begin":
                field_depth += 1
            elif kind == "end":
                field_depth = max(0, field_depth - 1)
        if node.tag == q("t") and not any(
            a.tag == q("del") for a in node.iterancestors()
        ):
            protected = field_depth > 0 or any(
                a.tag in {q("fldSimple"), "{" + M + "}oMath"}
                for a in node.iterancestors()
            )
            nodes.append((node, protected))
    return nodes


def visible(paragraph):
    return "".join(n.text or "" for n, _ in visible_nodes(paragraph))


def protected_parts(root):
    return {
        name: [ET.tostring(n, method="c14n") for n in root.findall(path, NS)]
        for name, path in [
            ("field_instructions", ".//w:instrText"),
            ("field_markers", ".//w:fldChar"),
            ("equations", ".//m:oMath"),
            ("simple_fields", ".//w:fldSimple"),
        ]
    }


class TrackedEditor:
    def __init__(self, root):
        self.root = root
        self.paragraphs = root.findall(".//w:p", NS)
        prior = [
            int(n.get(q("id"))) for n in root.iter() if n.get(q("id"), "").isdigit()
        ]
        self.next_id = max(prior, default=0) + 1
        self.new_ids = set()
        self.changes = []
        self.date = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def revision(self, kind, run):
        node = ET.Element(q(kind))
        rid = str(self.next_id)
        self.next_id += 1
        self.new_ids.add(rid)
        node.set(q("id"), rid)
        node.set(q("author"), "Codex")
        node.set(q("date"), self.date)
        node.append(run)
        return node

    def run(self, style, text, deleted=False):
        run = ET.Element(q("r"))
        if style is not None:
            run.append(copy.deepcopy(style))
        node = ET.SubElement(run, q("delText" if deleted else "t"))
        node.set(XMLSPACE, "preserve")
        node.text = text
        return run

    def replace(self, index, old, new, reason):
        paragraph = self.paragraphs[index]
        text = visible(paragraph)
        if text.count(old) != 1:
            raise ValueError(
                f"paragraph {index}: expected one match for {old[:100]!r}; found {text.count(old)}"
            )
        start, end = text.index(old), text.index(old) + len(old)
        offset = 0
        replacements = []
        for node, protected in visible_nodes(paragraph):
            value = node.text or ""
            left, right = max(0, start - offset), min(len(value), end - offset)
            if left < right:
                if protected:
                    raise ValueError(
                        f"paragraph {index}: edit would alter a citation/field"
                    )
                run = node.getparent()
                if run.tag != q("r"):
                    raise ValueError(
                        f"paragraph {index}: edit intersects a non-text run"
                    )
                replacements.append(
                    (node, value[:left], value[left:right], value[right:])
                )
            offset += len(value)
        for i, (node, prefix, deleted, suffix) in enumerate(replacements):
            run = node.getparent()
            style = run.find("w:rPr", NS)
            parent = run.getparent()
            position = parent.index(run)
            children = [c for c in run if c.tag != q("rPr")]
            text_pos = children.index(node)
            before_run = ET.Element(q("r"), attrib=dict(run.attrib))
            after_run = ET.Element(q("r"), attrib=dict(run.attrib))
            if style is not None:
                before_run.append(copy.deepcopy(style))
                after_run.append(copy.deepcopy(style))
            # Move rather than clone other text/field/break nodes so subsequent
            # span pieces keep their identity and field structure is preserved.
            for child in children[:text_pos]:
                before_run.append(child)
            for child in children[text_pos + 1 :]:
                after_run.append(child)
            additions = []
            if len(before_run) > (1 if style is not None else 0):
                additions.append(before_run)
            if prefix:
                additions.append(self.run(style, prefix))
            additions.append(self.revision("del", self.run(style, deleted, True)))
            if i == 0 and new:
                additions.append(self.revision("ins", self.run(style, new)))
            if suffix:
                additions.append(self.run(style, suffix))
            if len(after_run) > (1 if style is not None else 0):
                additions.append(after_run)
            parent.remove(run)
            for n in additions:
                parent.insert(position, n)
                position += 1
        expected = text[:start] + new + text[end:]
        if visible(paragraph) != expected:
            raise AssertionError(f"paragraph {index}: visible replacement mismatch")
        self.changes.append(
            {"paragraph_index": index, "old": old, "new": new, "reason": reason}
        )

    def reject_new(self):
        root = copy.deepcopy(self.root)
        for node in list(root.iter())[::-1]:
            if (
                node.tag not in {q("ins"), q("del")}
                or node.get(q("id")) not in self.new_ids
            ):
                continue
            parent = node.getparent()
            if node.tag == q("ins"):
                parent.remove(node)
            else:
                for t in node.iter(q("delText")):
                    t.tag = q("t")
                pos = parent.index(node)
                for child in list(node):
                    node.remove(child)
                    parent.insert(pos, child)
                    pos += 1
                parent.remove(node)
        return root


def revise(source, output, patches, unchanged_paragraphs=()):
    source, output = Path(source), Path(output)
    with ZipFile(source) as z:
        entries = {
            info.filename: (info, z.read(info.filename)) for info in z.infolist()
        }
    original = ET.fromstring(entries["word/document.xml"][1])
    editor = TrackedEditor(copy.deepcopy(original))
    before = protected_parts(original)
    prior_revisions = [
        (n.tag, n.get(q("id")), n.get(q("author")), n.get(q("date")))
        for n in original.iter()
        if n.tag in {q("ins"), q("del")}
    ]
    for index, old, new, reason in patches:
        editor.replace(index, old, new, reason)
    if before != protected_parts(editor.root):
        raise AssertionError("field/equation structure changed")
    rejected = editor.reject_new()
    if [visible(p) for p in rejected.findall(".//w:p", NS)] != [
        visible(p) for p in original.findall(".//w:p", NS)
    ]:
        raise AssertionError("rejecting new revisions does not restore original text")
    after_prior = [
        (n.tag, n.get(q("id")), n.get(q("author")), n.get(q("date")))
        for n in editor.root.iter()
        if n.tag in {q("ins"), q("del")} and n.get(q("id")) not in editor.new_ids
    ]
    if prior_revisions != after_prior:
        raise AssertionError("prior revision identities changed")
    for index in unchanged_paragraphs:
        if ET.tostring(editor.paragraphs[index], method="c14n") != ET.tostring(
            original.findall(".//w:p", NS)[index], method="c14n"
        ):
            raise AssertionError(f"deferred paragraph changed: {index}")
    document = ET.tostring(
        editor.root, encoding="UTF-8", xml_declaration=True, standalone=True
    )
    replacements = {"word/document.xml": document}
    if "word/settings.xml" in entries:
        settings = ET.fromstring(entries["word/settings.xml"][1])
        if settings.find("w:trackRevisions", NS) is None:
            settings.append(ET.Element(q("trackRevisions")))
            replacements["word/settings.xml"] = ET.tostring(
                settings, encoding="UTF-8", xml_declaration=True, standalone=True
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w") as z:
        for name, (info, data) in entries.items():
            z.writestr(info, replacements.get(name, data))
    with ZipFile(output) as z:
        assert z.testzip() is None
        for name, (_, data) in entries.items():
            if name not in replacements and z.read(name) != data:
                raise AssertionError(f"unrelated DOCX part changed: {name}")
    receipt = {
        "source": str(source),
        "output": str(output),
        "changes": editor.changes,
        "new_revision_ids": sorted(editor.new_ids, key=int),
        "preserved_field_instructions": len(before["field_instructions"]),
        "preserved_equations": len(before["equations"]),
        "preserved_prior_revisions": len(prior_revisions),
        "reject_new_restores_original": True,
        "unchanged_paragraph_indices": list(unchanged_paragraphs),
    }
    output.with_suffix(".changes.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2)
    )
    output.with_suffix(".final.txt").write_text(
        "\n\n".join(visible(p) for p in editor.paragraphs if visible(p).strip())
    )
    return receipt
