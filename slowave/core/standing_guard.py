"""Injection hygiene for the standing tier: text normalisation and deny patterns.

IMPORTANT - defence in depth, NOT a guarantee.  A regex/lexicon filter over free text cannot
decide intent: a determined author can always paraphrase past it (new languages, indirection,
splitting an instruction across several rules).  It exists to remove the cheap, known attack
shapes (override phrases in several languages, unicode obfuscation, role/chat-template tokens,
secret exfiltration, arbitrary-exec primitives, tool/lifecycle manipulation, review bypass) and
to raise the cost of the rest.  The real controls are structural: delivery is capped and
labelled "verify, do not execute", only the canonical user-global scope crosses projects,
`always` is quota-limited, and cross-project retirement is project-local.

Matching runs on a *folded* copy of the text: NFKD, combining marks / zero-width / bidi /
control characters removed, case-folded, and common Cyrillic/Greek confusables mapped to
Latin.  The stored rule text is never altered.
"""

from __future__ import annotations

import re
import unicodedata

# Cyrillic / Greek letters that render like Latin ones (lower-case, post-casefold).
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "к": "k",
        "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ɡ": "g", "ո": "n", "ս": "u",
        "ν": "v", "ο": "o", "ρ": "p", "α": "a", "ε": "e", "ι": "i", "κ": "k", "τ": "t",
        "м": "m", "н": "h", "т": "t", "в": "b", "ь": "b", "ѵ": "v", "ӏ": "l",
    }
)  # fmt: skip


def fold(text: str) -> str:
    """Canonical form used only for matching (see module docstring)."""
    value = unicodedata.normalize("NFKD", str(text or ""))
    kept = []
    for ch in value:
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Me", "Cf", "Co", "Cs"):
            continue  # accents, zero-width, bidi, private use
        if cat == "Cc" and ch not in "\n\t\r":
            continue
        kept.append(ch)
    return "".join(kept).casefold().translate(_CONFUSABLES)


_I = re.IGNORECASE
_OVERRIDE = re.compile(
    r"\b(ignor\w*|disregard\w*|forget|dimentica\w*|olvida\w*|oubli\w*|vergiss\w*|missacht\w*|"
    r"annull\w*|ometti|omite\w*|ne\s+tenez\s+pas\s+compte)\b[^.\n]{0,50}"
    r"\b(instruction\w*|istruzion\w*|instruccion\w*|anweisung\w*|consigne\w*|regole|reglas|"
    r"regles|regeln|rules|prompt|vorherig\w*|precedent\w*|anterior\w*|previous|prior|above|"
    r"earlier|vorige\w*)\b",
    _I,
)
_ROLE = re.compile(
    r"((?:^|[\n.!?;]\s*)(?:system|assistant|developer|admin)\s*:|<\|[a-z_]+\|>|\[/?inst\]|"
    r"<</?sys>>|###\s*(?:system|instruction)|\bsystem\s+(?:prompt|override|message)\b|"
    r"\bnote for (?:ai|assistant|claude)\b|\byou are now\b|\bnew instructions?\s*:)",
    _I,
)
_EXEC = re.compile(
    r"(\bos\.(?:system|popen|exec\w*)\b|\bsubprocess\b|\bpython[0-9.]*\s+-c\b|\bnode\s+-e\b|"
    r"\bperl\s+-e\b|\bruby\s+-e\b|__import__|\beval\s*[(\$`]|\bexec\s*\(|"
    r"\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b|\|\s*python[0-9.]*\b|base64\s+(?:-d|--decode)|"
    r"\bnc\s+-[a-z]*e\b|/dev/tcp/|\bchmod\s+(?:-r\s+)?[0-7]*777\b)",
    _I,
)
_TOOLING = re.compile(
    r"(\bslowave_[a-z]+|\bsch_(?:\d|id|\s)|\bmark\b[^.\n]{0,50}\b(?:stale|superseded|irrelevant)\b|"
    r"\b(?:call|invoke|use|run)\b[^.\n]{0,25}\b(?:the\s+)?\w+\s+(?:tool|function)\b|"
    r"\b(?:assessment|memory_feedback|retrieval_id)\b)",
    _I,
)
_REVIEW_BYPASS = re.compile(
    r"(\bnever\s+(?:review|read|check|inspect)\b|\b(?:always|automatically)\s+approve\b|"
    r"\bapprove\s+(?:any|all|every)\b|\bskip\s+(?:the\s+)?(?:code\s+)?review\b|"
    r"\brubber.?stamp|\bauto.?approve\b|\bwithout\s+(?:reading|looking\s+at)\s+(?:it|them)\b)",
    _I,
)
_SECRET_PATH = re.compile(
    r"(\bid_(?:rsa|ed25519|ecdsa)\b|~?/?\.ssh\b|\.aws/credentials|\.netrc\b|\.npmrc\b|"
    r"\bcredentials(?:\.json)?\b|\.pem\b|private[ _]key|(?<![\w])\.env(?:\.\w+)?\b)",
    _I,
)
_URL = re.compile(r"\bhttps?://|\bftp://|\bwebhook\b", _I)
_EXFIL_VERB = re.compile(
    r"\b(send|post|upload|exfiltrat\w*|curl|wget|paste|include|contents?\s+of|cat|print|echo|"
    r"leak|email|attach)\b",
    _I,
)
_DOCS_HARD = re.compile(
    r"(\b(?:curl|wget)\b[^\n]*\s(?:-d|--data|--upload-file|-t)\s*@|"
    r"\b(?:curl|wget)\b[^\n]*\bhttps?://[^\s]*\s*\|\s*\w+)",
    _I,
)
_PROHIBITION = re.compile(
    r"\b(never|don't|do not|must not|avoid|ask before|confirm before|without asking|"
    r"forbidden|banned|not allowed|before running)\b",
    _I,
)


def injection_reason(text: str) -> str | None:
    """Return a short reason when *text* matches a known injection shape, else None."""
    value = fold(text)
    for label, pat in (
        ("instruction_override", _OVERRIDE),
        ("role_or_template_token", _ROLE),
        ("exec_primitive", _EXEC),
        ("tool_or_lifecycle_manipulation", _TOOLING),
        ("review_bypass", _REVIEW_BYPASS),
        ("exfil_command", _DOCS_HARD),
    ):
        if pat.search(value):
            return label
    if _SECRET_PATH.search(value):
        prohibited = bool(_PROHIBITION.search(value))
        if _URL.search(value) or (_EXFIL_VERB.search(value) and not prohibited):
            return "secret_exfiltration"
    return None
