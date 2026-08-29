"""Explain a prediction: what the class means, which syntactic markers fired,
and which lines the model actually leaned on.

Kept separate from inference.py so the CLI and the browser demo describe a
prediction the same way.
"""
import re
from typing import Callable, List, Tuple

VULN_INFO = {
    "Reentrancy": {
        "summary": "An external call hands control to another contract before this "
                   "one finishes updating its own state.",
        "why": "The callee can call straight back in while the first call is still "
               "mid-flight, re-running the withdrawal against a balance that has not "
               "been decremented yet, and drain the contract in a loop. This is the "
               "bug behind the 2016 DAO hack.",
        "fix": "Apply checks-effects-interactions: update state *before* making the "
               "external call. Add a reentrancy guard (OpenZeppelin's "
               "ReentrancyGuard) for anything holding funds.",
    },
    "Integer Overflow": {
        "summary": "Arithmetic can exceed the range of its type and wrap around, so a "
                   "balance or supply silently becomes a wildly wrong number.",
        "why": "Before Solidity 0.8, `uint256` arithmetic wrapped silently: subtracting "
               "1 from 0 yields 2^256-1. An attacker who can drive a balance negative "
               "gets an enormous one instead.",
        "fix": "Compile with Solidity >= 0.8, where arithmetic is checked and reverts "
               "on overflow. On older compilers use SafeMath for every operation.",
    },
    "Timestamp Dependency": {
        "summary": "Contract logic branches on `block.timestamp` (or `now`), which the "
                   "block producer can nudge.",
        "why": "Validators have some latitude over the timestamp they publish. Any "
               "payout, deadline, or 'random' outcome derived from it can be steered, "
               "and it is never a safe source of randomness.",
        "fix": "Do not use timestamps for randomness -- use a VRF or commit-reveal. "
               "Where time is genuinely needed, use block numbers or allow enough "
               "tolerance that a small shift cannot change the outcome.",
    },
    "Dangerous Delegatecall": {
        "summary": "`delegatecall` runs another contract's code against *this* "
                   "contract's storage.",
        "why": "If the target address is attacker-controlled or its layout does not "
               "match, the callee can overwrite any storage slot -- including the owner "
               "-- or `selfdestruct` the caller. The Parity multisig freeze came from "
               "this.",
        "fix": "Only delegatecall to a trusted, immutable address. Never take the "
               "target from user input. Keep storage layouts aligned when using a "
               "proxy pattern.",
    },
}

# Markers validated against the dataset: three classes have a near-perfect
# syntactic signature, Integer Overflow has none of its own (see README).
MARKERS = [
    ("Reentrancy", "low-level call forwarding value", re.compile(r"\.call\s*\.\s*value\s*\(")),
    ("Reentrancy", "low-level call with value option", re.compile(r"\.call\s*\{[^}]*value\s*:")),
    ("Dangerous Delegatecall", "delegatecall", re.compile(r"\.delegatecall\s*\(")),
    ("Timestamp Dependency", "block.timestamp", re.compile(r"block\s*\.\s*timestamp")),
    ("Timestamp Dependency", "now", re.compile(r"(?<![\w.])now(?![\w])")),
]

SAFE_ARITHMETIC = re.compile(r"SafeMath|pragma\s+solidity\s*[^;]*0\.[89]|pragma\s+solidity\s*[^;]*\^0\.[89]")


def find_markers(code: str) -> List[dict]:
    """Locate known vulnerability markers, with 1-based line numbers."""
    hits = []
    for line_no, line in enumerate(code.splitlines(), start=1):
        for cls, name, pat in MARKERS:
            m = pat.search(line)
            if m:
                hits.append({"class": cls, "marker": name, "line": line_no,
                             "text": line.strip()[:120]})
    return hits


def marker_summary(code: str) -> str:
    """One-line reading of the markers, including the ambiguity case."""
    hits = find_markers(code)
    classes = sorted({h["class"] for h in hits})
    if not classes:
        return ("No distinctive marker found. Integer Overflow has no signature of its "
                "own in this dataset, so it is often what the model falls back to -- "
                "treat such predictions with extra caution."
                + ("" if SAFE_ARITHMETIC.search(code) else
                   " No SafeMath or Solidity >=0.8 pragma detected either."))
    if len(classes) == 1:
        return f"Markers point to one class: {classes[0]}."
    return ("Markers for more than one class are present (" + ", ".join(classes) +
            "). Contracts like this are genuinely ambiguous -- the training labels "
            "force a single class onto them, which is the main source of error here.")


def occlusion_attribution(
    code: str,
    predict: Callable[[List[str]], List[List[float]]],
    target_idx: int,
    max_lines: int = 40,
) -> List[Tuple[int, str, float]]:
    """Rank lines by how much removing one drops the target class probability.

    `predict` takes a list of code strings and returns a probability row for each.
    Line-level occlusion keeps this to one forward pass per non-empty line, which
    is cheap enough to run interactively.

    Returns (line_no, line_text, importance) sorted by importance, descending.
    A positive score means the line supported the prediction.
    """
    lines = code.splitlines()
    idxs = [i for i, l in enumerate(lines) if l.strip()][:max_lines]
    if not idxs:
        return []

    variants = []
    for i in idxs:
        variants.append("\n".join(lines[:i] + lines[i + 1:]))

    base = predict([code])[0][target_idx]
    probs = predict(variants)

    scored = [(i + 1, lines[i].strip(), base - p[target_idx]) for i, p in zip(idxs, probs)]
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored
