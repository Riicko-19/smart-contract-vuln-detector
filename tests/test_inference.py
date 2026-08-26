"""Smoke test: the trained model loads and returns a valid class for each vuln type."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import classify, MODEL_DIR  # noqa: E402

SNIPPETS = {
    "reentrancy": "function withdraw(uint amount) public { msg.sender.call.value(amount)(\"\"); balances[msg.sender] -= amount; }",
    "delegatecall": "function forward(address target, bytes data) public { target.delegatecall(data); }",
    "timestamp": "function play() public { if (block.timestamp % 2 == 0) { winner = msg.sender; } }",
    "overflow": "function add(uint a, uint b) public pure returns (uint) { return a + b; }",
}


def test_classify_returns_known_label():
    classes = json.loads((MODEL_DIR / "label_classes.json").read_text())
    for name, code in SNIPPETS.items():
        label, confidence = classify(code)
        assert label in classes, f"{name}: unexpected label {label}"
        assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    test_classify_returns_known_label()
    print("OK")
