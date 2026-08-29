"""Adversarial / out-of-distribution probe suite.

The reported test accuracy comes from a split of the *same* corpus the model was
trained on, so it says little about behaviour on unfamiliar code. This suite
feeds hand-written contracts designed to break the model, grouped by what each
group is meant to expose:

  SAFE      benign modern contracts -- there is no "no vulnerability" class, so
            every prediction here is a false positive; what matters is how
            confident it is
  MODERN    the trained vulnerabilities written in post-0.8 syntax the corpus
            never contained (`.call{value: x}("")`)
  ABLATE    a real vulnerability with the trained keyword removed -- tests
            whether the model learned the bug or the token
  INJECT    a safe contract with a harmless keyword added -- the same test from
            the other direction
  PERTURB   a correctly-classified contract, renamed and commented -- prediction
            should not move
  JUNK      not Solidity at all

Run:  python src/robustness.py --model-dir models/codebert-vuln
"""
import argparse
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# --------------------------------------------------------------------------
# Probes. `expect` is what a correct system should say; None means "no class is
# right -- the model has no safe option and must be wrong".
# --------------------------------------------------------------------------
PROBES = [
    # ---------------- SAFE: modern, guarded, genuinely fine ----------------
    dict(id="safe_checked_math", group="SAFE", expect=None,
         note="Solidity 0.8 checked arithmetic, no external call, no timestamp",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Ledger {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function total(uint256 a, uint256 b) external pure returns (uint256) {
        return a + b;
    }
}"""),

    dict(id="safe_cei_guard", group="SAFE", expect=None,
         note="Correct checks-effects-interactions plus a reentrancy guard",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SafeVault {
    mapping(address => uint256) private balances;
    bool private locked;

    modifier nonReentrant() {
        require(!locked, "reentrant");
        locked = true;
        _;
        locked = false;
    }

    function withdraw(uint256 amount) external nonReentrant {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        require(ok, "transfer failed");
    }
}"""),

    dict(id="safe_pure_library", group="SAFE", expect=None,
         note="Pure math library, no state, no calls",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library Math {
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }

    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        return (a & b) + (a ^ b) / 2;
    }
}"""),

    dict(id="safe_storage", group="SAFE", expect=None,
         note="Trivial owner-gated storage",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Storage {
    address public immutable owner;
    uint256 private value;

    constructor() { owner = msg.sender; }

    function set(uint256 v) external {
        require(msg.sender == owner, "not owner");
        value = v;
    }

    function get() external view returns (uint256) { return value; }
}"""),

    # ---------------- MODERN: real bugs, new syntax ----------------
    dict(id="modern_reentrancy", group="MODERN", expect="Reentrancy",
         note="Genuine reentrancy using .call{value:} -- the corpus only ever saw .call.value()",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ModernBank {
    mapping(address => uint256) public balances;

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }
}"""),

    dict(id="modern_delegatecall", group="MODERN", expect="Dangerous Delegatecall",
         note="Attacker-controlled delegatecall target, modern syntax",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Proxy {
    function forward(address impl, bytes calldata data) external {
        (bool ok, ) = impl.delegatecall(data);
        require(ok, "delegatecall failed");
    }
}"""),

    dict(id="modern_overflow_unchecked", group="MODERN", expect="Integer Overflow",
         note="Overflow deliberately re-enabled with an unchecked block",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Unchecked {
    mapping(address => uint256) public balances;

    function unsafeSub(address to, uint256 amount) external {
        unchecked {
            balances[msg.sender] -= amount;
            balances[to] += amount;
        }
    }
}"""),

    # ---------------- ABLATE: real bug, trained keyword removed ----------------
    dict(id="ablate_reentrancy_transfer", group="ABLATE", expect="Reentrancy",
         note="Reentrancy through an external contract callback -- no .call.value anywhere",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IReceiver { function onTokensReceived(uint256 amount) external; }

contract TokenBank {
    mapping(address => uint256) public balances;

    function withdrawTokens(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        IReceiver(msg.sender).onTokensReceived(amount);
        balances[msg.sender] -= amount;
    }
}"""),

    dict(id="ablate_timestamp_blocknumber", group="ABLATE", expect="Timestamp Dependency",
         note="Time-based randomness via block.number/blockhash instead of block.timestamp",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Lottery {
    address public winner;

    function draw() external {
        if (uint256(blockhash(block.number - 1)) % 2 == 0) {
            winner = msg.sender;
        }
    }
}"""),

    # ---------------- INJECT: safe code, harmless keyword added ----------------
    dict(id="inject_timestamp_into_safe", group="INJECT", expect=None,
         note="safe_checked_math plus a timestamp used only in an event -- nothing depends on it",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Ledger {
    mapping(address => uint256) public balances;
    event Deposited(address indexed who, uint256 amount, uint256 at);

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value, block.timestamp);
    }

    function total(uint256 a, uint256 b) external pure returns (uint256) {
        return a + b;
    }
}"""),

    dict(id="inject_delegatecall_comment", group="INJECT", expect=None,
         note="Safe storage contract with the word delegatecall only in a comment",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Storage {
    address public immutable owner;
    uint256 private value;

    constructor() { owner = msg.sender; }

    // NOTE: we deliberately avoid .delegatecall( here; upgrades are out of scope.
    function set(uint256 v) external {
        require(msg.sender == owner, "not owner");
        value = v;
    }
}"""),

    dict(id="inject_callvalue_string", group="INJECT", expect=None,
         note="Safe contract where '.call.value(' appears only inside a revert string",
         code="""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Docs {
    uint256 public counter;

    function bump() external {
        require(counter < 100, "use .call.value( instead");
        counter += 1;
    }
}"""),

    # ---------------- PERTURB: same bug, cosmetic changes ----------------
    dict(id="perturb_baseline", group="PERTURB", expect="Reentrancy",
         note="Classic vulnerable withdraw, original naming",
         code="""contract PrivateBank {
    mapping (address => uint) public balances;

    function CashOut(uint _am) {
        if(_am <= balances[msg.sender]) {
            if(msg.sender.call.value(_am)()){
                balances[msg.sender] -= _am;
            }
        }
    }
}"""),

    dict(id="perturb_renamed", group="PERTURB", expect="Reentrancy",
         note="Identical bug, identifiers renamed and comments added",
         code="""// Treasury module -- handles member payouts.
contract MemberTreasury {
    mapping (address => uint) public credits;   // per-member credit

    // Pay out the requested amount to the caller.
    function redeem(uint requested) {
        if(requested <= credits[msg.sender]) {
            if(msg.sender.call.value(requested)()){
                credits[msg.sender] -= requested;   // settle afterwards
            }
        }
    }
}"""),

    # ---------------- JUNK: not Solidity ----------------
    dict(id="junk_python", group="JUNK", expect=None,
         note="Python, not Solidity",
         code="""def transfer(sender, receiver, amount):
    if balances[sender] >= amount:
        balances[sender] -= amount
        balances[receiver] += amount
    return True"""),

    dict(id="junk_prose", group="JUNK", expect=None,
         note="Plain English, no code",
         code="""The quarterly report shows a modest increase in customer retention,
driven largely by the new onboarding flow. We expect this trend to continue
into the next fiscal year, assuming headcount remains flat."""),
]


def load(model_dir: Path):
    classes = json.loads((model_dir / "label_classes.json").read_text())
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return classes, tok, model


def predict(code, classes, tok, model):
    accepted = set(inspect.signature(model.forward).parameters)
    inputs = tok(code, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k in accepted}
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    i = max(range(len(p)), key=p.__getitem__)
    return classes[i], p[i], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/codebert-vuln")
    ap.add_argument("--tag", default="codebert")
    args = ap.parse_args()

    model_dir = ROOT / args.model_dir
    classes, tok, model = load(model_dir)

    rows = []
    for pr in PROBES:
        label, conf, dist = predict(pr["code"], classes, tok, model)
        ok = (label == pr["expect"]) if pr["expect"] else None
        rows.append(dict(id=pr["id"], group=pr["group"], expect=pr["expect"],
                         predicted=label, confidence=conf, note=pr["note"],
                         correct=ok, dist=dict(zip(classes, dist))))

    print(f"\nRobustness probe -- {args.tag}\n")
    print(f"{'id':<32}{'group':<9}{'expected':<24}{'predicted':<24}{'conf':>7}  ")
    print("-" * 100)
    for r in rows:
        mark = {True: "PASS", False: "FAIL", None: " -- "}[r["correct"]]
        print(f"{r['id']:<32}{r['group']:<9}{str(r['expect'] or 'none correct'):<24}"
              f"{r['predicted']:<24}{r['confidence']:>6.1%}  {mark}")

    # Summaries that actually mean something
    print()
    gradeable = [r for r in rows if r["correct"] is not None]
    if gradeable:
        acc = sum(r["correct"] for r in gradeable) / len(gradeable)
        print(f"Accuracy on gradeable OOD probes: {acc:.0%} "
              f"({sum(r['correct'] for r in gradeable)}/{len(gradeable)})")

    safe = [r for r in rows if r["group"] in ("SAFE", "INJECT")]
    if safe:
        hi = [r for r in safe if r["confidence"] >= 0.90]
        print(f"Benign contracts flagged with >=90% confidence: {len(hi)}/{len(safe)}"
              "  (every one is a false positive -- there is no safe class)")

    inj = [r for r in rows if r["group"] == "INJECT"]
    for r in inj:
        print(f"  inject: {r['id']:<32} -> {r['predicted']} ({r['confidence']:.0%})")

    pert = [r for r in rows if r["group"] == "PERTURB"]
    if len(pert) == 2:
        same = pert[0]["predicted"] == pert[1]["predicted"]
        print(f"Rename/comment invariance: {'stable' if same else 'UNSTABLE'} "
              f"({pert[0]['predicted']} {pert[0]['confidence']:.0%} -> "
              f"{pert[1]['predicted']} {pert[1]['confidence']:.0%})")

    # Can confidence be used to screen benign code out? Only if the confidence
    # ranges do not overlap.
    benign = [r["confidence"] for r in rows if r["group"] in ("SAFE", "INJECT", "JUNK")]
    truly_vuln = [r["confidence"] for r in rows if r["group"] in ("MODERN", "ABLATE", "PERTURB")]
    if benign and truly_vuln:
        print(f"\nConfidence on benign/junk : min {min(benign):.1%}  max {max(benign):.1%}")
        print(f"Confidence on real vulns  : min {min(truly_vuln):.1%}  max {max(truly_vuln):.1%}")
        if max(benign) >= min(truly_vuln):
            print("=> ranges OVERLAP: no confidence threshold separates benign code from "
                  "genuine vulnerabilities. Confidence is not a usable safety signal.")
        else:
            print(f"=> separable at a threshold between {max(benign):.1%} and {min(truly_vuln):.1%}.")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"robustness_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
