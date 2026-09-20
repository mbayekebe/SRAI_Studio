# Mathematical Thinking and Reproducible Computation

**SRAI Book 1 · Chapter 1 · Production Unit PU-B01-C01**

> A number is useful to a decision-maker only when we can explain where it came from, reproduce the work, and defend the assumptions behind it.

This repository accompanies Chapter 1 of *SRAI Book 1 — Mathematical Foundations*. It introduces the discipline that should come before model selection: defining the real problem, stating the abstraction, making assumptions visible, reproducing the computation, and deciding whether the result is fit for use.

The companion notebook uses a deliberately simple electricity-demand model. Its purpose is not to present a realistic forecasting system. It is to show, in a small and inspectable example, how mathematical reasoning, code verification, limitations, and decision consequences belong in the same analytical record.

## Why this production unit matters

In official statistics, agricultural censuses, information systems, and decision-support work, I have repeatedly seen technically impressive results become unsafe because the data changed, an assumption remained undocumented, units were inconsistent, or the model was used outside the setting for which it was designed.

Reproducible code does not eliminate these risks, but it makes the analytical claim inspectable. It allows another analyst to reconstruct what was done and gives managers a sounder basis for questioning the result.

## Learning objectives

After working through the chapter and notebook, the reader should be able to:

1. distinguish a mathematical model from the reality it represents;
2. state variables, assumptions, objectives, constraints, and omissions;
3. build a computational experiment that can be reproduced;
4. verify mathematical expectations with numerical tests;
5. separate verification from validation; and
6. connect analytical evidence to a decision for which a named person remains accountable.

## The SRAI theory-to-decision chain

```text
Reality → Abstraction → Model → Algorithm → Python
        → Verification → Interpretation → Decision
```

Each step answers a different question. Skipping a step may produce a number faster, but it weakens the evidence supporting the eventual decision.

| Stage | Control question |
|---|---|
| Reality | What real system, population, or decision are we trying to understand? |
| Abstraction | What matters, and what has been deliberately left out? |
| Model | Which variables, relationships, assumptions, objectives, and constraints represent the problem? |
| Algorithm | What finite procedure converts the model into a result? |
| Python | Can the procedure be executed consistently in a controlled environment? |
| Verification | Did we implement and solve the intended model correctly? |
| Interpretation | What does the output mean—and what does it not mean? |
| Decision | How should evidence, uncertainty, costs, context, and human responsibility be combined? |

## Companion notebook

The canonical notebook is:

[`notebooks/M1_N01_mathematical_thinking.ipynb`](notebooks/M1_N01_mathematical_thinking.ipynb)

It contains:

- recorded environment information and random seed `42`;
- an immutable `LinearDemandModel` data class;
- predictions for three temperature values;
- assertions against known expected results;
- a numerical sensitivity check;
- an explicit list of omitted factors; and
- exercises from basic identification through a decision-alert capstone.

### Worked model

The notebook defines:

$$
D(T)=\beta_0+\beta_1T
$$

with baseline demand $\beta_0=120$ and temperature sensitivity $\beta_1=2.5$. For temperatures of 20°C, 25°C, and 30°C, the verified predictions are:

| Temperature | Predicted demand |
|---:|---:|
| 20°C | 170.0 |
| 25°C | 182.5 |
| 30°C | 195.0 |

The sensitivity test confirms that a one-degree increase changes modeled demand by `2.5` units.

## Run the notebook locally

### Requirements

- Python 3.11 or later
- Jupyter or VS Code with the Jupyter extension
- NumPy
- the accompanying `srai_math` package

### Windows setup

From the repository root:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt jupyterlab
.venv\Scripts\python.exe -m jupyter lab
```

Open `notebooks/M1_N01_mathematical_thinking.ipynb`, select the `.venv` kernel, and choose **Run All**.

### macOS or Linux setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt jupyterlab
.venv/bin/python -m jupyter lab
```

### Successful execution

A successful run should:

1. print the recorded environment information;
2. return `[170.0, 182.5, 195.0]` for the three demand predictions;
3. print `Verification passed.`;
4. return `[2.5]` for the sensitivity test; and
5. complete without an assertion error.

If an assertion fails, treat that as evidence to investigate—not as an inconvenience to suppress.

## Verification is not validation

The notebook verifies that the code implements the stated linear model correctly. It does **not** establish that the model is adequate for real electricity planning.

The example omits humidity, seasonality, weekdays, economic activity, prices, outages, and nonlinear thresholds. Before operational use, those omissions would need to be examined against real data, institutional knowledge, the intended population, and the consequences of false alarms or missed events.

That distinction is central to SRAI:

- **Verification:** Did we solve the stated model correctly?
- **Validation:** Is the stated model an adequate representation of the relevant reality for the intended use?

A computation may be reproducible and verified while still being unsuitable for a policy or operational decision.

## Exercises

- **Level A:** Identify the model variable, parameters, assumption, and output.
- **Level B:** Extend the model to include humidity: $D(T,H)=\beta_0+\beta_1T+\beta_2H$.
- **Level C:** Generate noisy synthetic observations and fit a linear regression.
- **Capstone:** Trigger a capacity alert above a selected threshold, then examine false alarms and missed alerts.

For every extension, record the seed, dependencies, assumptions, expected behavior, tests, and limitations.

## Decision checklist

Before relying on an analytical result, ask:

- Can another analyst reproduce it from controlled inputs?
- Has mathematical and computational correctness been verified?
- Has adequacy for the intended reality and population been validated?
- Are assumptions, exclusions, uncertainty, and limitations visible?
- Do the objective and constraints reflect the institution's real purpose?
- Has the impact of error on different groups been examined?
- Is a named human responsible for the final decision?

## Suggested repository structure

```text
PU-B01-C01/
├── README.md
├── requirements.txt
├── notebooks/
│   └── M1_N01_mathematical_thinking.ipynb
├── srai_math/
│   └── utils/
└── docs/
    └── PU-B01-C01_Executive_Brief_v1.1.pdf
```

## Source and release status

| Item | Value |
|---|---|
| Production unit | `PU-B01-C01` |
| Book | SRAI Book 1 — Mathematical Foundations |
| Chapter | Chapter 1 — Mathematical Thinking and Reproducible Computation |
| Canonical notebook | `M1_N01_mathematical_thinking.ipynb` |
| Landing-page version | `1.0` |
| Status | Draft for owner review |

This landing page was developed from the controlled Chapter 1 content, the validated canonical notebook M1_N01, and Executive Brief v1.1. Publication should follow substantive review and approval by Mbaye Kebe.

## Author

**Mbaye Kebe** is a senior statistician and information-systems specialist whose work spans official statistics, agricultural censuses, large surveys, analytical systems, data quality, and decision support in Africa and internationally.

SRAI brings together statistics, reasoning, and artificial intelligence with an emphasis on reproducibility, institutional usefulness, and accountable human judgment.

## Citation

```text
Kebe, Mbaye. "Mathematical Thinking and Reproducible Computation."
SRAI Book 1: Mathematical Foundations, Production Unit PU-B01-C01.
GitHub landing page, version 1.0.
```

## Rights and reuse

Copyright © Mbaye Kebe. All rights reserved pending the adoption of a formal repository licence. Do not assign an open-source licence until the intended terms for code, notebooks, educational text, and commercial reuse have been approved.
