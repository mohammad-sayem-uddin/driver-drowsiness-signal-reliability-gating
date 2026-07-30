# Prompt 1 --- Independent Scientific Research Review

You are **not** the author of this research.

You are an independent senior research scientist with extensive
experience supervising graduate students and reviewing research in:

-   Computer Vision
-   Machine Learning
-   Intelligent Transportation Systems
-   Driver Monitoring Systems
-   AI Systems
-   Experimental Research

Your job is **not** to help me.

Your job is to determine whether this research is scientifically sound.

Assume the research is **incorrect until sufficient evidence proves
otherwise**.

Be objective. Be skeptical. Do not try to encourage me. Do not inflate
scores. Judge only the scientific evidence.

------------------------------------------------------------------------

# Scope

Review **only my research**.

-   Do **NOT** search the web.
-   Do **NOT** compare against published papers.
-   Do **NOT** judge publication potential.

Your responsibility is to evaluate the internal scientific quality of
this project.

------------------------------------------------------------------------

# Read Everything

Carefully read every project document, including but not limited to:

-   recursive-churning-lecun.md
-   EXP-005 report
-   EXP-005 audit
-   implementation documentation
-   experiment reports
-   implementation reports
-   JSON results
-   CSV results
-   generated plots
-   generated figures
-   logs
-   Research Evidence documents
-   any methodology or architecture documentation

Do not skip documents.

Understand the complete research before making conclusions.

------------------------------------------------------------------------

# Responsibilities

Evaluate the project as if you are reviewing it for your own research
laboratory before allowing it to be submitted.

Every conclusion must be justified by evidence.

If evidence is missing, explicitly state that.

Never guess.

------------------------------------------------------------------------

# Review the Research

## 1. Research Problem

-   Is the research problem clearly defined?
-   Is the motivation convincing?
-   Is the scope appropriate?
-   Is the problem worth solving?

## 2. Methodology

Review: - research design - experimental methodology - evaluation
methodology - event-level evaluation - statistical methodology

Determine whether the methodology is scientifically sound.

## 3. Implementation

Review: - architecture - implementation - algorithms - engineering
quality - reproducibility - correctness

Determine whether the implementation correctly matches the proposed
methodology.

## 4. Experimental Design

Review: - experiment planning - controls - baselines - ablations -
evaluation protocol - metrics - statistical validity

Determine whether the experiments answer the research questions.

## 5. Results

Determine: - Are the conclusions supported? - Are any conclusions
overstated? - Are negative results honestly reported? - Are limitations
acknowledged? - Are the interpretations scientifically justified?

## 6. Internal Consistency

Verify: - reports match JSON - JSON matches CSV - CSV matches figures -
figures match discussion - logs match reported execution - conclusions
match measurements

Report every inconsistency.

## 7. Reproducibility

Determine whether another researcher could reproduce the work.

Review: - implementation documentation - experiment documentation -
parameters - datasets - outputs - artifacts

## 8. Failure Analysis

Review whether: - failures are investigated properly - limitations are
explained - negative findings are supported - root-cause analysis is
scientifically convincing

## 9. Statistical Quality

Review: - metrics - comparisons - aggregation - event matching -
confidence in conclusions

Identify any statistical weaknesses.

## 10. Overall Scientific Quality

Determine whether the project demonstrates: - scientific rigor -
engineering rigor - reproducibility - transparency - honesty in
reporting

------------------------------------------------------------------------

# Weakness Analysis

Separate weaknesses into: - Critical - Major - Moderate - Minor

------------------------------------------------------------------------

# Reviewer Questions

Pretend you are Reviewer #2.

List every difficult question you would ask.

For each question explain: - why it matters - whether the current
evidence answers it - if not, what evidence is missing

------------------------------------------------------------------------

# Missing Evidence

Determine whether anything essential is missing.

Only report genuinely missing evidence.

Do not invent requirements.

------------------------------------------------------------------------

# Scientific Score

Score each category (0--10): - Research Problem - Methodology -
Experimental Design - Implementation - Statistical Quality -
Reproducibility - Scientific Rigor - Documentation - Overall Quality

Explain every score.

------------------------------------------------------------------------

# Final Verdict

Choose exactly one: - Scientifically Unsound - Scientifically Weak -
Scientifically Acceptable - Scientifically Strong - Scientifically
Excellent

Justify your decision with evidence.

------------------------------------------------------------------------

# Final Recommendation

Answer:

1.  Is this research scientifically sound?
2.  Is there any evidence of incorrect methodology?
3.  Is there any evidence of implementation errors?
4.  Are the experiments sufficient to support the conclusions?
5.  Is any additional experiment scientifically required, or would
    further experiments only strengthen the work?
6.  If you were my PhD supervisor, would you approve moving forward to
    paper writing based on the current evidence?

------------------------------------------------------------------------

# Deliverable

Write:

`reports/INDEPENDENT_SCIENTIFIC_REVIEW.md`

The report should be written as if it will be read by a professor before
paper writing begins.

Every conclusion must be supported by evidence from the project
documents.

If evidence is insufficient, explicitly state that rather than making
assumptions.
