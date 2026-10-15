# Recommendation page: research, design, and implementation plan

## Brief and audience

A beginner should be able to answer: which stock does this model favor, what
observations explain that preference, and does its historical performance justify
acting on it? A past loss must never look like a forecast or quality score.

The output is a **recommendation for further research**, not an unsupported buy
instruction. This project has no executable trading backtest, calibrated stock
probabilities, or established statistical reliability. Design must not hide that.

## Research findings

- [NN/g: Choosing chart types—consider context](https://www.nngroup.com/articles/choosing-chart-types/)
  recommends selecting a visualization around the reader's question and providing
  useful comparisons. Applied here: explain the stock's observation against its
  own normal activity or the same-date stock universe, not as an isolated number.
- [Government Analysis Function: Publishing charts](https://analysisfunction.civilservice.gov.uk/support/communicating-analysis/introduction-to-data-visualisation-e-learning/module-10-publishing-charts/)
  separates the takeaway from the statistical description and calls for text
  equivalents. Applied here: put the recommendation and reason in prose before
  optional score charts; keep dates, horizons, and units explicit.
- [W3C: Use of color](https://www.w3.org/WAI/WCAG21/Understanding/use-of-color.html)
  requires meaning to be communicated beyond color. Applied here: label favorable
  and unfavorable score effects in words; do not equate green with a buy signal.
- [W3C: Contrast minimum](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
  specifies normal-text contrast of at least 4.5:1. Check intended foreground and
  background pairs, including white button text on the accent color.

These are communication and accessibility sources, not evidence that the stock
selection model works. Its claims must come from the experiment's own results.

## Visual direction

An approachable research brief, not a trading terminal or KPI wall.

Palette: paper `#F8FAF9`, pale sage panel `#EDF3F0`, dark green ink `#18312E`,
deep teal interaction accent `#17645B`, muted secondary text `#53645F`,
and dark amber caution `#86561B`. Neutral/caution text accompanies every status.

Typography: local Segoe UI with a sans-serif fallback, 16px body, semibold
headings, and a restrained 30/24/20px hierarchy. Avoid external font downloads
in the corporate-network environment. Left-aligned short paragraphs; no tracked
all-caps labels, giant percentage metrics, or decorative gradients.

### Layout and revision against the brief

The previous three equal cards made historical returns look like the primary
decision variable. Repeating that layout with new colors would preserve the
problem. Replace it with one recommendation brief and a distinct evidence panel;
show alternatives below, with lower visual emphasis.

```text
Date, horizon, model and snapshot controls

Stock to investigate first           Is this enough to buy?
Company / ticker, first of N         Honest, plain-language verdict
Expected relative ordering          Model-level historical evidence
Why this model favors it            What has not been established

Other candidates: ranked names and short reasons

Inspect any stock
What helped / what counted against it, in ordinary language
Recent history — NOT a forecast [expand]
Exact score calculations and chart [expand]

Complete ranking / later outcomes / snapshot history [expand]
```

Native columns stack on narrow displays. No essential explanation depends on
hover, color, or opening a chart. A tie produces joint leaders; all-tied scores
produce no recommendation. Unknown stocks and future models retain generic labels.

## Execution plan

1. Add tested, deterministic text helpers for observed inputs, score effects,
   and historical evidence. Explain negative returns as past falls, not goals.
2. Replace the pick cards with the brief above. Promote model rank and a plain
   reason; demote historical percentages. Keep research-vs-buy scope prominent.
3. Use real model-score effects, including unfavorable evidence. Label linear
   contributions separately from hypothetical single-input sensitivity.
4. Apply the native theme tokens and align chart labels/colors with their meaning.
5. Preserve existing day/week/month, horizon, model discovery, replay, downloads,
   and winner-only test behavior. Do not retrain when switching views.
6. Test copy branches, ties, weak/undefined evidence, unknown future models, and
   all UI interactions. Run the full suite and inspect desktop/mobile rendering
   where browser tooling permits. Record limitations rather than claiming checks
   that did not complete.

## Verification and critique

- Full regression suite: 69 tests passed, including plain-language negative-return
  handling, volume references, unfavorable/undefined evidence, tied leaders,
  absence of headline percentage metrics, and existing page/model/date behavior.
- Theme tests verify intended foreground/background pairs at 4.5:1 or better.
  This is not a claim of full WCAG conformance for Streamlit or its third-party widgets.
- The running app rendered the real 31 December 2024 snapshot without page errors.
  Browser text confirmed the NEE recommendation, ordinary-language reasons,
  company names, and the model's actual validation evidence beside the suggestion.
- The first rendered-layout review found controls crowding out the result. The
  revised layout collapses date/model controls while keeping them clearly named.
  Repeated caveats were trimmed; the research-vs-buy boundary remains prominent.
- Pixel screenshot automation hung and was stopped. No completed desktop/mobile
  screenshot review is claimed. Responsive behavior uses native Streamlit columns
  and controls rather than custom fixed-width CSS. Desktop body text, heading
  positions, and the applied new background color were inspected in the browser.
- No training logic, target construction, model-selection rule, or dependencies
  changed for this redesign. Existing source-data and historical-test warnings remain.
