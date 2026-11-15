# futureleak

A look-ahead leakage auditor for time-series feature pipelines and trading indicators.

A feature computed at time `t` must not change when data after `t` changes. futureleak
tests that property directly. It calls a function on truncated and perturbed copies of its
input and compares the outputs at or before `t`. It does not read the function's source, so
it works on any callable that maps a time-ordered `DataFrame` or `Series` to features on the
same index.

After building it I ran it over every indicator in six open-source technical-analysis
libraries. The results are in [Findings](#findings). The short version: 668 of the 684 indicators I could audit showed no look-ahead with their
default parameters, 10 are flagged, and I traced each of the 10 to a line of source.

## The problem

A backtest usually computes features over the whole history in one vectorised pass. Any
operation that reads rows after `t` then puts future information into the feature at `t`,
and nothing fails: the code runs, the numbers look plausible, and the strategy looks better
than it is. The common causes are short and easy to miss in review: `shift(-1)`, a centred
rolling window, a z-score with the full-sample mean, a backward fill, a label left among the
features.

This is one case of data leakage, which is a documented problem well beyond finance. Kapoor
and Narayanan surveyed the literature in fields that use machine learning and found
"17 fields where leakage has been found, collectively affecting 294 papers" [1]. The
freqtrade documentation says of trading strategies that look-ahead bias "is sometimes very
easy to introduce" but "can be very hard to detect" [5].

Leakage is a different problem from backtest overfitting, where the best of many tried
strategies is reported [2]. The usual corrections for overfitting, the Deflated Sharpe Ratio
and the Probability of Backtest Overfitting, adjust a result for the amount of searching
behind it. They do not look at how a feature was computed, so a leak has to be found in the
feature code itself.

## What the tool does

```python
import pandas as pd
from futureleak import audit, synthetic_ohlcv

def features(df):
    close = df["close"]
    return pd.DataFrame({
        "momentum_10": close.pct_change(10),
        "zscore": (close - close.mean()) / close.std(),   # full-sample statistics
        "smooth": close.rolling(5, center=True).mean(),   # centred window
        "next_return": close.pct_change().shift(-1),      # the label, left in by mistake
    })

print(audit(features, synthetic_ohlcv(600, seed=0)))
```

Output (`python examples/quickstart.py`):

```
features: leak (600 rows, 20 cuts)
  momentum_10: causal
  zscore: leak [full_sample] leaking cuts 100%, horizon >=431 (to end of data), earliest offset -597
  smooth: leak [fixed_horizon] leaking cuts 100%, horizon 2, earliest offset -1
  next_return: leak [fixed_horizon] leaking cuts 100%, horizon 1, earliest offset 0
```

For each of `n_cuts` cut positions `t` (20 by default, spread over the last three quarters
of the data) the audit makes three comparisons on rows up to and including `t`:

1. Truncation: `f(data[:t+1])` against `f(data)[:t+1]`.
2. Perturbation: `f(data with every row after t replaced by noise)[:t+1]` against
   `f(data)[:t+1]`.
3. Horizon search, when the perturbation changes something: perturb only the rows after
   `t + h` and search for the smallest `h` at which the outputs up to `t` stop changing.
   That `h` is the look-ahead horizon in rows.

Any difference beyond a float tolerance is a proof by counterexample that the output depends
on later rows. Missing values compare equal to each other. The report gives, for each output
column, the verdict, the share of cut points that leak, the horizon, the earliest affected
row relative to the cut, and the kind of leak:

| kind | meaning | examples | repair |
|---|---|---|---|
| `fixed_horizon` | reads the same bounded number of rows ahead at every cut | `shift(-k)`, centred window | lag the output by the horizon |
| `full_sample` | the horizon runs to the end of the data at every leaking cut | global z-score, min-max scaling, full-sample rank | recompute on data up to `t` (expanding or rolling) |
| `variable_horizon` | the horizon depends on the data | `bfill`, interpolation, weekly aggregates broadcast to days | recompute on data up to `t` |
| `truncation_only` | changes when later rows are removed but not when their values change | `arange(n) / n`, anything that depends on `len(data)` | recompute on data up to `t` |

The last three are truncation-sensitive: the value at `t` depends on how much data exists
after `t`, so no lag repairs them.

Cases it handles:

- Outputs that are shorter than the input (dropped warm-up rows, `dropna`) are matched by
  index label.
- Warm-up `NaN`s are not differences. A column that is entirely missing is reported as
  `inconclusive`, not as causal.
- Non-deterministic functions are detected by calling the function twice on the same input
  and are reported as `non_deterministic`, per column, not as leaks.
- Functions that raise on short inputs are still audited by perturbation, which keeps the
  input at full length. The report counts the cut points at which a call raised.
- `Series`, `DataFrame`, array, tuple and mapping outputs; numeric, boolean, string and
  datetime columns; date and integer indexes.

### In a test suite

```python
from futureleak import assert_causal

def test_features_do_not_look_ahead(prices):
    assert_causal(build_features, prices)
```

`assert_causal` raises `LeakageError` (an `AssertionError`) with the report attached. It also
fails when the function cannot be verified: when it is non-deterministic, or when a column
is entirely missing on the test data.

### On the command line

```
futureleak audit mypackage.features:build_features --data prices.csv --index-col date
futureleak audit path/to/features.py:build_features          # seeded synthetic OHLCV bars
futureleak scan-lib ta                                       # every indicator of a library
```

Both commands take `--format text|json|markdown` and `--output FILE`. The exit status is 0
when nothing leaks, 1 when a leak is found, 2 on an error and 3 when the function could not
be verified. `python -m futureleak ...` does the same as the `futureleak` command, which
helps on Windows machines whose policy blocks unsigned launcher executables (mine does,
intermittently).

### Install

```
pip install -e .                 # core: numpy and pandas only
pip install -e ".[indicators]"   # plus the libraries audited below
```

Python 3.10 or later.

## Findings

Every number in this section is copied from the files in `results/`, which the two scripts
in `experiments/` write. My run was on Python 3.14.8, Windows, pandas 3.1.0 and numpy 2.5.4.

### 1. Do the indicator libraries look ahead?

`experiments/audit_indicator_libraries.py` finds the indicators of each library by
introspection, calls each one with its default parameters on 600 seeded synthetic daily
bars, and audits it at 40 cut points. Full output: `results/indicator_audit.md` and
`results/indicator_audit.json`.

| library | version | found | audited | causal | leak | inconclusive | errored | skipped |
|---|---|---|---|---|---|---|---|---|
| ta | 0.11.0 | 43 | 42 | 41 | 1 | 0 | 0 | 1 |
| finta | 1.3 | 90 | 78 | 73 | 5 | 0 | 11 | 1 |
| stockstats | 0.6.13 | 54 | 54 | 54 | 0 | 0 | 0 | 0 |
| pandas_ta_classic | 0.9.0 | 224 | 211 | 206 | 3 | 2 | 2 | 11 |
| talib (TA-Lib) | 0.8.3 | 237 | 236 | 234 | 0 | 2 | 0 | 1 |
| btalib (bta-lib) | 1.0.0 | 109 | 63 | 60 | 1 | 2 | 37 | 9 |
| total | | 757 | 684 | 668 | 10 | 6 | 50 | 23 |

No indicator was non-deterministic. The 10 flagged indicators, with the cause I confirmed
by reading the installed source:

| library | indicator | leaking output | measured | classification | cause (source) |
|---|---|---|---|---|---|
| ta | `trend.KSTIndicator` | kst, kst_sig, kst_diff | full sample; only rows up to 51 affected | undocumented leak | rates of change use `close.shift(r, fill_value=close.mean())`, the mean of the whole series (`ta/trend.py:500-535`) |
| finta | `STOCHRSI` | its one output | full sample, 60% of cuts | undocumented leak | RSI is scaled by `rsi.min()` and `rsi.max()` of the whole series, not of the trailing window (`finta/finta.py:1237`) |
| finta | `SAR` | its one output | full sample; only rows up to 88 affected | undocumented leak | the recursion is seeded with `(high - low).std()` over the whole series (`finta/finta.py:829`) |
| btalib | `ht_trendmode` | trendline | horizon 1..6 | undocumented leak | index handling: a positional slice ends 6 rows after the label-based reads around it (`btalib/indicators/ht_trendmode.py:165`) |
| finta | `ICHIMOKU` | CHIKOU | horizon 26 | leaks by design, not documented | `close.shift(-26)`, the standard Chikou span (`finta/finta.py:1897-1900`) |
| finta | `WILLIAMS_FRACTAL` | BearishFractal, BullishFractal | horizon 1..2 | leaks by design, documented | `rolling(5, center=True)` (`finta/finta.py:2227, 2233`); the docstring says the extremum is "preceded and followed" |
| pandas_ta_classic | `overlap.ichimoku` | ICS_26 | horizon 26 | leaks by design, documented | `close.shift(-kijun)` (`overlap/ichimoku.py:57`); docstring: "Not causal" |
| pandas_ta_classic | `trend.dpo` | DPO_20 | horizon 11 | leaks by design, documented | `centered=True` by default (`trend/dpo.py:39`); docstring: "Not causal" |
| pandas_ta_classic | `statistics.tos_stdevall` | all 7 columns | full sample | leaks by design, documented | one regression over the whole series (`statistics/tos_stdevall.py:72-74`); docstring: "Not causal" |
| finta | `QSTICK` | its one output | truncation only | output depends on input length, no future values read | returns `tail(period)` rows instead of a rolling average (`finta/finta.py:1775-1780`) |

What I take from this:

- Almost everything is clean. 668 of 684 audited indicators showed no dependence on later
  rows. TA-Lib (236 audited) and stockstats (54 audited) had no flags at all. If you expected
  popular indicator libraries to be full of look-ahead, this run says otherwise.
- Four of the ten flags are deliberate and documented, and one more (the Chikou span in
  finta) follows the standard definition of the indicator without saying so. These columns
  are meant for charts. They still leak if they are fed to a model, which is easy to do with
  a call that returns every column of an indicator at once.
- Four flags are undocumented leaks, and they differ a lot in reach. finta's `STOCHRSI`
  touches every row of the series. So does bta-lib's `ht_trendmode`, but only on an integer
  index: on a date index it raises under pandas 3.1. The `ta` KST leak is confined to rows
  14-51 of the series with default windows. The finta `SAR` leak reaches no further than
  row 88 in this run and also needs an integer index.
- The `STOCHRSI` leak is real in the code but its effect depends on the data. It only
  changes the output when the whole-series minimum or maximum of RSI falls after the cut.
  I re-checked it on a second, independent price series (500 business days, a different
  generator and seed, cuts every 40 rows): the scaling line is the same, but 0 of 11 cuts
  changed any value, because the RSI extremes on that series occur in the first rows. The
  `ta` KST leak reproduced on the same series (30 rows changed). So "60% of cuts" describes
  this audit's data, not how often a user will be affected.
- `QSTICK` in finta is a defect of a different kind. It returns values for the last 14
  rows of whatever it is given, so the audit flags it, but no value uses a future price.
- The `ta` fill is a known pattern. The same `fill_value=close.mean()` was reported for the
  daily-return indicator in `ta` issue 323 and removed there in pull request 324. The KST
  lines are unchanged on the master branch as of 2026-11-15, and 0.11.0 is the latest
  release on PyPI.
- pandas_ta_classic gave me an independent check on the detector. Its source marks
  `ichimoku`, `dpo` and `tos_stdevall` as "Not causal" with defaults. The audit flagged
  exactly those three among the 211 it audited, with horizons of 26, 11 and the full
  sample, before I had read those docstrings.

One non-default setting, reported separately: every `ta` indicator accepts `fillna=True`.
With it, the warm-up rows are returned instead of `NaN`, and 5 of 42 audited `ta`
indicators are flagged (KAMA, DPO, KST, TRIX and Vortex). The KAMA case comes from
`np.roll`, which wraps the last rows of the series around to the first, and was reported
upstream in 2020 (`ta` issue 181, still open). Details are in `results/indicator_audit.md`.

What the run does not cover:

- 50 indicators raised with default parameters and were not audited: 37 in bta-lib, 11 in
  finta and 2 in pandas_ta_classic. Most are incompatibilities between the two older
  libraries and pandas 3.1 (`Series.append`, `Series.iteritems`, read-only arrays). 5 of
  the finta ones are stubs that raise `NotImplementedError`, and the 2 in pandas_ta_classic
  need a benchmark series. 23 were skipped because they need an input other than
  OHLCV, have a required parameter with no default, or do not return one row per input row.
- 6 indicators returned only missing values on this data (`acos` and `asin` of prices in
  three libraries), and 25 of the 668 counted as causal returned a constant, mostly
  candlestick patterns that never occur in the synthetic bars. The audit says little about
  those.
- pandas-ta itself could not be installed: it depends on numba, whose build refuses Python
  3.14. I audited the pandas-ta-classic fork instead. tti 0.2.2 installs, but all 57 of its
  indicator classes raise a `TypeError` on pandas 3.1.
- Default parameters and synthetic data only. The script repeats the audit on a second
  synthetic data set (another seed, 20 cut points) and all 757 outcomes were the same, but
  a leak that needs a non-default option, or a data pattern that random-walk bars lack,
  will not show up.

### 2. What a leak is worth in a backtest

`experiments/leak_inflates_backtest.py` trades a mean-reversion rule on driftless random
walks, where no strategy has an edge: short when the z-score of the price is positive, long
when it is negative, with a one-day lag. The only difference between the two versions is
the z-score. The leaky one uses the mean and standard deviation of the whole path; the
causal one uses expanding statistics. Mean annualised Sharpe ratio over 1,000 paths per
backtest length (`results/leak_inflates_backtest.md`):

| backtest length (days) | leaky: mean Sharpe | s.e. | paths with Sharpe > 0 | causal: mean Sharpe | s.e. | paths with Sharpe > 0 |
|---|---|---|---|---|---|---|
| 63 | 1.70 | 0.06 | 80% | 0.02 | 0.08 | 51% |
| 126 | 1.40 | 0.04 | 87% | 0.04 | 0.05 | 51% |
| 252 | 1.00 | 0.03 | 89% | 0.00 | 0.03 | 51% |
| 504 | 0.72 | 0.02 | 91% | -0.01 | 0.02 | 51% |
| 1260 | 0.49 | 0.01 | 92% | 0.01 | 0.01 | 52% |
| 2520 | 0.35 | 0.01 | 92% | 0.02 | 0.01 | 52% |

On a one-year backtest the leak alone is worth a Sharpe ratio of 1.00, against 0.00 for the
causal version. The size depends on the length of the backtest. The leak earns a roughly
fixed profit per path, about the distance the price wanders from its mean, so its effect on
the Sharpe ratio falls as the sample grows: 1.70 on a quarter, 0.35 on ten years. The
causal version has a mean Sharpe ratio between -0.01 and 0.04 at every length and is
profitable on about half of the paths, as it should be. Both functions were also audited:
`leaky_signal` is flagged and `causal_signal` passes.

## How to reproduce

```
git clone <this repository>
cd futureleak
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux: source .venv/bin/activate
pip install -e ".[dev,indicators]"

python -m pytest                                    # test suite
python experiments/audit_indicator_libraries.py     # 5 to 7 minutes on my laptop
python experiments/leak_inflates_backtest.py        # about 10 seconds
```

Both scripts are seeded. I ran each of them more than once and the results were identical. The libraries are optional: a library that is not installed, or that fails to
import, is listed under "Libraries not audited" in the output. The counts above depend on
the library versions and on the pandas version, because many of the errors are pandas
incompatibilities. The causes in `experiments/confirmed_causes.json` are tied to a library
version; if a flagged indicator has no entry for the installed version, the script reports
it as unconfirmed.

## Limitations

- A clean report is not a proof. The audit observes the function on one data set, at a
  finite number of cut points, under one perturbation. A leak that only fires on a data
  pattern that is absent, or with a parameter that was not used, is missed. Fonseca shows
  that look-ahead-freedom is undecidable in general and writes that empirical detectors
  "certify nothing by their silence" [6]. A reported leak, on the other hand, is
  a concrete counterexample and can be reproduced from the seed.
- The measured horizon is a lower bound. It is exact for outputs that respond to any
  change in their inputs. For discrete outputs it can be short: finta's fractal flags read
  2 rows ahead and were measured as 1 at some cut points.
- Contaminated inputs are out of reach. If the data already contains future information
  (restated fundamentals, prices adjusted with later corporate actions, a universe chosen
  with hindsight), a function can be causal in its input and still leak.
- Label leakage outside the function is not covered: target construction, overlapping
  labels across a train/test split, scalers fitted before the split, features selected on
  the test period.
- One row per time step. Cuts are row positions, so a panel with several rows per
  timestamp has to be audited one entity at a time or pivoted wide first.
- Outputs have to keep the input's index labels, or have the same length. A shorter output
  with a fresh integer index cannot be matched to input rows reliably.
- Hidden state is not handled. A function that caches across calls or reads the clock can
  pass or fail for the wrong reason. Only non-determinism between two identical calls is
  detected.
- The cost is two calls per cut point, plus roughly `2 * log2(horizon)` calls per leaking
  cut for the horizon search. That is fine for indicator-sized functions and slow for a
  pipeline that takes minutes per call.
- The perturbed rows are not realistic prices (highs can fall below lows). That cannot
  create a false leak, since only rows after the cut are perturbed, but a function that
  validates its input may raise, which the report counts.

## Related work

What already exists, as far as I could find in November 2026:

- freqtrade ships `lookahead-analysis`, which reruns a strategy on sliced data and compares
  indicator values and signals with the full backtest [5]. It is the same truncation idea,
  tied to freqtrade strategies.
- [peekahead](https://github.com/arielb57/peekahead) (September 2026) is a black-box
  detector for feature functions that also removes and perturbs future rows, names the
  input column that leaks, measures the horizon, and prices the leak in Sharpe terms. It is
  the closest thing to this project that I found, and it predates it. Its evaluation is a
  corpus of 24 hand-written leaky features, not third-party libraries.
- Static analysis: [backtestlint](https://github.com/TPOALOS/backtestlint) flags patterns
  such as `shift(-1)` and `center=True` in backtest code with AST rules. In the research
  literature, Yang et al. detect train/test leakage in notebooks statically [3] and
  Drobnjaković et al. do so by abstract interpretation [4]. Fonseca formalises
  look-ahead-freedom as temporal non-interference and gives a type system for a decidable
  fragment [6].
- Deflated Sharpe Ratio and Probability of Backtest Overfitting implementations are easy to
  find on GitHub. They address selection bias [2], not leakage.

What is not new here: the method. Comparing outputs under truncation and perturbation of
future rows is what freqtrade and peekahead already do, and it is an empirical detector in
the sense of [6], with the limits that paper points out.

What I believe is new:

- The audit itself. I found no published look-ahead audit of open-source indicator
  libraries. This one covers 757 indicators in six libraries, with every flag traced to a
  source line and the documented cases separated from the undocumented ones. Individual
  cases were known before: the `ta` issue tracker has reports from 2020 and 2023, and
  pandas-ta-classic documents its own non-causal indicators.
- The tool also differs from the ones above in smaller ways, which I do not claim as
  firsts: it separates leaks that a lag repairs from leaks that need recomputation, says
  when a leak is confined to the start of a series, reports non-determinism per column, and
  has `scan-lib` for auditing a whole library.

## Who would use or pay for this

I have no users and no revenue. This is who I think has the problem:

- Quant funds and proprietary trading firms that review research before allocating
  capital. A leak check that needs no access to the researcher's reasoning and runs in CI
  is cheap compared with trading a strategy that only worked in the backtest.
- Strategy marketplaces and signal vendors that accept submissions from third parties and
  cannot read every submission's code.
- Machine-learning forecasting teams outside finance (demand, energy, operations) whose
  feature pipelines have the same failure mode.

The library is MIT-licensed and will stay that way. If there is a business here, I expect
it to be in what a team needs around the check: audits of private feature stores at scale,
continuous monitoring of pipelines as dependencies change, and reports that a risk or
compliance function can sign off. I have not validated that with anyone yet.

## References

1. S. Kapoor and A. Narayanan. Leakage and the reproducibility crisis in
   machine-learning-based science. Patterns 4(9):100804, 2023.
   doi:10.1016/j.patter.2023.100804. Preprint: arXiv:2207.07048.
2. D. H. Bailey, J. M. Borwein, M. López de Prado and Q. J. Zhu. Pseudo-mathematics and
   financial charlatanism: the effects of backtest overfitting on out-of-sample
   performance. Notices of the American Mathematical Society 61(5):458, 2014.
   doi:10.1090/noti1105.
3. C. Yang, R. A. Brower-Sinning, G. A. Lewis and C. Kästner. Data leakage in notebooks:
   static detection and better processes. arXiv:2209.03345, 2022.
4. F. Drobnjaković, P. Subotić and C. Urban. Abstract interpretation-based data leakage
   static analysis. arXiv:2211.16073, 2022.
5. freqtrade documentation, "Lookahead analysis".
   https://github.com/freqtrade/freqtrade/blob/develop/docs/lookahead-analysis.md
6. X. Fonseca. Look-ahead-freedom as temporal non-interference: a verifiable correctness
   property for backtesting and agentic trading pipelines. arXiv:2607.04958, 2026.

## Licence

MIT. See `LICENSE`.
