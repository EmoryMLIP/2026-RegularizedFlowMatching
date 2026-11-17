# Reproducing the paper

This repository reproduces the tables, figures and quantitative results of "Explicit
Regularization for Flow Matching" (L. Ruthotto, E. Treister). The paper `\input`s the generated
files `tables/*.tex` and `figures/*.tex`, copied from `experiments/paper/`.

## Two stages

| stage | code | input | output | cost |
|---|---|---|---|---|
| 1. compute | `experiments/compute/<script>.py` | `smoothfm` (`src/`), `config.py` | `<results>/<name>.{npz,log,manifest.json}` | 6802 core-h for the 126 runs (below) |
| 2. report | `experiments/report/report.py` | one results directory | `OUT/tables/*.tex`, `OUT/figures/*.tex` and their data files | seconds |

A compute run is identified by its result name (`oned`, `twod_circle_h1`, ...). The Makefile
has one target `compute-<name>` per run (`make list` prints the 126 names in dependency order),
and `experiments/runs.txt` lists the same runs with the results each one reads. Each log starts
with the commit, the tree state, package versions, CPU model and every config constant, then a
worked example of the first job; the manifest records the cost and the sha256 of every output.

The report stage is deterministic: two runs on the same results give byte-identical files.

## Rebuild the paper's tables and figures (seconds)

    uv sync --frozen --extra cholmod
    make fetch-results RESULTS_URL=<URL or path of the archive>   # default RESULTS_ROOT=results
    make release-report OUT=<dir>                                # default OUT=experiments/paper
    make sync PAPER=<paper repository>                           # copies OUT/tables, OUT/figures

The results are not in the repository. They are published as one archive,
`2026-RegularizedFlowMatching-results.tar.gz` (161 MB; CC BY 4.0), whose URL and sha256 are in
`experiments/report/results_archive.txt`. `make fetch-results` downloads it (`RESULTS_URL`
overrides the URL and accepts `https://`, `file://` or a local path; `RESULTS_SHA256` overrides
the checksum for another archive), checks its sha256 before unpacking, unpacks it into
`RESULTS_ROOT/2026-RegularizedFlowMatching-results/` and checks every file against the
archive's `SHA256SUMS`. `make release-report` runs the report from its `paper/results/`. The
archive holds the 123 result files the report reads (unchanged copies) and the manifests and
logs of the 126 runs of `make list` (the cost table reads the manifests; the text quotes some
logs), with host names and absolute paths removed; `SOURCES.txt` in the archive names each
file's run directory and commit, and its `README.md` maps the files to the tables and
figures.
`experiments/report/release_results.py` builds the archive from the composed directory below:
it runs the report under an audit hook that records every file the report opens, copies those
files and the run manifests and logs (sanitized by `sanitize_text` and `sanitize_log`), and
packs them deterministically (the same inputs give the same
bytes).

The original result store has one directory per line of
`experiments/report/paper_results.txt`, each with its `results/` subdirectory and the logs:

    results/c2ddb03/results/oned.npz, oned.log, oned.manifest.json, ...
    results/57a5270-v2/results/...
    results/c530cb8-stability/results/...
    results/v6/results/...
    results/v7/results/...
    results/v8/results/...

With `RESULTS_ROOT` pointing at such a store, `make paper-report` composes
`RESULTS_ROOT/paper/results` from these run directories (symlinks; a later directory overrides
an earlier one per file; `RESULTS_ROOT/paper/SOURCES.txt` names each file's run) and runs the
report from it; the release archive is this composed directory, restricted to the files the
report reads. `make compact RESULTS=<results directory>` extracts only the arrays the report
reads (38 MB, images as float32) into `results/compact/`, and `make report
RESULTS=results/compact` rebuilds the same tables and figures from them; `make compact` also
packs that directory into `results/compact.tar.gz` (25 MB). The report regression of the tests
reads `$SMOOTHFM_REPORT_RESULTS`, else the unpacked release archive, else `results/compact/`.
Neither the results root nor the paper repository has a machine-specific default; set them as

    export RESULTS_ROOT=<directory of the unpacked results>
    export SMOOTHFM_PAPER=<paper repository>

(`make sync` uses `SMOOTHFM_PAPER` when `PAPER` is not given; the tests that compare the
generated file names with the paper's `\input`s read `SMOOTHFM_PAPER` and skip without it).

### Publishing the archive

    uv run python experiments/report/upload_results.py --archive <dir>/2026-RegularizedFlowMatching-results.tar.gz \
        --dest <user>@<server>:<directory>/ --url https://<site>/<path>/2026-RegularizedFlowMatching-results.tar.gz

(or `RESULTS_ARCHIVE`, `RESULTS_DEST`, `RESULTS_URL`). It checks the archive against its
`.sha256` file and its members against the `SHA256SUMS` inside, and prints the rsync command
and the checks it would run; nothing leaves the machine without `--yes`. With `--yes` it
uploads the archive and its `.sha256` with rsync (the caller's ssh configuration; the script
stores no host or credential), downloads the public URL to a temporary file, compares its
sha256 and only then writes `url` and `sha256` into `experiments/report/results_archive.txt`.
`--verify-only` skips the upload (for a manual upload).

### Spot check of a code tree against the archive

    make fetch-results RESULTS_URL=<URL or path of the archive>
    make spotcheck SPOT_ARGS=--dry-run          # plan and time estimate only
    make spotcheck SPOT_OUT=<scratch dir>       # SPOT_WORKERS (default 8)
    make spotcheck-tree AGAINST=<other tree>    # file by file, seconds

`experiments/report/spotcheck.py` recomputes four complete runs of `experiments/runs.txt` with
the tree it lives in and compares them key by key with the archived results:
lam_sweep_trimodal_n100 and lam_sweep_checkerboard_n100 (1D and 2D solver, hold-out over
constant lambda, RK4, TV), stability_1d_m256 (perturbed training sets, re-selected hold-out
lambda, TV and H2 distances) and kde2d_checkerboard (KDEs with LSCV). From the archived
manifests (wall time x workers) they took 3.97 core-h, about 30 min on 8 cores at the speed of
the production cores; the script prints this estimate before it starts. The compute scripts
have no option to restrict a run to some training sets or N (lam_sweep's `--n` aside), so the
check uses complete cheap runs; oracle selection (oned, twod, 4 to 12 core-h each) is not
covered. A full run refuses a non-git or dirty tree, so run the check in a fresh `git clone`,
not in a `git archive` export. Strings, integers and selected parameters (`lam`, `sel_idx`,
the constant grid) must match exactly; every other array must have the same NaN pattern and
differ by at most 1e-8 (FAIL above, WARN above 1e-12); timing keys are skipped. The script
prints a pass/fail table and the platform (Python, NumPy, SciPy, cvxopt, NumPy's BLAS).
`spotcheck-tree` compares the tracked and untracked unignored files of this tree with another
tree by sha256, ignoring `.DS_Store`, caches and `results/`.

### Provenance

The results were computed with earlier versions of this code, at the commits in the table
below (manifest field `commit`, each from a clean tree). The hashes identify commits of the
authors' development repository; this repository starts without that history, so they do not
resolve here. Run on the release archive, the report code of this release writes all eleven
tables (`tables/*.tex`), the four figure files of Figures 1 to 4 and their 56 data files byte
for byte as they appear in the paper's sources, and `numbers.txt` as committed in
`experiments/paper/`. The report code of the development repository at the time of the last
result already wrote the same tables and figure data.

| run directory | development commit (manifest field `commit`) | contents |
|---|---|---|
| `c2ddb03/results` | c2ddb03 | 100 of the 126 runs; the lines below add or replace the others |
| `57a5270-v2/results` | 57a5270 | integration_error_T |
| `c530cb8-stability/results` | c530cb8 | stability_1d[_m256, _m512, _m1024], stability_circle (positive candidates only) |
| `v6/results` | 37d9883, 0356ab6 | lam_sweep_T6_n100, _n3000 (hold-out constant row of Table 1) and stability_circle_l2 at 37d9883; stability_circle_l2_dx005 at 0356ab6 |
| `v7/results` | a339f69, c2ddb03 | controls[_iid]_T[_h1, _l2]_m512 (m = 512 rows of Table 6), controls_iid_T at every N (lambda -> 0 rows of Table 6); controls_circle_m512 and controls_circle_h1_m512 are from c2ddb03 |
| `v8/results` | 5ceb16c | tend_T (2D blocks of Table 4) |

A result that is missing prints `\tbd` in its table cells and "pending" in its figure panels,
and the report lists it. A cell that config requires but an existing result lacks (an N that
config lists and the result does not hold) also prints `\tbd` (tables.rowq).

## Paper item, report, results, compute

T runs over the 2D targets circle, checkerboard, spirals, gauss8; T6 over these and the 1D
targets trimodal, inhom; `[_h1, _l2]` are the gradient (k = 1) and smallness (k = 0)
regularizers, no suffix the Hessian (k = 2). Every compute run is `make compute-<result>`.
Table and figure numbers are those of the arXiv version, which contains the former supplement
as Appendices A to C; Tables 2 to 11 are in Appendix C.

| paper item | report (`experiments/report/`) | results read | compute script |
|---|---|---|---|
| Figure 1 (fig:hero), `figures/fig_hero.tex` | `hero.build` | hero_1d | `hero_1d.py` (reads oned, oned_eval) |
| Table 1 (tab:main), `tables/tab_main.tex` | `tables.tab_main` | oned, oned_eval, kde4, gabriel; twod_T, kde2d_T, twod_eval_T, kde4_2d_T; lam_sweep_T6, lam_sweep_T6_n100, _n3000 | `oned.py`, `oned_eval.py`, `kde4.py`, `gabriel.py`, `twod.py`, `kde2d.py`, `twod_eval.py`, `kde4_2d.py`, `lam_sweep.py` |
| Figure 2 (fig:accuracy), `figures/fig_accuracy.tex` | `figures.fig_accuracy` | controls_1d[_h1, _l2], controls_T[_h1, _l2], controls_iid_1d, controls_iid_T, controls_circle[_h1, _l2]_m512, controls_iid_circle_m512 | `controls.py`, `controls_iid.py` |
| Figure 3 (fig:stability), `figures/fig_stability.tex` | `stability.build_figure` | stability_1d, stability_circle, stability_circle_l2_dx005 | `stability.py` |
| Figure 4 (fig:twod-ops), `figures/fig_twod_ops.tex` | `figures.fig_twod_ops` | twod_eval_T[_h1, _l2], kde4_2d_T | `twod_eval.py`, `kde4_2d.py` |
| Table 2 (tab:oned-full), `tables/tab_oned_full.tex` | `tables.tab_oned_full` | oned, oned_eval, kde4, gabriel, controls_iid_1d | `oned.py`, `oned_eval.py`, `kde4.py`, `gabriel.py`, `controls_iid.py` |
| Table 3 (tab:twod-full), `tables/tab_twod_full.tex` | `tables.tab_twod_full` | twod_T, kde2d_T, twod_eval_T, kde4_2d_T, em8, controls_iid_T | `twod.py`, `kde2d.py`, `twod_eval.py`, `kde4_2d.py`, `em8.py`, `controls_iid.py` |
| Table 4 (tab:tend), `tables/tab_tend.tex` | `tables.tab_tend` | tend, tend_T | `tend.py` |
| Table 5 (tab:refine-1d), `tables/tab_refine_1d.tex` | `tables.tab_refine_1d` | controls_1d[_h1, _l2], controls_iid_1d | `controls.py`, `controls_iid.py` |
| Table 6 (tab:refine-2d), `tables/tab_refine_2d.tex` | `tables.tab_refine_2d` | controls_T[_h1, _l2], controls_T[_h1, _l2]_m512, controls_iid_T, controls_iid_T_m512 | `controls.py`, `controls_iid.py` |
| Table 7 (tab:integrator), `tables/tab_integrator.tex` | `tables.tab_integrator` | integration_error, integration_error_T, oned_eval | `integration_error.py` |
| Table 8 (tab:stability), `tables/tab_stability.tex` | `stability.write_table` | stability_1d, stability_circle, stability_circle_l2_dx005 | `stability.py` |
| Table 9 (tab:stability-refine), `tables/tab_stability_refine.tex` | `stability.write_refine_table` | stability_1d_m256, _m512, _m1024 | `stability.py --dim 1 --m M` |
| Table 10 (tab:reg-full), `tables/tab_reg_full.tex` | `tables.tab_reg_full` | oned[_h1, _l2], twod_T[_h1, _l2] | `oned.py --reg`, `twod.py --reg` |
| Table 11 (tab:lam-sweep-full), `tables/tab_lam_sweep_full.tex` | `tables.tab_lam_sweep_full` | lam_sweep_T6[_h1, _l2] | `lam_sweep.py` |

Dependencies between results (for example twod_eval_T reads twod_T and kde2d_T) are
the third column of `experiments/runs.txt`; a script whose dependency is missing exits with a
message naming it.

## Numbers in the text

Numbers of the text that no table or figure prints come from `experiments/paper/numbers.txt`
(written by `experiments/report/prose_numbers.py`, one line per number with the exact value,
the source arrays and the compute run) or, where marked, from a run log of the original result
store, which the release archive contains in sanitized form:

| paper location | number | report function | result | compute script |
|---|---|---|---|---|
| Sec. 4.2 (sec:cv), last paragraph | TV change up to 0.005 when the solution is interpolated in time | log: `compare_time.log`, summary | compare_time | `compare_time.py` |
| Sec. 5.2 (sec:setup) | latent TV to N(0, I) 5.7e-7 (1D), 1.4e-5 (2D) | log: `floors.log` and `compare_time.log`, worked example | floors, compare_time | `floors.py`, `compare_time.py` |
| Sec. 5.2 (sec:setup) | population-velocity TV 1.0e-4 and 2.5e-4, same order with 48 and 192 steps | log: `floors.log`, summary | floors | `floors.py` |
| Sec. 5.3 (sec:flowaccuracy) | against the best KDE for each N separately, 18 to 39% lower on the noisy circle and on the spirals with N >= 1000, between 2% lower and 26% higher in the other cases | `prose_numbers.kde_best_per_n` | oned, oned_eval, kde4; twod_T, kde2d_T, twod_eval_T, kde4_2d_T | `oned.py`, `oned_eval.py`, `kde4.py`, `twod.py`, `kde2d.py`, `twod_eval.py`, `kde4_2d.py` |
| Sec. 5.4 (sec:discretization) | the 96 RK4 steps change the TV error of the hold-out flows by at most 2.6e-4 (1D) and 9e-4 (2D) against DOP853 | `prose_numbers.integrator_tv` | integration_error, integration_error_T | `integration_error.py [--target T]` |
| Sec. 5.5 (sec:stability) | candidate changes at no more than 3 of the 48 times | `prose_numbers.lam_changed` | stability_1d, stability_circle | `stability.py --dim 1`, `stability.py --target circle` |
| Sec. 5.5 | laws within a factor of 3.4, fields up to 42 | `prose_numbers.resel_factors` | stability_1d, stability_circle | `stability.py --dim 1`, `stability.py --target circle` |
| Sec. 5.5 | field difference of v_N grows tenfold per decade of 1 - t | `prose_numbers.exact_field_growth` | stability_1d, stability_circle | `stability.py --dim 1`, `stability.py --target circle` |
| Sec. 5.5 | from m = 256 to 1024 the smallness constants grow by 5 to 9 (TV) and 64 to 109 (H2 field); the Hessian TV and L2 constants with fixed lambda(t) do not change, its H2 constant grows by at most 6 | Table 9, growth rows (`stability.write_refine_table`) | stability_1d_m256, _m512, _m1024 | `stability.py --dim 1 --m M` |
| Sec. 5.5 | 68 and 108 times as much on the circle | `prose_numbers.circle_ratio` | stability_circle_l2_dx005, stability_circle | `stability.py --target circle --reg l2 --grid-dx 0.005`, `stability.py --target circle` |
| Sec. 5.5 | density mass off by up to 1.1% at grid spacing 0.02 | log: `stability_circle_l2.log`, grid mass | stability_circle_l2 | `stability.py --target circle --reg l2` |
| Sec. 5.6 (sec:operator) | null space for at most 5.4% of the pairs | `prose_numbers.null_space_share` | oned[_h1, _l2], twod_T[_h1, _l2] | `oned.py [--reg R]`, `twod.py --target T [--reg R]` |
| Sec. 5.6 | Hessian more accurate than gradient in 57 of the 60 sets | `prose_numbers.hessian_vs_gradient` | oned[_h1], twod_T[_h1] | `oned.py [--reg h1]`, `twod.py --target T [--reg h1]` |
| Sec. 5.6 | 31% of the samples within 1e-3 of a training point, against 15 to 17% | `prose_numbers.operator_1d` | gabriel_hero_reg | `gabriel_hero.py --regs` |
| Sec. 5.6 | hold-out lambda/lambda_s about 3e6 | `prose_numbers.operator_1d` | gabriel_hero_reg | `gabriel_hero.py --regs` |
| Sec. 5.7 (sec:lambda) | hold-out within one decade of the oracle, up to three decades | `prose_numbers.holdout_vs_oracle_1d` | oned | `oned.py` |
| Sec. 5.7 | constant lambda less accurate on all 60 training sets | `prose_numbers.const_vs_holdout` | twod_T, twod_eval_T, lam_sweep_T, lam_sweep_T_n100, _n3000 | `twod.py`, `twod_eval.py`, `lam_sweep.py` |
| Sec. 5.7 | 77% of the pairs with t < 0.9 | `prose_numbers.circle_late_share` | twod_circle | `twod.py --target circle` |

## Running the compute stage

Every script takes `--smoke`, `--example`, `--workers N` (default `$NWORKERS`), `--overwrite`
and `--results DIR` (default `results`).

Laptop smoke run (minutes per script, a few hours for all; N = 100, one training set, coarse
meshes, 9 of the 49 times, `config.SMOKE`):

    make smoke NWORKERS=4                 # every run, into results/smoke/
    make report RESULTS=results/smoke OUT=/tmp/smoke_out

`make smoke-g1` .. `smoke-g4` run one CI group with the dependencies it needs. `--example`
prints and logs only the worked example of a script's first job.

Full run: `make compute-<name>` runs one result, `make compute` all 126 in dependency order
(one at a time; on one machine this takes weeks). A full run refuses a dirty tree (`src/`,
`experiments/`) and an existing result (`OVERWRITE=1` passes `--overwrite`). Apart from the
dependencies in `experiments/runs.txt` the runs are independent, so they can be spread over any
number of machines, each running `make compute-<name>` once the results it reads are in its
results directory. The results of the paper used one compute node per run with 14 to 28 worker
processes (`NWORKERS`). Memory is small except for `stability_circle_l2_dx005` (TV grid of
1041^2 points), which peaks at about 0.8 GB per worker in its smoke run.

Expected cost: the 126 runs took 6802 core-h on Intel Xeon E5-4627 v3 and Xeon Gold 6248R
cores, one BLAS thread per worker. The largest scripts are tend (5047 core-h over 5 runs, 4812
of them for the four 2D runs tend_T), controls (724 over 27), integration_error (320 over 5),
lam_sweep (183 over 30) and oned_eval (106). A core-hour here is wall time times worker
processes, so it includes workers that wait for the last jobs of a run. In tend_T the jobs
themselves used 1650 core-h; each run took 50 to 58 hours of wall time.
`tables/tab_cost.tex`, written by the report (`cost.py`, restricted to the runs of `make list`)
but not input by the paper, lists wall time, core-h and factorizations per script from the
manifests.

## Results that no table prints

- `gabriel_hero_reg`: the hold-out flows of the three regularizers and Gabriel et al.'s G1 on
  the trimodal target, N = 100; `prose_numbers.operator_1d` reads the Sec. 5.6 numbers from it.
- `floors`, `compare_time`, `stability_circle_l2`: only their logs are quoted (table above).

## Tests

    make test       # unit tests, pytest -m "not slow"
    make test-all   # adds every script with --smoke and --example, and the report regression

`.github/workflows/ci.yml` runs the unit tests, the smoke runs in four groups and the report
regression on Python 3.12 and 3.13.
