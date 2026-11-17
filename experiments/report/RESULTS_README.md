# Results of "Explicit Regularization for Flow Matching"

This archive holds the computed results behind every table, figure and quantitative
statement of the paper

> Lars Ruthotto and Eran Treister, *Explicit Regularization for Flow Matching*, arXiv preprint,
> 2026, [arXiv:2611.19587](https://arxiv.org/abs/2611.19587).

The code that computed the results and writes the tables and figures from them is
<https://github.com/EmoryMLIP/2026-RegularizedFlowMatching>. The report stage of that code
reads this archive and writes the paper's tables and figures in seconds; the compute stage that
produced the archive took about 6800 core-hours.

## Layout

    2026-RegularizedFlowMatching-results/
      README.md          this file
      SOURCES.txt        per file: run directory of the original result store, commit
      SHA256SUMS         sha256 of every other file (shasum -a 256 -c SHA256SUMS)
      paper/results/     123 result files <name>.npz, and for each of the 126 runs its
                         manifest <name>.manifest.json and its log <name>.log

A result `<name>` is one invocation of a compute script, listed in `experiments/runs.txt` of
the code with its arguments; `make compute-<name>` reruns it. Each `.npz` holds NumPy arrays
(no pickled objects; load with `numpy.load(path, allow_pickle=False)`). Each manifest
records the commit, a flag that the source tree was clean, the command line, the
parameters and configuration constants, the package versions, the CPU model, the worker
count, the start and end time, the wall time per job and the sha256 of the result file and
the log. Each log starts with the same provenance and every configuration constant, prints a
worked example of the first job, and then records per job the selected lambda(t), the
factorizations, the timing and the metrics as the run computed them.

## File groups

T runs over the two-dimensional targets circle, checkerboard, spirals, gauss8; T6 over these
and the one-dimensional targets trimodal and inhom. The suffixes `_h1` and `_l2` mark the
gradient (k = 1) and the smallness (k = 0) regularizer; no suffix is the Hessian (k = 2).
Table and figure numbers are those of the arXiv version (Tables 2 to 11 are in Appendix C).

| results | experiment | paper item |
|---|---|---|
| `oned`, `oned_h1`, `oned_l2`, `oned_eval`, `kde4`, `gabriel` | 1D flows (oracle and hold-out lambda), unregularized flows, KDEs, convolution baseline | Table 1, Table 2, Table 10 |
| `twod_T[_h1, _l2]`, `twod_eval_T[_h1, _l2]`, `kde2d_T`, `kde4_2d_T`, `em8` | 2D flows, their evaluation, KDEs, Gaussian mixture | Table 1, Table 3, Table 10, Figure 4 |
| `hero_1d` | trajectories of the empirical and the regularized flow | Figure 1 |
| `gabriel_hero_reg` | hold-out flows of the three regularizers, trimodal, N = 100 | numbers of Sec. 5.6 |
| `tend`, `tend_T` | stopping the flow at t_end < 1 | Table 4 |
| `controls_1d[_h1, _l2]`, `controls_T[_h1, _l2]`, `controls_T[_h1, _l2]_m512`, `controls_iid_1d`, `controls_iid_T[_m512]` | mesh refinement and the limit lambda -> 0 | Figure 2, Tables 5 and 6 (lambda -> 0 rows also in Tables 2 and 3) |
| `integration_error`, `integration_error_T` | RK4 against DOP853 and a step-doubled reference | Table 7, numbers of Sec. 5.4 |
| `stability_1d`, `stability_circle`, `stability_circle_l2_dx005` | perturbation of the training data | Figure 3, Table 8, numbers of Sec. 5.5 |
| `stability_1d_m256`, `_m512`, `_m1024` | stability constants under mesh refinement | Table 9 |
| `lam_sweep_T6[_h1, _l2]`, `lam_sweep_T6_n100`, `_n3000` | multiples of the hold-out lambda(t) and constant lambda | Table 11, hold-out constant row of Table 1, numbers of Sec. 5.7 |

The numbers of the text that no table prints are recomputed by the report into
`experiments/paper/numbers.txt` of the code, one line per number with its source arrays.

Three runs have no `.npz` here: `floors`, `compare_time` and `stability_circle_l2`. The
report reads no array of these runs; the paper quotes their logs: `compare_time.log` (Sec. 4.2,
time interpolation; Sec. 5.2, latent truncation in 2D), `floors.log` (Sec. 5.2, latent
truncation in 1D and population-velocity error) and `stability_circle_l2.log` (Sec. 5.5, grid
mass). `docs/REPRODUCE.md` of the code names the log line of each number. The cost table of
the code (`tables/tab_cost.tex`) sums the cost of all 126 runs from the manifests.

## Provenance

The results were computed at the commits listed below (the manifest field `commit`;
`SOURCES.txt` gives it per file), each from a clean source tree. The hashes identify commits
of the authors' development repository; the public code repository starts without that
history, so they do not resolve there. The released code regenerates every table and figure
of the paper from the archived results byte for byte (below). The run directories are those
of the original result store; the code composes them in this order, a later directory
replacing an earlier one per file (`experiments/report/paper_results.txt`).

| run directory | commit | results |
|---|---|---|
| `c2ddb03/results` | c2ddb03 | 78 results and the manifests of floors and compare_time |
| `57a5270-v2/results` | 57a5270 | integration_error_T |
| `c530cb8-stability/results` | c530cb8 | stability_1d[_m256, _m512, _m1024], stability_circle |
| `v6/results` | 37d9883 | lam_sweep_T6_n100, _n3000; manifest of stability_circle_l2 |
| `v6/results` | 0356ab6 | stability_circle_l2_dx005 |
| `v7/results` | a339f69 | controls_T[_h1, _l2]_m512 and controls_iid_T[_m512] except the two below |
| `v7/results` | c2ddb03 | controls_circle_m512, controls_circle_h1_m512 |
| `v8/results` | 5ceb16c | tend_T |

The report code of the release, run on this archive, writes all eleven tables
(`tables/*.tex`), the four figure files of Figures 1 to 4 and their 56 data files byte for
byte as they appear in the paper.

The `.npz` files are byte copies of the original results; their sha256 equal the ones the
manifests record. The manifests and logs are edited copies: the absolute paths of the result
directories (in the command lines, the recorded sources and the output lists) are replaced by
`<results>`, the manifests lose the host name, and the log headers record
`host unrecorded`. Everything else, including every number, is unchanged; the sha256 of a
log therefore differs from the one its manifest records for the original. Completion markers
(`.done`) are not included, and neither are 15 results that no table, figure or number of the
paper reads.

## Use with the code

    git clone https://github.com/EmoryMLIP/2026-RegularizedFlowMatching
    cd 2026-RegularizedFlowMatching
    uv sync --frozen --extra cholmod
    make fetch-results RESULTS_URL=<URL or path of this archive>
    make release-report OUT=<dir>

`make fetch-results` downloads the archive (the URL and sha256 are in
`experiments/report/results_archive.txt`; `RESULTS_URL` and `RESULTS_SHA256` override them,
and a local path or a `file://` URL also works), checks its sha256, unpacks it into
`RESULTS_ROOT/2026-RegularizedFlowMatching-results/` (default `RESULTS_ROOT=results`) and
checks `SHA256SUMS`. `make release-report` writes the tables and figures from
`paper/results/` to `OUT/tables` and `OUT/figures`. Without the Makefile:

    uv run python experiments/report/report.py \
        --results results/2026-RegularizedFlowMatching-results/paper/results --out <dir>

## License

The results are licensed under the Creative Commons Attribution 4.0 International License
(CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>). Please cite the paper when you
use them. The code is under the MIT License of its repository.
