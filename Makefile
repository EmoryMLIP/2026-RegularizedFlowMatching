# smoothfm: compute and report stages of "Explicit Regularization for Flow Matching"
# (docs/REPRODUCE.md).
#
#   make compute-<name>          one result results/<name>.{npz,log,manifest.json} (full run; a
#                                full run refuses a dirty tree and an existing result, unless
#                                OVERWRITE=1, which appends --overwrite)
#   make compute                 every result, one at a time, in the order of experiments/runs.txt
#   make smoke                   every invocation with --smoke (results/smoke/), same order
#   make smoke-g1 .. smoke-g4    one CI group, with the dependencies it needs from other groups
#   make list                    the result names of experiments/runs.txt
#   make report                  tables and figures: RESULTS (default results) -> OUT
#                                (default experiments/paper)/{tables,figures}
#   make hero                    the hero figure fig_hero (Figure 1) only, RESULTS -> OUT
#   make compact                 the arrays the report reads -> results/compact/, packed into
#                                results/compact.tar.gz (a reduced copy: images as float32)
#   make fetch-results           download the release archive of the results
#                                (experiments/report/results_archive.txt; RESULTS_URL and
#                                RESULTS_SHA256 override it), check its sha256, unpack it into
#                                RESULTS_ROOT/<name>/ and check its SHA256SUMS
#   make release-report          the report from the fetched archive,
#                                RESULTS_ROOT/<name>/paper/results -> OUT
#   make spotcheck               recompute four cheap runs with this tree (a git clone) and
#                                compare them key by key with the fetched archive (about 30 min
#                                on 8 cores; SPOT_OUT scratch dir, SPOT_WORKERS, SPOT_ARGS=--dry-run)
#   make spotcheck-tree AGAINST=<dir>  this tree against another, file by file (sha256)
#   make paper-report            the paper's inputs: RESULTS_ROOT/paper/results composed from the
#                                result directories of experiments/report/paper_results.txt
#                                (symlinks, later directories override earlier ones per file;
#                                RESULTS_ROOT/paper/SOURCES.txt names each file's source), then
#                                the report from it -> OUT (default experiments/paper)
#   make sync PAPER=<dir>        copy OUT/tables and OUT/figures into the paper repository
#                                (PAPER defaults to $SMOOTHFM_PAPER; no other default)
#   make test / make test-all    pytest without / with the slow tiers
#
# The compute and smoke targets are generated from experiments/runs.txt. The measured cost of
# each run is in tables/tab_cost.tex and docs/REPRODUCE.md.

NWORKERS ?= 10
export NWORKERS
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1
export VECLIB_MAXIMUM_THREADS = 1

PY := uv run python
COMPUTE := experiments/compute
REPORT := experiments/report
RESULTS ?= results
OUT ?= experiments/paper
COMPACT ?= results/compact
COMPACT_ARCHIVE ?= results/compact.tar.gz
# the release archive of the results: name, url, sha256 (url and sha256 overridable by
# RESULTS_URL, RESULTS_SHA256)
RESULTS_CONF := $(REPORT)/results_archive.txt
RELEASE_NAME := $(shell awk '$$1 == "name" {print $$2}' $(RESULTS_CONF))
# the release results, laid out as <run>/results per line of PAPER_SPEC (docs/REPRODUCE.md)
RESULTS_ROOT ?= results
RELEASE_RESULTS = $(RESULTS_ROOT)/$(RELEASE_NAME)/paper/results
PAPER_SPEC ?= $(REPORT)/paper_results.txt
OW = $(if $(filter 1,$(OVERWRITE)), --overwrite)

# ---------------------------------------------------------------- the run list
# One word per line of experiments/runs.txt, <name>|<group>|<deps>|<script>|<arg>|..., in file
# order; the functions below read its fields.
RUNS_TXT := experiments/runs.txt
RUN_WORDS := $(shell awk 'substr($$1, 1, 1) != "\043" && NF { s = $$1; for (i = 2; i <= NF; i++) s = s "|" $$i; print s }' $(RUNS_TXT))
COMPUTE_ORDER := $(foreach w,$(RUN_WORDS),$(firstword $(subst |, ,$(w))))
GROUPS := g1 g2 g3 g4

run_fields = $(subst |, ,$(filter $(1)|%,$(RUN_WORDS)))
run_script = $(word 4,$(call run_fields,$(1)))
run_args = $(wordlist 5,$(words $(call run_fields,$(1))),$(call run_fields,$(1)))
# " --a b ..." or empty
run_argstr = $(if $(call run_args,$(1)), $(call run_args,$(1)))
# the names of a group and of every run it depends on (transitively), in file order
group_names = $(shell awk -v g=$(1) 'substr($$1, 1, 1) != "\043" && NF { n++; name[n] = $$1; grp[n] = $$2; dep[n] = $$3 } \
  END { for (i = n; i >= 1; i--) { if (grp[i] == g) need[name[i]] = 1; \
          if (need[name[i]] && dep[i] != "-") { k = split(dep[i], d, ","); for (j = 1; j <= k; j++) need[d[j]] = 1 } } \
        for (i = 1; i <= n; i++) if (need[name[i]]) print name[i] }' $(RUNS_TXT))

compute_cmd = $(PY) $(COMPUTE)/$(call run_script,$(1)).py$(call run_argstr,$(1)) --results $(RESULTS) --workers $(NWORKERS)$(OW)
smoke_cmd = $(PY) $(COMPUTE)/$(call run_script,$(1)).py --smoke$(call run_argstr,$(1)) --results $(RESULTS) --workers $(NWORKERS)

define NL


endef

.PHONY: compute smoke $(addprefix smoke-,$(GROUPS)) report hero compact fetch-results \
  release-report spotcheck spotcheck-tree paper-report sync test test-all list $(addprefix compute-,$(COMPUTE_ORDER))

# ---------------------------------------------------------------- report stage (seconds)

report:
	$(PY) $(REPORT)/report.py --results $(RESULTS) --out $(OUT)

compact:
	$(PY) $(REPORT)/report.py --results $(RESULTS) --out $(OUT) --compact-out $(COMPACT)
	$(PY) $(REPORT)/compact_archive.py $(COMPACT) $(COMPACT_ARCHIVE)

# The archive unpacks to <name>/ (README.md, SOURCES.txt, SHA256SUMS, paper/results/). It is
# unpacked only after its sha256 matches; the download replaces RESULTS_ROOT/<name>.
# url: https://... or file://... (curl), or a local path (cp).
fetch-results:
	@url="$${RESULTS_URL:-$$(awk '$$1 == "url" {print $$2}' $(RESULTS_CONF))}"; \
	sha="$${RESULTS_SHA256:-$$(awk '$$1 == "sha256" {print $$2}' $(RESULTS_CONF))}"; \
	case "$$url $$sha" in *TODO*) \
	  echo "fetch-results: $(RESULTS_CONF) holds a placeholder; set RESULTS_URL (https://..., file://... or a path) and RESULTS_SHA256"; \
	  exit 1;; esac; \
	test -n "$(RELEASE_NAME)" -a -n "$(RESULTS_ROOT)" || { \
	  echo "fetch-results: no name in $(RESULTS_CONF) or empty RESULTS_ROOT"; exit 1; }; \
	file="$(RESULTS_ROOT)/$(RELEASE_NAME).tar.gz"; \
	mkdir -p "$(RESULTS_ROOT)" && \
	case "$$url" in \
	  *://*) curl -fsSL -o "$$file" "$$url";; \
	  *) test "$$url" -ef "$$file" || cp "$$url" "$$file";; \
	esac && \
	echo "$$sha  $$file" | shasum -a 256 -c - && \
	rm -rf "$(RESULTS_ROOT)/$(RELEASE_NAME)" && tar -xzf "$$file" -C "$(RESULTS_ROOT)" && \
	(cd "$(RESULTS_ROOT)/$(RELEASE_NAME)" && shasum -a 256 -c --quiet SHA256SUMS) && \
	echo "$(RELEASE_RESULTS): $$(ls "$(RELEASE_RESULTS)" | wc -l | tr -d ' ') files, checksums OK"

release-report:
	$(PY) $(REPORT)/report.py --results "$(RELEASE_RESULTS)" --out $(OUT)

SPOT_OUT ?= spotcheck_out
SPOT_WORKERS ?= 8
spotcheck:
	$(PY) $(REPORT)/spotcheck.py --results "$(RELEASE_RESULTS)" --out "$(SPOT_OUT)" \
	  --workers $(SPOT_WORKERS) $(SPOT_ARGS)

spotcheck-tree:
	@test -n "$(AGAINST)" || { echo "usage: make spotcheck-tree AGAINST=<other tree>"; exit 1; }
	$(PY) $(REPORT)/spotcheck.py --tree-only --against "$(AGAINST)"

hero:
	$(PY) $(REPORT)/report.py --results $(RESULTS) --out $(OUT) --only hero

# the composed directory holds symlinks only; compose_results.py refuses to recreate it
# otherwise and never deletes anything outside it
paper-report:
	$(PY) $(REPORT)/compose_results.py --spec "$(PAPER_SPEC)" --root "$(RESULTS_ROOT)" \
	  --dest "$(RESULTS_ROOT)/paper/results" --sources "$(RESULTS_ROOT)/paper/SOURCES.txt"
	$(PY) $(REPORT)/report.py --results "$(RESULTS_ROOT)/paper/results" --out $(OUT)

# the paper repository: PAPER=<dir> or the environment variable SMOOTHFM_PAPER; no default
PAPER ?= $(SMOOTHFM_PAPER)
sync:
	@test -n "$(PAPER)" || { echo "usage: make sync PAPER=<paper repository> (or export SMOOTHFM_PAPER)"; exit 1; }
	mkdir -p "$(PAPER)/tables" "$(PAPER)/figures"
	cp -R $(OUT)/tables/. "$(PAPER)/tables/"
	cp -R $(OUT)/figures/. "$(PAPER)/figures/"

# ---------------------------------------------------------------- tests

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

list:
	@echo $(COMPUTE_ORDER) | tr ' ' '\n'

# ---------------------------------------------------------------- compute stage (hours)
# compute runs the targets one after another (a sub-make each), so make -j cannot reorder them
# and a dependency is never rerun; a target that fails stops the loop.

compute:
	@for n in $(COMPUTE_ORDER); do $(MAKE) --no-print-directory compute-$$n || exit 1; done

define COMPUTE_RULE
compute-$(1):
	$$(call compute_cmd,$(1))
endef
$(foreach n,$(COMPUTE_ORDER),$(eval $(call COMPUTE_RULE,$(n))))

# ---------------------------------------------------------------- smoke (CI configuration)
# one recipe line per run (each line is its own shell, so a failing run stops the target)

smoke:
	$(foreach n,$(COMPUTE_ORDER),$(call smoke_cmd,$(n))$(NL))

define SMOKE_RULE
smoke-$(1):
	$$(foreach n,$$(call group_names,$(1)),$$(call smoke_cmd,$$(n))$$(NL))
endef
$(foreach g,$(GROUPS),$(eval $(call SMOKE_RULE,$(g))))
