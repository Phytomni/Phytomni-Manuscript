#!/usr/bin/env bash
# One-command figure reproduction for the Phytomni manuscript repo.
# Runs every figure notebook/script with PHYTOMNI_SAVE=1 so each emits artifacts
# into that figure's own output/ dir. Missing-data / missing-toolchain targets are
# skipped (never a hard failure). `--check` prints a summary and exits non-zero on
# any failure of a runnable target (for smoke tests / CI).
set -uo pipefail
cd "$(dirname "$0")"

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1
export PHYTOMNI_SAVE=1

have_r=1;  command -v Rscript >/dev/null 2>&1 || have_r=0
have_ir=1; jupyter kernelspec list 2>/dev/null | grep -qw ir || have_ir=0

pass=0; fail=0; skip=0
declare -a LINES

record() { # $1 sym(OK|XX|--)  $2 label  $3 note
  local sym="$1" mark
  case "$sym" in OK) mark="✓"; pass=$((pass+1));; XX) mark="✘"; fail=$((fail+1));; --) mark="⊘"; skip=$((skip+1));; esac
  LINES+=("$mark $2${3:+  ($3)}")
}

nb() { # $1 path  $2 label  $3 kernel(py|ir)
  if [[ "$3" == "ir" && $have_ir -eq 0 ]]; then
    record -- "$2" "ir kernel missing: R -e \"IRkernel::installspec()\""; return; fi
  if jupyter nbconvert --to notebook --execute --inplace "$1" >/dev/null 2>&1; then
    record OK "$2"; else record XX "$2"; fi
}

# --- Python notebooks ---
nb "Fig. 2/fig. 2.ipynb"                                   "Fig. 2"            py
nb "Fig. 3/fig. 3.ipynb"                                   "Fig. 3"            py
nb "Extended Data Fig. 5/extended_data_fig. 5d.ipynb"      "Ext. Data 5d"      py
nb "Extended Data Fig. 6/extended_data_fig. 6de.ipynb"     "Ext. Data 6d,e"    py
nb "Extended Data Fig. 6/extended_data_fig. 6fg.ipynb"     "Ext. Data 6f,g"    py
nb "Supplementary Fig. 8/supplementary_fig. 8.ipynb"       "Supp. 8"           py
nb "Supplementary Fig. 9/supplementary_fig. 9.ipynb"       "Supp. 9"           py
nb "Supplementary Fig. 10-13/supplementary_fig. 10-13.ipynb" "Supp. 10-13"     py
nb "Supplementary Fig. 14/supplementary_fig. 14.ipynb"     "Supp. 14"          py
nb "Supplementary Fig. 19/supplementary_fig. 19.ipynb"     "Supp. 19"          py
nb "Supplementary Fig. 24/supplementary_fig. 24.ipynb"     "Supp. 24"          py

# --- R notebooks (ir kernel) ---
nb "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb"     "Ext. Data 5a,b"    ir
nb "Extended Data Fig. 6/extended_data_fig. 6abc.ipynb"    "Ext. Data 6a-c"    ir
nb "Extended Data Fig. 7/extended_data_fig. 7.ipynb"       "Ext. Data 7"       ir

# --- standalone Python script (runs from its own dir) ---
if ( cd "Supplementary Fig. 7" && python3 plot.py ) >/dev/null 2>&1; then
  record OK "Supp. 7 (plot.py)"; else record XX "Supp. 7 (plot.py)"; fi

# --- standalone R script ---
if [[ $have_r -eq 0 ]]; then
  record -- "Ext. Data 5c (R script)" "Rscript not found"
elif ( cd "Extended Data Fig. 5" && Rscript "extended_data_fig. 5c.R" ) >/dev/null 2>&1; then
  record OK "Ext. Data 5c (R script)"; else record XX "Ext. Data 5c (R script)"; fi

# --- R Markdown radar (needs external data + ggradar) ---
RMD_DIR="Extended Data Fig. 6"
if [[ $have_r -eq 0 ]]; then
  record -- "Ext. Data 6 radar (Rmd)" "Rscript not found"
elif [[ ! -f "$RMD_DIR/PhytoBench-RAG-for_plot.csv" ]]; then
  record -- "Ext. Data 6 radar (Rmd)" "data pending: PhytoBench-RAG-for_plot.csv"
elif ! Rscript -e 'if(!requireNamespace("ggradar",quietly=TRUE)) quit(status=1)' >/dev/null 2>&1; then
  record -- "Ext. Data 6 radar (Rmd)" "ggradar missing: remotes::install_github('ricardo-bion/ggradar')"
elif Rscript -e 'rmarkdown::render("Extended Data Fig. 6/extended_data_fig. 6abc.Rmd")' >/dev/null 2>&1; then
  record OK "Ext. Data 6 radar (Rmd)"; else record XX "Ext. Data 6 radar (Rmd)"; fi

# --- report ---
printf '%s\n' "${LINES[@]}"
echo "————————————————————————————————————————————"
echo "$((pass+fail+skip)) targets / $pass ok / $fail failed / $skip skipped"
if [[ $CHECK -eq 1 && $fail -gt 0 ]]; then exit 1; fi
exit 0
