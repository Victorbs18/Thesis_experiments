"""
Render the full_comparison tables (tables/full_comparison_*.csv, written by full_comparison.py)
as one self-contained HTML page: tables/full_comparison.html

Run from the repo root:  python analysis/tables_to_html.py
"""
import csv
import html

SEL_METHODS = ['ATC', 'DOC', 'Nuc', 'MDE', 'Dsp', 'Agr', 'TV', 'MMD', 'Wass', 'PAD', 'CrA', 'CrA-TV']
R_METHODS = ['ATC', 'DOC', 'Nuc', 'MDE', 'Dsp', 'Agr', 'TV', 'CrA', 'MMD', 'Wass', 'PAD']
OURS = ('CrA', 'CrA-TV')

NOTES = {
    'ATC': 'Average Thresholded Confidence (Garg et al. 2022), pick max',
    'DOC': 'Difference of confidence, validation minus test, pick min',
    'Nuc': 'Nuclear norm of the test softmax matrix, pick max',
    'MDE': 'Mean dispersion energy, pick min',
    'Dsp': 'Dispersion of pseudo-class centroids, pick max',
    'Agr': 'Agreement with seeds of the same algorithm, pick max',
    'TV': 'Training-domain validation accuracy (standard selection), pick max',
    'MMD': 'MMD of test softmax to pooled ERM, direction from the estimated ACL',
    'Wass': 'Wasserstein distance to pooled ERM, direction from the estimated ACL',
    'PAD': 'Proxy A-distance to pooled ERM, direction from the estimated ACL',
    'CrA': 'Agreement with the ERM seeds on the test images, min if the estimated ACL is negative, max if positive '
           '(ERM seeds: leave-one-out agreement with the other ERM seeds)',
    'CrA-TV': 'Training-domain validation if the estimated ACL slope is positive, minimum CrA if negative',
    'Oracle': 'Selects by the test environment\'s out-split accuracy, reports its in-split accuracy (DomainBed)',
}

POOLED = [('pooled_cra_dg', 'CrA (DG)'), ('pooled_oracle_dg', 'Oracle (DG)'), ('pooled_cra_all', 'CrA (all)'),
          ('pooled_cratv_all', 'CrA-TV (all)'), ('pooled_oracle_all', 'Oracle (all)')]


def read(name):
    return list(csv.DictReader(open(f'tables/full_comparison_{name}.csv')))


def num(v):
    return float(v) if v not in ('', None) else None


def esc(s):
    return html.escape(str(s))


def gap_class(value, oracle):
    if value is None or oracle is None:
        return ''
    gap = oracle - value
    if gap <= 1:
        return 'g0'
    if gap <= 5:
        return 'g1'
    if gap <= 15:
        return 'g2'
    return 'g3'


def acl_table(acl):
    out = ['<table class="data"><thead><tr><th>Dataset</th><th>Models</th><th>True slope</th><th>True R</th>'
           '<th>Est. slope</th><th>Est. R</th><th>True regime</th><th>Est. regime</th><th>OOD est. error</th>'
           '<th>Axis c: train</th><th>c: test</th></tr></thead><tbody>']
    for r in acl:
        ok = r['sign_match'] == 'True'
        out.append(
            f'<tr><th scope="row">{esc(r["dataset"])}<span class="sub">{esc(r["algorithms"])}</span></th>'
            f'<td>{r["n_models"]}</td><td>{num(r["true_slope"]):+.2f}</td><td>{num(r["true_R"]):+.2f}</td>'
            f'<td>{num(r["est_slope"]):+.2f}</td><td>{num(r["est_R"]):+.2f}</td>'
            f'<td><span class="pill {"neg" if r["true_regime"] == "well-specified" else "pos"}">{esc(r["true_regime"])}</span></td>'
            f'<td><span class="pill {"neg" if r["est_regime"] == "well-specified" else "pos"}">{esc(r["est_regime"])}</span>'
            f' <span class="{"ok" if ok else "bad"}">{"✓" if ok else "✗"}</span></td>'
            f'<td>{num(r["ood_estimate_mae_pts"]):.1f} pts</td><td class="mono">{esc(r["c_train"])}</td>'
            f'<td class="mono">{num(r["c_test"]):.3f}</td></tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def pooled_table(acl):
    head = ''.join(f'<th class="{"ours" if "CrA" in label else ""}">{label}</th>' for _, label in POOLED)
    out = [f'<table class="data"><thead><tr><th>Dataset</th>{head}</tr></thead><tbody>']
    for r in acl:
        oracle = num(r['pooled_oracle_all_acc'])
        cells = []
        for key, label in POOLED:
            acc = num(r[f'{key}_acc'])
            cls = 'oracle' if 'Oracle' in label else f'{gap_class(acc / 100, oracle / 100)} ours'
            cells.append(f'<td class="{cls}">{acc:.1f}<span class="sub">{esc(r[f"{key}_pick"])}</span></td>')
        out.append(f'<tr><th scope="row">{esc(r["dataset"])}</th>{"".join(cells)}</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def checks_list(acl):
    items = []
    for r in acl:
        ok = r['labels_check'].startswith('ok')
        items.append(f'<li><b>{esc(r["dataset"])}</b>: <span class="{"ok" if ok else "bad"}">'
                     f'{"✓" if ok else "✗"}</span> {esc(r["labels_check"].split(": ", 1)[1])}</li>')
    return '<ul>' + ''.join(items) + '</ul>'


def selection_row(label, values, oracle, extra_th=''):
    """values: {method: (mean, std)}; best label-free method(s) in bold."""
    label_free = [values[m][0] for m in SEL_METHODS if values[m][0] is not None]
    best = max(label_free) if label_free else None
    cells = []
    for m in SEL_METHODS + ['Oracle']:
        mean, sd = values[m]
        if mean is None:
            cells.append('<td class="na">–</td>')
            continue
        cls = [gap_class(mean, oracle) if m not in ('Oracle',) else 'oracle']
        if m in OURS:
            cls.append('ours')
        if m in SEL_METHODS and best is not None and abs(mean - best) < 1e-9:
            cls.append('best')
        sd_html = f'<span class="sd">±{sd:.1f}</span>' if sd is not None else ''
        cells.append(f'<td class="{" ".join(c for c in cls if c)}">{mean:.1f}{sd_html}</td>')
    return f'<tr>{extra_th}<th scope="row">{esc(label)}</th>{"".join(cells)}</tr>'


def selection_table(sel, acl, summary):
    cols = SEL_METHODS + ['Oracle']
    head = ''.join(f'<th title="{esc(NOTES[m])}" class="{"ours" if m in OURS else ""}">{m}</th>' for m in cols)
    out = [f'<table class="data sel"><thead><tr><th>Algorithm</th>{head}</tr></thead>']
    for a in acl:
        ds = a['dataset']
        match = a['sign_match'] == 'True'
        out.append(f'<tbody><tr class="group"><th colspan="{len(cols) + 1}">{esc(ds)}'
                   f'<span class="meta">ACL true {num(a["true_slope"]):+.2f}, estimated {num(a["est_slope"]):+.2f}'
                   f' → {esc(a["est_regime"])}, CrA picks {"min" if a["est_regime"] == "well-specified" else "max"}'
                   f' · estimated sign = true sign: {"yes" if match else "no"}</span></th></tr>')
        rows = [r for r in sel if r['dataset'] == ds]
        for r in rows:
            values = {m: (num(r[m]), num(r[m + '_std'])) for m in cols}
            out.append(selection_row(r['algorithm'], values, values['Oracle'][0]))
        s = next(s for s in summary if s['dataset'] == ds)
        values = {m: (num(s[m]), None) for m in cols}
        out.append(selection_row(f'Mean of {s["n_dg_algorithms"]} DG', values, values['Oracle'][0])
                   .replace('<tr>', '<tr class="mean">', 1))
        if not match:   # true-sign variants are shown only where the estimated sign is wrong
            stars = ' · '.join(f'{esc(r["algorithm"])}: CrA* {num(r["CrA*"]):.1f}, CrA-TV* {num(r["CrA-TV*"]):.1f}'
                               for r in rows if r['CrA*'] and r['CrA-TV*'])
            out.append(f'<tr class="note"><td colspan="{len(cols) + 1}">With the true sign: {stars}</td></tr>')
        out.append('</tbody>')
    out.append('</table>')
    return ''.join(out)


def summary_table(summary):
    cols = SEL_METHODS + ['Oracle']
    head = ''.join(f'<th class="{"ours" if m in OURS else ""}">{m}</th>' for m in cols)
    out = [f'<table class="data sel"><thead><tr><th>Dataset</th>{head}</tr></thead><tbody>']
    for s in summary:
        values = {m: (num(s[m]), None) for m in cols}
        out.append(selection_row(s['dataset'], values, values['Oracle'][0]))
    out.append('</tbody></table>')
    return ''.join(out)


def spearman_table(spear, acl):
    head = ''.join(f'<th class="{"ours" if m == "CrA" else ""}">{m}</th>' for m in R_METHODS)
    out = [f'<table class="data"><thead><tr><th>Algorithm</th><th>Seeds</th>{head}</tr></thead>']
    for a in acl:
        ds = a['dataset']
        out.append(f'<tbody><tr class="group"><th colspan="{len(R_METHODS) + 2}">{esc(ds)}</th></tr>')
        for r in [r for r in spear if r['dataset'] == ds]:
            cells = []
            for m in R_METHODS:
                v = num(r[m])
                if v is None or v != v:
                    cells.append('<td class="na">–</td>')
                else:
                    cls = 'rpos' if v >= 0.7 else 'rneg' if v <= -0.7 else ''
                    cells.append(f'<td class="{cls} {"ours" if m == "CrA" else ""}">{v:+.2f}</td>')
            out.append(f'<tr><th scope="row">{esc(r["algorithm"])}</th><td>{r["n_seeds"]}</td>{"".join(cells)}</tr>')
        out.append('</tbody>')
    out.append('</table>')
    return ''.join(out)


STYLE = """
/* Layout: one reading column of section headings, each table in its own horizontal scroller */
:root {
  --bg: #f7f8f9; --surface: #ffffff; --fg: #1b2027; --muted: #5d6673; --line: #dde1e6;
  --accent: #2f5bd3; --accent-soft: #e6ecfb;
  --g0: #d9f2e1; --g1: #eef6dc; --g2: #fbeed6; --g3: #f9dcdc;
  --pos: #1f6f43; --neg: #a03434;
  --font-head: "Space Grotesk", system-ui, sans-serif;
  --font-body: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12151a; --surface: #1a1f26; --fg: #e4e8ee; --muted: #9aa4b1; --line: #2c333d;
  --accent: #8aa8ff; --accent-soft: #1f2a45;
  --g0: #17392a; --g1: #2b3a1d; --g2: #43341a; --g3: #4a2020;
  --pos: #7fd3a2; --neg: #f19a9a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12151a; --surface: #1a1f26; --fg: #e4e8ee; --muted: #9aa4b1; --line: #2c333d;
  --accent: #8aa8ff; --accent-soft: #1f2a45;
  --g0: #17392a; --g1: #2b3a1d; --g2: #43341a; --g3: #4a2020;
  --pos: #7fd3a2; --neg: #f19a9a; color-scheme: dark }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 14px; line-height: 1.5;
  padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1400px; margin: 0 auto; display: grid; gap: 40px; }
header { display: grid; gap: 8px; max-width: 70ch; }
h1, h2 { font-family: var(--font-head); text-wrap: balance; margin: 0; }
h1 { font-size: 28px; letter-spacing: -0.01em; }
h2 { font-size: 19px; }
section { display: grid; gap: 12px; min-width: 0; }
p { margin: 0; color: var(--muted); max-width: 75ch; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
table.data { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
table.data th, table.data td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
table.data thead th { position: sticky; top: 0; background: var(--surface); font-family: var(--font-head);
  font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); cursor: help; }
table.data th[scope="row"], table.data thead th:first-child { text-align: left; }
table.data th[scope="row"] { font-weight: 600; }
tr.group th { text-align: left; background: var(--accent-soft); font-family: var(--font-head); font-size: 14px; padding-top: 10px; }
.meta { font-family: var(--font-body); font-weight: 400; color: var(--muted); margin-left: 12px; font-size: 13px; }
.sub { display: block; font-weight: 400; color: var(--muted); font-size: 12px; }
tr.mean th, tr.mean td { font-weight: 600; border-bottom: 2px solid var(--line); }
tr.note td { text-align: left; color: var(--muted); font-size: 12px; white-space: normal; }
td.g0 { background: var(--g0); } td.g1 { background: var(--g1); } td.g2 { background: var(--g2); } td.g3 { background: var(--g3); }
td.best { font-weight: 700; }
td.ours, th.ours { box-shadow: inset 2px 0 0 var(--accent), inset -2px 0 0 var(--accent); }
thead th.ours { color: var(--accent); }
td.oracle { color: var(--muted); }
td.na { color: var(--muted); text-align: center; }
.sd { color: var(--muted); font-size: 11px; margin-left: 3px; }
.mono { font-family: var(--font-mono); font-size: 12px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; border: 1px solid currentColor; }
.pill.pos { color: var(--pos); } .pill.neg { color: var(--neg); }
.ok { color: var(--pos); font-weight: 700; } .bad { color: var(--neg); font-weight: 700; }
td.rpos { color: var(--pos); } td.rneg { color: var(--neg); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 12px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid var(--line); }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; font-size: 13px; }
dt { font-weight: 600; font-family: var(--font-head); } dd { margin: 0; color: var(--muted); }
ul { margin: 0; padding-left: 20px; color: var(--muted); max-width: 80ch; }
@media (max-width: 600px) { dl { grid-template-columns: 1fr; } dd { margin-bottom: 6px; } }
"""


def main():
    acl, sel, spear, summary = read('acl'), read('selection'), read('spearman'), read('summary')
    legend = ('<div class="legend"><span><i style="background:var(--g0)"></i>within 1 pt of oracle</span>'
              '<span><i style="background:var(--g1)"></i>1–5 pts below</span>'
              '<span><i style="background:var(--g2)"></i>5–15 pts below</span>'
              '<span><i style="background:var(--g3)"></i>more than 15 pts below</span>'
              '<span><b>Bold</b> = best label-free method in the row</span>'
              '<span><i style="box-shadow: inset 2px 0 0 var(--accent), inset -2px 0 0 var(--accent)"></i>CrA and CrA-TV (ours)</span></div>')
    notes = ''.join(f'<dt>{m}</dt><dd>{esc(t)}</dd>' for m, t in NOTES.items())
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>CrA Results Tables</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;600&family=JetBrains+Mono&display=swap">
<style>{STYLE}</style>
<main>
<header>
  <h1>CrA results across datasets</h1>
  <p>Label-free model selection for domain generalization. CrA ranks DG models by their agreement with ERM on the
  unlabeled test environment. The ERM-referenced ACL estimate decides the direction: a negative estimated slope
  means pick the model least like ERM, a positive one means pick the model most like ERM.</p>
</header>

<section>
  <h2>1. Accuracy on the line: true vs estimated</h2>
  <p>True ACL: probit line of test accuracy against ID accuracy over all models (uses test labels).
  Estimated ACL: the same line with each model's test accuracy extrapolated along the disagreement-with-ERM axis c
  (no test labels). OOD estimate error is the mean absolute error of those per-model estimates.</p>
  <div class="scroll">{acl_table(acl)}</div>
  <p>Pooled selection: one model chosen from all DG seeds of a dataset (DG), or from every model including the ERM
  seeds (all). Each cell shows the true test accuracy (%) and the selected model.</p>
  <div class="scroll">{pooled_table(acl)}</div>
</section>

<section>
  <h2>2. Model selection, averaged over DG algorithms</h2>
  <p>True test accuracy (%) of the model each method selects, averaged over the DG algorithms of each dataset.</p>
  {legend}
  <div class="scroll">{summary_table(summary)}</div>
</section>

<section>
  <h2>3. Model selection per algorithm</h2>
  <p>True test accuracy (%) of the hyperparameter seed each method selects among an algorithm's 20 seeds,
  ± the std over that seed's 3 trials. Every method ranks a seed by its mean over the 3 trials.
  Hover a column header for the method's definition.</p>
  {legend}
  <div class="scroll">{selection_table(sel, acl, summary)}</div>
</section>

<section>
  <h2>4. Ranking quality per algorithm</h2>
  <p>Spearman correlation between each score and the true test accuracy over an algorithm's 20 seeds
  (uses test labels, shown for validation only). Green ≥ +0.7, red ≤ −0.7. A strongly negative CrA r means the seeds
  least like ERM are the best ones.</p>
  <div class="scroll">{spearman_table(spear, acl)}</div>
</section>

<section>
  <h2>Checks</h2>
  <p>Test labels are saved once per results folder. Every run's recorded test accuracy is recomputed from its saved
  predictions and these labels; if all runs reproduce it, all algorithms were scored on the same test images, which
  image-by-image agreement requires.</p>
  {checks_list(acl)}
</section>

<section>
  <h2>Methods</h2>
  <dl>{notes}</dl>
</section>

<section>
  <h2>Caveats</h2>
  <ul>
    <li>ColoredMNIST is the only well-specified benchmark; it uses per-trial splits with fixed, seeded data.</li>
    <li>RotatedMNIST, PACS and ACSIncome come from older single-split runs, before full determinism; to be re-run.</li>
    <li>PACS ResNet50 env1 and both PACS CLIP configurations contain only ERM and IRM.</li>
    <li>Label-free scores are computed on the 20% out split of the test environment; test accuracy is reported on
    the 80% in split.</li>
    <li>The ERM row uses leave-one-out CrA.</li>
  </ul>
</section>
</main>
"""
    with open('tables/full_comparison.html', 'w', encoding='utf-8') as f:
        f.write(page)
    print('saved tables/full_comparison.html')


if __name__ == '__main__':
    main()
