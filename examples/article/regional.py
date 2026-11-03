#!/usr/bin/env python
"""Article section 5 -- regional organization of the parameters.

  Group maps + network table (tables/regional_group_maps.csv,
  regional_network_table.csv, fig5_networks)
      per region: median over subjects of every parameter; per functional
      group: median of its regions, contrast vs all other live regions (z),
      Kruskal-Wallis across groups.
  Reproducibility (tables/regional_reliability.csv, fig5_networks b)
      split-half reliability of the group map vs mean pairwise
      subject-map similarity -- "group maps are reproducible, single-subject
      maps much less so".
  External maps (``--external-maps``; tables/regional_external.csv, fig5_external)
      cortical group maps vs T1w/T2w myelin, the principal FC gradient and
      GABA-A / NMDA receptor density (neuromaps), Spearman r with a spin
      test (Alexander-Bloch nulls on fsLR-32k, Schaefer-100 17-network
      CIFTI from the CBIG repository; volumetric maps projected with
      regfusion, so no Connectome Workbench is needed). Needs ``neuromaps``
      (``pixi add --pypi neuromaps``, or PYTHONPATH to an unpacked wheel) and
      network access the first time (downloads cached under --cache).
"""
from __future__ import annotations

import argparse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import C_FIT, DEFAULT_BIDS, DEFAULT_ROOT, INK_2, NEUTRAL_C, Out, panel, plt, style

EXTERNAL = [  # (label, source, desc, space, density)
    ("T1w/T2w myelin", "hcps1200", "myelinmap", "fsLR", "32k"),
    ("principal FC gradient", "margulies2016", "fcgradient01", "fsLR", "32k"),
    ("GABA-A (flumazenil)", "norgaard2021", "flumazenil", "MNI152", "1mm"),
    ("NMDA (GE-179)", "galovic2021", "ge179", "MNI152", "1mm"),
]
CBIG = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
        "Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_100Parcels_17Networks_order.dlabel.nii")


def schaefer_fslr(cache: Path):
    """Schaefer-100 (17 networks) as a (left, right) pair of fsLR-32k label
    GIFTIs + the parcel names in label order (1..100, LH then RH) -- from
    the CBIG CIFTI, so nothing needs Connectome Workbench."""
    import nibabel as nib
    f = cache / Path(CBIG).name
    if not f.exists():
        urllib.request.urlretrieve(CBIG, f)
    img = nib.load(str(f))
    data = np.asarray(img.get_fdata()[0], dtype=np.int32)
    table = img.header.get_axis(0).label[0]
    giis = []
    for structure in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"):
        for name, slc, bm in img.header.get_axis(1).iter_structures():
            if name == structure:
                arr = np.zeros(bm.nvertices[name], dtype=np.int32)
                arr[bm.vertex] = data[slc]
        lt = nib.gifti.GiftiLabelTable()
        for key in sorted(set(arr.tolist())):
            lab = nib.gifti.GiftiLabel(key=int(key))
            lab.label = table[key][0]
            lt.labels.append(lab)
        giis.append(nib.gifti.GiftiImage(labeltable=lt, darrays=[nib.gifti.GiftiDataArray(
            arr, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")]))
    names = [table[k][0] for k in sorted(k for k in table if k != 0)]
    return tuple(giis), names


def group_tables(pop, params, frozen_cells, args):
    from params_report_cli import GROUP_ORDER, LIVE_ON
    from population_params_report_cli import reliability
    rng = np.random.default_rng(args.seed)
    regions = pop["regions"]
    maps, net, rel = [], [], []
    for p in params:
        live = [r["idx"] for r in regions if r["is_surface"] == (LIVE_ON[p] == "surface")
                and np.isfinite(pop["V"][p][:, r["idx"]]).mean() >= args.min_coverage
                and r["idx"] not in frozen_cells.get(p, [])]
        M = pop["V"][p][:, live]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(M, axis=0)
        for r, v in zip(live, med):
            maps.append({"param": p, "region": regions[r]["name"], "region_idx": r,
                         "group": regions[r]["group"], "hemi": regions[r]["hemi"], "median": v,
                         "iqr": float(np.subtract(*np.nanpercentile(M[:, live.index(r)], [75, 25])))})
        groups = [regions[r]["group"] for r in live]
        mu, sd = np.mean(med), np.std(med)
        samples = [med[[g == G for g in groups]] for G in dict.fromkeys(groups)]
        kw = stats.kruskal(*[s for s in samples if len(s) > 1]) if sum(len(s) > 1 for s in samples) > 1 else None
        for G in GROUP_ORDER:
            sel = np.array([g == G for g in groups])
            if not sel.any():
                continue
            other = med[~sel]
            net.append({"param": p, "group": G, "n_regions": int(sel.sum()), "median": float(np.median(med[sel])),
                        "z_vs_other": float((med[sel].mean() - other.mean()) / sd) if sd else np.nan,
                        "p_mw_vs_other": float(stats.mannwhitneyu(med[sel], other)[1]) if sel.sum() > 1 and len(other) > 1 else np.nan,
                        "kruskal_H": kw.statistic if kw else np.nan, "kruskal_p": kw.pvalue if kw else np.nan})
        rel.append({"param": p, "n_regions": len(live), **reliability(M, rng)})
    net = pd.DataFrame(net)
    from population_params_report_cli import _fdr
    net["q_mw_vs_other"] = _fdr(net.p_mw_vs_other)
    return pd.DataFrame(maps), net, pd.DataFrame(rel)


def fig_networks(net: pd.DataFrame, rel: pd.DataFrame, out: Out) -> None:
    from population_params_report_cli import DIV_CMAP
    piv = net.pivot(index="group", columns="param", values="z_vs_other")
    order = [g for g in dict.fromkeys(net.group)]
    piv = piv.loc[order, list(dict.fromkeys(net.param))]
    sig = net.pivot(index="group", columns="param", values="q_mw_vs_other").loc[piv.index, piv.columns] < 0.05
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    lim = np.nanmax(np.abs(piv.to_numpy()))
    im = ax.imshow(piv.to_numpy(), cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.1f}" + ("*" if sig.iloc[i, j] else ""), ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.6 * lim else INK_2)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="network vs other regions (z)")
    ax.set_title("group maps by network (* FDR q < .05)")
    panel(ax, "a")
    ax = axs[1]
    x = np.arange(len(rel))
    ax.bar(x - 0.18, rel.split_half_r, width=0.34, color=C_FIT, label="group map, split-half r")
    ax.bar(x + 0.18, rel.pairwise_subject_r, width=0.34, color=NEUTRAL_C, label="subject vs subject r")
    ax.set_xticks(x, rel.param)
    ax.set_ylim(min(0, rel.pairwise_subject_r.min() - 0.05), 1)
    ax.set_ylabel("Spearman r across regions")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    ax.set_title("reproducibility")
    panel(ax, "b")
    out.fig(fig, "fig5_networks")


def external(maps: pd.DataFrame, params: list[str], cache: Path, n_perm: int, seed: int) -> pd.DataFrame:
    from neuromaps import datasets, nulls, transforms
    from neuromaps import stats as nstats
    from neuromaps.parcellate import Parcellater
    cache.mkdir(parents=True, exist_ok=True)
    parc, names = schaefer_fslr(cache)
    parcellater = Parcellater(parc, "fsLR")
    ext = {}
    for label, src, desc, space, den in EXTERNAL:
        img = datasets.fetch_annotation(source=src, desc=desc, space=space, den=den, data_dir=str(cache))
        if space != "fsLR":
            img = transforms.mni152_to_fslr(img, "32k")
        ext[label] = np.asarray(parcellater.fit_transform(img, "fsLR"), float).ravel()
        assert len(ext[label]) == len(names), f"{label}: {len(ext[label])} parcels"
    rows = []
    for p in params:
        m = maps[maps.param == p].set_index("region")["median"]
        x = np.array([m.get(n, np.nan) for n in names])
        if np.isfinite(x).sum() < 0.9 * len(x):
            continue
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        null = nulls.alexander_bloch(x, atlas="fsLR", density="32k", parcellation=parc, n_perm=n_perm, seed=seed)
        for label, y in ext.items():
            r, pv = nstats.compare_images(x, y, nulls=null, metric="spearmanr")
            rows.append({"param": p, "map": label, "spearman_r": float(r), "p_spin": float(pv)})
    df = pd.DataFrame(rows)
    from population_params_report_cli import _fdr
    df["q_spin"] = _fdr(df.p_spin)
    return df


def fig_external(df: pd.DataFrame, out: Out) -> None:
    from population_params_report_cli import DIV_CMAP
    piv = df.pivot(index="param", columns="map", values="spearman_r")
    q = df.pivot(index="param", columns="map", values="q_spin").loc[piv.index, piv.columns]
    fig, ax = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    im = ax.imshow(piv.to_numpy(), cmap=DIV_CMAP, vmin=-0.7, vmax=0.7, aspect="auto")
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        ax.text(j, i, f"{v:+.2f}" + ("*" if q.iloc[i, j] < 0.05 else ""), ha="center", va="center", fontsize=6, color=INK_2)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=20, ha="right", fontsize=6)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r (spin-tested; * FDR q < .05)")
    out.fig(fig, "fig5_external")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/article")
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--external-maps", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path.home() / "neuromaps-data")
    ap.add_argument("--n-spin", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Out(args.out or args.root / "article")
    style()
    from population_params_report_cli import find_frozen, load_population
    pop = load_population(args.root, args.bids, args.bids / "participants.tsv")
    frozen, frozen_cells = find_frozen(pop)
    params = [p for p in pop["params"] if p not in frozen]
    maps, net, rel = group_tables(pop, params, frozen_cells, args)
    out.table(maps, "regional_group_maps")
    out.table(net, "regional_network_table")
    out.table(rel, "regional_reliability")
    fig_networks(net, rel, out)
    num = {"reliability": rel.set_index("param").to_dict(orient="index"),
           "top_networks": {p: g.sort_values("z_vs_other", ascending=False).head(3)[["group", "z_vs_other", "q_mw_vs_other"]]
                            .to_dict(orient="records") for p, g in net.groupby("param")},
           "kruskal": net.groupby("param")[["kruskal_H", "kruskal_p"]].first().to_dict(orient="index")}
    if args.external_maps:
        from params_report_cli import LIVE_ON
        ext = external(maps, [p for p in params if LIVE_ON[p] == "surface"], args.cache, args.n_spin, args.seed)
        out.table(ext, "regional_external")
        fig_external(ext, out)
        num["external"] = ext.to_dict(orient="records")
    out.numbers("section5_regional", num)


if __name__ == "__main__":
    main()
