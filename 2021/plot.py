#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sn

import column_names as cols


file_formats = ["pdf", "svg"]


def save(name):
    if not os.path.isdir("figs"):
        os.mkdir("figs")
    for fmt in file_formats:
        plt.savefig(
            "figs/%s.%s" % (name, fmt),
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )


# entire community
df = pd.read_csv("data/spack-user-survey-2021-responses.csv")

# get reasonable column names
df.columns = [cols.description_to_name[c.strip()] for c in df.columns]


# just members of ECP
# ecp = df[df.in_ecp == "Yes"]
# ecp_col_name = "ECP"

# just RSEs
# ecp = df[df.user_type == "Research Software Engineer (RSE)"]
# ecp_col_name = "RSEs"

# just NNSA
ecp = df[df.workplace == "DOE/NNSA Lab (e.g., LLNL/LANL/SNL)"]
ecp_col_name = "DOE/NNSA"

#
# Are you part of ECP?
#
ax = df.in_ecp.value_counts().plot.pie(
    figsize=(6, 4),
    fontsize=12,
    autopct=lambda p: ("%.1f%%" % p) if p > 4 else "",
    explode=[0.05] * 2,
    ylabel="",
    legend=False,
    labels=[""] * 2,
    pctdistance=0.7,
    title=cols.names["in_ecp"],
    textprops={"color": "w"},
)
ax.legend(
    loc="lower left",
    fontsize=12,
    bbox_to_anchor=(-0.2, 0),
    frameon=False,
    labels=["All", ecp_col_name],
)
save("pie_in_ecp")


#
# Pie charts
#
def two_pies(col, legend_cols=2, same=False):
    """Plot two pie charts to compare all responses with ECP responses.

    Args:
        col (str): name of column to compare
        legend_cols (int): number of columns in the legend
        same (bool): whether ECP results were pretty much the same as all (in
            which case we omit the ECP-specific ones)

    """
    plt.close()
    combined = pd.DataFrame()
    combined["All"] = df[col].value_counts()
    if not same:
        combined[ecp_col_name] = ecp[col].value_counts()
    axes = combined.plot.pie(
        subplots=True,
        layout=(1, 2),
        figsize=(8, 8),
        fontsize=12,
        autopct=lambda p: ("%.1f%%" % p) if p > 4 else "",
        explode=[0.05] * len(combined),
        legend=False,
        labels=[""] * combined.shape[0],
        ylabel="",
        pctdistance=0.7,
        title=cols.names[col],
        textprops={"color": "w"},
    )

    plt.tight_layout()
    axes[0][0].set_title("All\n(ECP responses were similar)")
    if not same:
        axes[0][0].set_title("All")
        axes[0][1].set_title(ecp_col_name)
    axes[0][0].get_figure().subplots_adjust(top=1.3)

    axes[0][0].legend(
        ncol=legend_cols,
        bbox_to_anchor=(0, 0),
        loc="upper left",
        labels=combined.index,
        fontsize=12,
        frameon=False,
    )
    save("two_pies_" + col)


two_pies("user_type")
two_pies("languages")
two_pies("workplace")
two_pies("country", legend_cols=3)
two_pies("how_find_out")
two_pies("how_bad_only_py3", legend_cols=1)
two_pies("would_attend_workshop")
two_pies("did_tutorial")
two_pies("how_often_docs")
two_pies("commercial_support")
two_pies("software_foundation")

#
# Simple bar charts
#
def two_bars(col):
    """Plot two bar charts to compare all responses with ECP responses.

    Args:
        col (str): name of column to compare
    """
    plt.close()
    combined = pd.DataFrame()
    combined["All"] = df[col].value_counts(sort=False)
    combined[ecp_col_name] = ecp[col].value_counts(sort=False)
    axes = combined.plot.bar(
        subplots=True,
        layout=(1, 2),
        figsize=(8, 3),
        fontsize=12,
        legend=False,
        ylabel="",
        xlabel="at least N years",
        title=cols.names[col],
    )

    plt.tight_layout()
    axes[0][0].set_title("All")
    axes[0][1].set_title(ecp_col_name)

    save("two_bars_" + col)


# not pie charts
two_bars("how_long_using")

#
# Multi-choice bar charts
#
def two_multi_bars(col, sort=None, index=None, filt=None, name=None, figsize=(5, 4)):
    """Plot two bar charts to compare all responses with ECP responses.

    Args:
        col (str): name of column to compare
        index (list): custom index for plot
        filt (callable): optional function to filter column by
        name (str): name for the figure
        figsize (tuple): dimensions in inches for the figure
    """
    if filt is None:
        filt = lambda x: x
    if name is None:
        name = col

    plt.close()
    combined = pd.DataFrame(index=index)
    split = filt(df[col].str.split(",\s+", expand=True))
    combined["All"] = split.stack().value_counts()
    combined["All"] /= df.shape[0]
    combined["All"] *= 100

    split = filt(ecp[col].str.split(",\s+", expand=True))
    combined[ecp_col_name] = split.stack().value_counts()
    combined[ecp_col_name] /= ecp.shape[0]
    combined[ecp_col_name] *= 100

    if not index:
        combined = combined.sort_values(by="All", ascending=True)
    ax = combined.plot.barh(
        figsize=figsize,
        legend=True,
        title=cols.names[col],
    )
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    plt.xlabel("Percent of respondents")
    plt.tight_layout()
    save("two_multi_bars_" + name)


two_multi_bars("app_area", figsize=(5, 5))
two_multi_bars("how_contributed")

two_multi_bars(
    "spack_versions",
    filt=lambda df: df.replace("Not sure. ", "do not know").replace(
        "Do not know", "do not know"
    ),
)

two_multi_bars(
    "os",
    filt=lambda df: df.replace("Windows Subsystem for Linux (WSL)", "WSL"),
    figsize=(5, 7),
)

two_multi_bars("python_version", index=["2.6", "2.7", "3.5", "3.6", "3.7", "3.8"])
two_multi_bars(
    "how_use_pkgs",
    figsize=(6, 5),
    filt=lambda df: df.replace(["Environment Modules (TCL modules)"], "TCL Modules"),
)
two_multi_bars(
    "used_features",
    filt=lambda df: df.replace(r" \([^)]*\)", "", regex=True)
    .replace(
        "Concretization preferences in packages.yaml", "Concretization preferences"
    )
    .replace("Externals in packages.yaml", "External packages"),
    figsize=(6, 5),
)
two_multi_bars("local_repos")
two_multi_bars("local_mods")
two_multi_bars("cpus_next_year")
two_multi_bars("gpus_next_year")
two_multi_bars("compilers_next_year", figsize=(7, 4))
two_multi_bars("how_get_help")
two_multi_bars(
    "num_installations",
    index=reversed(
        ["1 - 10", "10 - 100", "100 - 200", "200 - 500", "500-1,000", "> 1,000"]
    ),
)

linuxes = [
    ".*Linux.*",
    "Gentoo.*",
    "Cray.*",
    "Amazon Linux.*",
    "Alpine.*",
    "TOSS.*",
    "Arch.*",
    "Fedora.*",
    ".*SuSE.*",
    "SLES.*",
    "Debian.*",
    "Ubuntu.*",
    "Red Hat.*",
    "PureOS.*",
    "RHEL.*",
    "CentOS.*",
    "Rocky.*",
    ".*Raspberry.*",
]


def linuxize(df):
    linux = df.replace(linuxes, "Linux", regex=True).replace(
        "Windows Subsystem for Linux (WSL)", "WSL"
    )
    is_duplicate = linux.apply(pd.Series.duplicated, axis=1)
    return linux.where(~is_duplicate, None)


two_multi_bars("os", filt=linuxize, name="os_simple")


def linux_type(df):
    linux = df.replace(
        {
            ".*SuSE.*": "SuSE",
            "SLES.*": "SuSE",
            "Cray.*": "SuSE",
            "Ubuntu.*": "Ubuntu",
            "Amazon Linux.*": "RHEL-based",
            "Red Hat.*": "RHEL-based",
            "TOSS.*": "RHEL-based",
            "RHEL.*": "RHEL-based",
            "CentOS.*": "RHEL-based",
            "Rocky.*": "RHEL-based",
            "Alpine.*": "Other Linux",
            "Arch.*": "Other Linux",
            "Fedora.*": "Other Linux",
            "Debian.*": "Other Linux",
            "PureOS.*": "Other Linux",
            "Gentoo.*": "Other Linux",
            ".*Linux.*": "Other Linux",
            ".*Raspberry.*": "Other Linux",
            "Windows Subsystem for Linux (WSL)": "WSL",
        },
        regex=True,
    )
    is_duplicate = linux.apply(pd.Series.duplicated, axis=1)
    return linux.where(~is_duplicate, None)


two_multi_bars("os", filt=linux_type, name="os_linux_types")


mods = ("Environment Modules (TCL modules)", "Lmod")


def modulize(df):
    """Add another column for "any module system"."""
    has_modules = df.apply(lambda ser: ser.isin(mods).any(), axis=1)
    mod_col = has_modules.apply(lambda c: "Modules (TCL or Lmod)" if c else None)
    frame = pd.concat([df, mod_col], axis=1)
    frame = frame.replace(["Environment Modules (TCL modules)"], "TCL Modules")
    return frame


two_multi_bars("how_use_pkgs", filt=modulize, name="how_use_pkgs_any", figsize=(6, 5))


gpus = ("NVIDIA", "AMD", "Intel")


def any_gpu(df):
    """Add another column for "any module system"."""
    has_gpu = df.apply(lambda ser: ser.isin(gpus).any(), axis=1)
    extra_col = has_gpu.apply(lambda c: "Any GPU" if c else None)
    frame = pd.concat([df, extra_col], axis=1)
    return frame


two_multi_bars("gpus_next_year", filt=any_gpu, name="gpus_next_year_any")

#
# Multi-choice bar charts
#
def feature_bar_chart(
    df,
    title,
    name,
    feature_cols,
    ratings,
    xlabels,
    figsize,
    rot=25,
    ha="right",
    ymax=None,
    colors=None,
):
    # value counts for all columns
    values = (
        df[feature_cols]
        .apply(pd.Series.value_counts, sort=False)
        .reindex(ratings)
        .transpose()
    )

    values /= len(df) / 100  # normalize to percentages

    ax = values.plot.bar(y=ratings, figsize=figsize, rot=0, color=colors)
    ax.legend(ncol=5, labels=ratings, frameon=False)
    plt.ylabel("Percent of respondents")
    plt.xticks(rotation=rot)
    if ymax:
        plt.ylim(0, ymax)
    if xlabels:
        ax.set_xticklabels(xlabels, ha=ha)
    plt.tight_layout()
    plt.title(title)
    save("feature_bars_" + name)


def score_averages(df, feature_cols, ratings, weights):
    """Calculate average scores for features

    Args:
        df (DataFrame): data set
        feature_cols (list): list of column names to average
        ratings (list): values from the feature cols associated w/weights,
            e.g. "bad", "ok", "good"
        weights (dict): weights associated with ratings, e.g.,
            {"bad": 0, "ok": 1, "good": 2}.
    """
    values = df[feature_cols].apply(pd.Series.value_counts).reindex(ratings)
    w = pd.Series(weights, index=ratings)
    return values.multiply(w, axis="index").sum() / values.sum()


def heat_map(
    title, filename, feature_cols, ratings, weights, labels, transpose, cmap, data_sets
):
    """Generate a heat ma of

    Args:
        title (str): title for figure
        filename (str): name for figure file
        feature_cols (list): list of column names to average
        ratings (list): values from the feature cols associated w/weights,
            e.g. "bad", "ok", "good"
        weights (dict): weights associated with ratings, e.g.,
            {"bad": 0, "ok": 1, "good": 2}.
        labels (list): labels for the features -- default is feature col names.
        transpose (bool): True for features on X axis, False for labels on Y.
        cmap (str): Name of colormap to use
        data_sets (dict str -> DataFrame): names for y axis of heat map,
            mapped to data frames to get stats from.
    """
    plt.close()
    plt.figure()
    heat_map = pd.DataFrame(
        {
            name: score_averages(frame, feature_cols, ratings, weights)
            for name, frame in data_sets.items()
        }
    )
    if transpose:
        heat_map = heat_map.transpose()

    # sort from highest to loweset rated
    heat_map = heat_map.sort_values(by=heat_map.columns[0], ascending=False)

    # order labels by value sort
    if not transpose:
        feature_labels = dict(zip(feature_cols, labels))
        labels = [feature_labels[col] for col in heat_map.index]

    ax = sn.heatmap(
        heat_map,
        cmap=cmap,
        annot=True,
        vmin=0,
        vmax=4,
        square=True,
        fmt=".1f",
        annot_kws={"size": 6},
    )

    cbar = ax.collections[0].colorbar
    cbar.set_ticks(range(5))
    cbar.set_ticklabels(
        ["%d - %s" % (i, s.replace(" ", "\n     ")) for i, s in enumerate(ratings)]
    )
    cbar.ax.tick_params(labelsize=6)

    plt.title(title + "\n", fontsize=11)
    plt.xticks(rotation=45)
    if transpose:
        ax.set_xticklabels(labels, ha="right")
    else:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(data_sets.keys(), ha="right")

    ax.tick_params(axis="both", which="major", labelsize=6)

    plt.tight_layout()
    save("heat_map_" + filename)


ratings = [
    "Not Important",
    "Slightly Important",
    "Somewhat important",
    "Very Important",
    "Critical",
]
weights = {r: i for i, r in enumerate(ratings)}


feature_cols = [
    "feature_better_error_messages",
    "feature_compiler_deps",
    "feature_separate_build_deps",
    "feature_optimized_binaries",
    "feature_build_testing",
    "feature_language_virtuals",
    "feature_testing",
    "feature_better_flag_handling",
    "feature_windows",
    "feature_external_autodetect",
    "feature_performance",
    "feature_env_reproducibility",
    "feature_gpu_compiler_support",
    "feature_github_caches",
    "feature_developer_support",
]

xlabels = [
    "Better error messages",
    "Compilers as dependencies",
    "Separate build-deps",
    "Optimized binaries",
    "Build testing (CI)",
    "Language virtuals",
    "ReFrame/Pavilion integration",
    "Better flag handling",
    "Windows support",
    "Autodetect more externals",
    "Peformance improvement",
    "Better env reproducibility",
    "Better GPU compiler support",
    "GitHub actions build caches",
    "Better dev support",
]

plt.close()


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap("CustomMap", cdict)


c = mcolors.ColorConverter().to_rgb
feature_cmap = make_colormap([c("#e9e9fc"), c("#0000ad")])

feature_bar_colors = [feature_cmap(v) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]


feature_bar_chart(
    df,
    "Rank these upcoming Spack features by importance",
    "all_features",
    feature_cols,
    ratings,
    xlabels,
    figsize=(12, 3),
    colors=feature_bar_colors,
)
# feature_bar_chart(
#    ecp, "Rank these upcoming Spack features by importance (ECP)",
#    "ecp_features", feature_cols, ratings, xlabels, figsize=(12, 3),
#    colors=feature_bar_colors)


sectors = {
    "All": df,
    "ECP": df[df.in_ecp == "Yes"],
    "NNSA": df[df.workplace == "DOE/NNSA Lab (e.g., LLNL/LANL/SNL)"],
    "ASCR": df[df.workplace == "DOE/Office of Science Lab (e.g., ORNL/ANL/LBL)"],
    "Industry": df[(df.workplace == "Company") | (df.workplace == "Cloud Provider")],
    "University HPC Center": df[df.workplace == "University HPC/Computing Center"],
    "University Research Group": df[df.workplace == "University research group"],
    "Public Lab": df[df.workplace == "Other Public Research Lab"],
    "Private Lab": df[df.workplace == "Private Research Lab"],
}


roles = {
    "All": df,
    "Developer": df[(df.user_type == "Software Developer")],
    "RSE": df[(df.user_type == "Research Software Engineer (RSE)")],
    "Computational Scientist": df[(df.user_type == "Computational Scientist")],
    "Code user/analyst": df[(df.user_type == "Code user/analyst")],
    "Scientist": df[(df.user_type == "Scientist/Researcher")],
    "Sys Admin": df[(df.user_type == "System Administrator")],
    "User Support": df[(df.user_type == "User Support Staff")],
    "Manager": df[(df.user_type == "Manager")],
}


heat_map(
    "Average feature importance by workplace",
    "features_by_workplace",
    feature_cols,
    ratings,
    weights,
    xlabels,
    False,
    feature_cmap,
    sectors,
)

heat_map(
    "Average feature importance by job type",
    "features_by_job",
    feature_cols,
    ratings,
    weights,
    xlabels,
    False,
    feature_cmap,
    roles,
)

#
# Quality ratings
#
ratings = ["Horrible", "Bad", "OK", "Good", "Excellent"]
weights = {r: i for i, r in enumerate(ratings)}

feature_cols = [
    "quality_spack",
    "quality_community",
    "quality_docs",
    "quality_packages",
]

xlabels = ["Spack", "Community", "Docs", "Packages"]


quality_cmap = "RdYlGn"
# These colors match to the Red/Yellow/Green heat maps
bar_colors = ["#cc2222", "orange", "#dddd00", "#94c772", "green"]

plt.close()
feature_bar_chart(
    df,
    "Rate the overall quality of...",
    "all_quality",
    feature_cols,
    ratings,
    xlabels,
    figsize=(7, 2),
    rot=0,
    ha="center",
    ymax=60,
    colors=bar_colors,
)
# feature_bar_chart(ecp, "Rate the overall quality of... (ECP)",
#                  "ecp_quality", feature_cols, ratings, xlabels,
#                  figsize=(7, 2), rot=0, ha="center", ymax=60, colors=bar_colors)

heat_map(
    "Average quality rating by workplace",
    "quality_by_workplace",
    feature_cols,
    ratings,
    weights,
    xlabels,
    True,
    quality_cmap,
    sectors,
)

heat_map(
    "Average quality rating by job type",
    "quality_by_job",
    feature_cols,
    ratings,
    weights,
    xlabels,
    True,
    quality_cmap,
    roles,
)
