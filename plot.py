#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
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
            bbox_inches='tight',
            pad_inches=0,
            transparent=True,
        )

# entire community
df = pd.read_csv("data/spack-user-survey-2020-responses.csv")

# get reasonable column names
df.columns = [cols.description_to_name[c.strip()] for c in df.columns]

# just members of ECP
ecp = df[df.in_ecp == "Yes"]

#
# Are you part of ECP?
#
ax = df.in_ecp.value_counts().plot.pie(
    figsize=(6,4),
    fontsize=12,
    autopct=lambda p: ("%.1f%%" % p) if p > 4 else "",
    explode=[0.05] * 2,
    ylabel='',
    pctdistance=0.7,
    title=cols.names["in_ecp"],
    textprops={'color':"w"}
)
ax.legend(loc="lower left", fontsize=12, bbox_to_anchor=(-.2, 0),
          frameon=False)
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
        combined["ECP"] = ecp[col].value_counts()
    axes = combined.plot.pie(
        subplots=True,
        layout=(1, 2),
        figsize=(8, 8),
        fontsize=12,
        autopct=lambda p: ("%.1f%%" % p) if p > 4 else "",
        explode=[0.05] * len(combined),
        legend=False,
        labels=[''] * combined.shape[0],
        ylabel='',
        pctdistance=0.7,
        title=cols.names[col],
        textprops={'color':"w"}
    )

    plt.tight_layout()
    axes[0][0].set_title("All\n(ECP responses were similar)")
    if not same:
        axes[0][0].set_title("All")
        axes[0][1].set_title("ECP")
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
two_pies("workplace")
two_pies("country", legend_cols=3)
two_pies("how_find_out")
two_pies("how_bad_no_py26", legend_cols=1)
two_pies("how_bad_only_py3", legend_cols=1)
two_pies("would_attend_workshop")
two_pies("did_tutorial")
two_pies("how_often_docs")
two_pies("commercial_support")

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
    combined["ECP"] = ecp[col].value_counts(sort=False)
    axes = combined.plot.bar(
        subplots=True,
        layout=(1, 2),
        figsize=(8, 3),
        fontsize=12,
        legend=False,
        ylabel='',
        xlabel="at least N years",
        title=cols.names[col],
    )

    plt.tight_layout()
    axes[0][0].set_title("All")
    axes[0][1].set_title("ECP")

    save("two_bars_" + col)

# not pie charts
two_bars("how_long_using")

#
# Multi-choice bar charts
#
def two_multi_bars(col, sort=None, index=None, filt=None, name=None,
                   figsize=(5, 4)):
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
    split = filt(df[col].str.split(',\s+', expand=True))
    combined["All"] = split.stack().value_counts()
    combined["All"] /= df.shape[0]
    combined["All"] *= 100

    split = filt(ecp[col].str.split(',\s+', expand=True))
    combined["ECP"] = split.stack().value_counts()
    combined["ECP"] /= ecp.shape[0]
    combined["ECP"] *= 100

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
two_multi_bars("spack_versions")
two_multi_bars("os", filt=lambda df: df.replace(
    "Windows Subsystem for Linux (WSL)", "WSL"))
two_multi_bars("python_version",
               index=['2.6', '2.7', '3.5', '3.6', '3.7', '3.8'])
two_multi_bars("how_use_pkgs", figsize=(6, 5), filt=lambda df: df.replace(
    ["Environment Modules (TCL modules)"], "TCL Modules"))
two_multi_bars(
    "used_features",
    filt=lambda df: df.replace(r' \([^)]*\)', '', regex=True).replace(
        "Concretization preferences in packages.yaml",
        "Concretization preferences"
    ).replace("Externals in packages.yaml", "External packages"),
    figsize=(6, 5))
two_multi_bars("cpus_next_year")
two_multi_bars("gpus_next_year")
two_multi_bars("compilers_next_year", figsize=(7, 4))
two_multi_bars("how_get_help")
two_multi_bars(
    "num_installations", index=reversed([
        "1 - 10", "10 - 100", "100 - 200", "200 - 500", "500-1,000", "> 1,000"
    ]))

linuxes = [
    "Gentoo", "Cray", "Amazon Linux", "Alpine", "TOSS", "Arch",
    "Fedora", "SuSE", "Debian", "Ubuntu", "Red Hat", "CentOS",
]

def linuxize(df):
    linux = df.replace(linuxes, "Linux").replace(
        "Windows Subsystem for Linux (WSL)", "WSL")
    is_duplicate = linux.apply(pd.Series.duplicated, axis=1)
    return linux.where(~is_duplicate, None)

two_multi_bars("os", filt=linuxize, name="os_simple")


mods = ("Environment Modules (TCL modules)", "Lmod")
def modulize(df):
    """Add another column for "any module system"."""
    has_modules = df.apply(lambda ser: ser.isin(mods).any(), axis=1)
    mod_col = has_modules.apply(lambda c: "Modules (TCL or Lmod)" if c else None)
    frame = pd.concat([df, mod_col], axis=1)
    frame = frame.replace(["Environment Modules (TCL modules)"], "TCL Modules")
    return frame


two_multi_bars("how_use_pkgs", filt=modulize, name="how_use_pkgs_any",
               figsize=(6, 5))

#
# Multi-choice bar charts
#
def feature_bar_chart(
        df, title, name, feature_cols, ratings, xlabels, figsize, rot=25,
        ha="right", ymax=None, colors=None):
    # value counts for all columns
    values = df[feature_cols].apply(
        pd.Series.value_counts, sort=False).reindex(ratings).transpose()

    ax = values.plot.bar(y=ratings, figsize=figsize, rot=0, color=colors)
    ax.legend(ncol=5, labels=ratings, frameon=False)
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
        title, filename, feature_cols, ratings, weights, labels, transpose,
        data_sets):
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
        data_sets (dict str -> DataFrame): names for y axis of heat map,
            mapped to data frames to get stats from.
    """
    plt.close()
    plt.figure()
    heat_map = pd.DataFrame({
        name: score_averages(frame, feature_cols, ratings, weights)
        for name, frame in data_sets.items()
    })
    if transpose:
        heat_map = heat_map.transpose()

    # sort from highest to loweset rated
    heat_map = heat_map.sort_values(by=heat_map.columns[0], ascending=False)

    # order labels by value sort
    if not transpose:
        feature_labels = dict(zip(feature_cols, labels))
        labels = [feature_labels[col] for col in heat_map.index]

    ax = sn.heatmap(
        heat_map, cmap="RdYlGn", annot=True, vmin=0, vmax=4, square=True,
        fmt=".1f", annot_kws={"size": 9})

    cbar = ax.collections[0].colorbar
    cbar.set_ticks(range(5))
    cbar.set_ticklabels([
        "%d - %s" % (i, s.replace(' ', '\n     '))
        for i, s in enumerate(ratings)
    ])
    cbar.ax.tick_params(labelsize=9)

    plt.title(title + "\n", fontsize=11)
    plt.xticks(rotation=45)
    if transpose:
        ax.set_xticklabels(labels, ha="right")
    else:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(data_sets.keys(), ha="right")

    ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    save("heat_map_" + filename)

# These colors match to the Red/Yellow/Green heat maps
colors = ["#cc2222", "orange", "#dddd00", "#94c772", "green"]

ratings = [
    "Not Important",
    "Slightly Important",
    "Somewhat important",
    "Very Important",
    "Critical"
]
weights = { r: i for i, r in enumerate(ratings) }


feature_cols = [
    'feature_use_existing_installs',
    'feature_new_concretizer',
    'feature_better_flag_handling',
    'feature_developer_support',
    'feature_separate_build_deps',
    'feature_language_virtuals',
    'feature_pkg_notifications',
    'feature_build_testing',
    'feature_optimized_binaries',
    'feature_testing',
    'feature_cloud_integration',
    'feature_windows',
]

xlabels = [
    "Reuse existing installs",
    "New concretizer",
    "Better flag handling",
    "Better dev support",
    "Separate build-deps",
    "Language virtuals",
    "Pkg maintainer notif.",
    "Build testing (CI)",
    "Optimized binaries",
    "Package testing",
    "Cloud integration",
    "Windows support",
]

plt.close()
feature_bar_chart(
    df, "Rank these upcoming Spack features by importance",
    "all_features", feature_cols, ratings, xlabels, figsize=(12, 3),
    colors=colors)
feature_bar_chart(
    ecp, "Rank these upcoming Spack features by importance (ECP)",
    "ecp_features", feature_cols, ratings, xlabels, figsize=(12, 3),
    colors=colors)

heat_map(
    "Average feature importance by workplace",
    "features_by_workplace",
    feature_cols, ratings, weights, xlabels, False, {
        "All"        : df,
        "ECP"        : df[df.in_ecp == "Yes"],
        "NNSA"       : df[df.workplace == "DOE/NNSA Lab (e.g., LLNL/LANL/SNL)"],
        "ASCR"       : df[df.workplace == "DOE/Office of Science Lab (ORNL/ANL/LBL)"],
        "Industry"   : df[(df.workplace == "Company")
                          | (df.workplace == "Cloud Provider")],
        "University" : df[df.workplace == "University HPC/Computing Center"],
        "Public Lab" : df[df.workplace == "Other Public Research Lab"],
    }
)

heat_map(
    "Average feature importance by job type",
    "features_by_job",
    feature_cols, ratings, weights, xlabels, False, {
        "All"                  : df,
        "Developer"    : df[(df.user_type == "Software Developer")
                            | (df.user_type == "All of the Above")],
        "Scientist"    : df[(df.user_type == "Scientist/Researcher")
                            | (df.user_type == "All of the Above")],
        "Sys Admin"    : df[(df.user_type == "System Administrator")
                            | (df.user_type == "All of the Above")],
        "User Support" : df[(df.user_type == "User Support Staff")
                            | (df.user_type == "All of the Above")],
        "Manager"      : df[(df.user_type == "Manager")
                            | (df.user_type == "All of the Above")],
    }
)

#
# Quality ratings
#
ratings = ["Horrible", "Bad", "OK", "Good", "Excellent"]
weights = { r: i for i, r in enumerate(ratings) }

feature_cols = [
    "quality_spack",
    "quality_community",
    "quality_docs",
    "quality_packages",
]

xlabels = ["Spack", "Community", "Docs", "Packages"]

plt.close()
feature_bar_chart(df, "Rate the overall quality of...",
                  "all_quality", feature_cols, ratings, xlabels,
                  figsize=(7, 2), rot=0, ha="center", ymax=110, colors=colors)
feature_bar_chart(ecp, "Rate the overall quality of... (ECP)",
                  "ecp_quality", feature_cols, ratings, xlabels,
                  figsize=(7, 2), rot=0, ha="center", ymax=40, colors=colors)

heat_map(
    "Average quality rating by workplace",
    "quality_by_workplace",
    feature_cols, ratings, weights, xlabels, True, {
        "All"        : df,
        "ECP"        : df[df.in_ecp == "Yes"],
        "NNSA"       : df[df.workplace == "DOE/NNSA Lab (e.g., LLNL/LANL/SNL)"],
        "ASCR"       : df[df.workplace == "DOE/Office of Science Lab (ORNL/ANL/LBL)"],
        "Industry"   : df[(df.workplace == "Company")
                          | (df.workplace == "Cloud Provider")],
        "University" : df[df.workplace == "University HPC/Computing Center"],
        "Public Lab" : df[df.workplace == "Other Public Research Lab"],
    }
)

heat_map(
    "Average quality rating by job type",
    "quality_by_job",
    feature_cols, ratings, weights, xlabels, True, {
        "All"          : df,
        "Developer"    : df[(df.user_type == "Software Developer")
                            | (df.user_type == "All of the Above")],
        "Scientist"    : df[(df.user_type == "Scientist/Researcher")
                            | (df.user_type == "All of the Above")],
        "Sys Admin"    : df[(df.user_type == "System Administrator")
                            | (df.user_type == "All of the Above")],
        "User Support" : df[(df.user_type == "User Support Staff")
                            | (df.user_type == "All of the Above")],
        "Manager"      : df[(df.user_type == "Manager")
                            | (df.user_type == "All of the Above")],
    }
)
