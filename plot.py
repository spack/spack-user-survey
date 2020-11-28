#!/usr/bin/env python3

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

import column_names as cols

if not os.path.isdir("figs"):
    os.mkdir("figs")

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
ax.legend(loc="lower left", fontsize=12, bbox_to_anchor=(-.2, 0))
plt.savefig("figs/in_ecp.pdf")


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
    )
    plt.savefig("figs/%s.pdf" % col)

two_pies("user_type")
two_pies("workplace")
two_pies("country", legend_cols=3)
two_pies("how_find_out")
two_pies("how_bad_no_py26", legend_cols=1, same=True)
two_pies("how_bad_only_py3", legend_cols=1, same=True)
two_pies("would_attend_workshop", same=True)
two_pies("did_tutorial")
two_pies("how_often_docs")
two_pies("commercial_support", same=True)



#
# Simple bar charts
#
def two_bars(col, same=False):
    """Plot two bar charts to compare all responses with ECP responses.

    Args:
        col (str): name of column to compare
        same (bool): whether ECP results were pretty much the same as all (in
            which case we omit the ECP-specific ones)

    """
    plt.close()
    combined = pd.DataFrame()
    combined["All"] = df[col].value_counts(sort=False)
    combined["ECP"] = ecp[col].value_counts(sort=False)
    axes = combined.plot.bar(
        subplots=True,
        layout=(1, 2),
        figsize=(8, 4),
        fontsize=12,
        legend=False,
        ylabel='',
        xlabel="at least N years",
        title=cols.names[col],
    )

    plt.tight_layout()
    axes[0][0].set_title("All\n(ECP responses were similar)")
    if not same:
        axes[0][0].set_title("All")
        axes[0][1].set_title("ECP")

    plt.savefig("figs/%s.pdf" % col)

# not pie charts
two_bars("how_long_using")

#
# Multi-choice bar charts
#
def two_multi_bars(col, sort=None, index=None):
    """Plot two bar charts to compare all responses with ECP responses.

    Args:
        col (str): name of column to compare
        index (list): custom index for plot
    """
    plt.close()
    combined = pd.DataFrame(index=index)
    combined["All"] = df[
        col].str.split(',\s+', expand=True).stack().value_counts()
    combined["All"] /= df.shape[0]
    combined["All"] *= 100

    combined["ECP"] = ecp[
        col].str.split(',\s+', expand=True).stack().value_counts()
    combined["ECP"] /= ecp.shape[0]
    combined["ECP"] *= 100

    axes = combined.plot.barh(
        figsize=(8, 4),
        fontsize=12,
        legend=True,
        title=cols.names[col],
    )

    plt.tight_layout()
    plt.savefig("figs/%s.pdf" % col)

two_multi_bars("app_area")
two_multi_bars("how_contributed")
two_multi_bars("spack_versions")
two_multi_bars("os")
two_multi_bars("python_version",
               index=reversed(['2.6', '2.7', '3.5', '3.6', '3.7', '3.8']))
two_multi_bars("how_use_pkgs")
two_multi_bars("used_features")
two_multi_bars("cpus_next_year")
two_multi_bars("gpus_next_year")
two_multi_bars("compilers_next_year")
two_multi_bars("how_get_help")


#
# Multi-choice bar charts
#
def clustered_bars(cols):
    """Plot two clustered bar charts comparing all responses with ECP.

    Args:
        col (str): name of column to compare
        index (list): custom index for plot
    """
    plt.close()

    combined = pd.DataFrame()
    combined["All"] = df[
        col].str.split(',\s+', expand=True).stack().value_counts()
    combined["All"] /= df.shape[0]
    combined["All"] *= 100

    combined["ECP"] = ecp[
        col].str.split(',\s+', expand=True).stack().value_counts()
    combined["ECP"] /= ecp.shape[0]
    combined["ECP"] *= 100

    axes = combined.plot.barh(
        figsize=(8, 4),
        fontsize=12,
        legend=True,
        title=cols.names[col],
    )

    plt.tight_layout()
    plt.savefig("figs/%s.pdf" % col)


ratings = [
    "Not Important",
    "Slightly Important",
    "Somewhat important",
    "Very Important",
    "Critical"
]
weights = { r: i for i, r in enumerate(ratings) }


plt.close()
feature_cols = [
    "feature_importance_concretizer",
    "feature_use_existing_installs",
    "feature_separate_build_deps",
    "feature_cloud_integration",
    "feature_optimized_binaries",
    "feature_developer_support",
    "feature_pkg_notifications",
    "feature_build_testing",
    "feature_language_virtuals",
    "feature_testing",
    "feature_better_flag_handling",
    "feature_windows",
]

def feature_bar_chart(df, name):
    # value counts for all columns
    values = df[feature_cols].apply(
        pd.Series.value_counts, sort=False).reindex(ratings).transpose()

    ax = values.plot.bar(y=ratings, figsize=(12, 4))
    ax.legend(ncol=5, labels=ratings)
    plt.tight_layout()
    plt.savefig("figs/%s.pdf" % name)

feature_bar_chart(df, "all_features")
feature_bar_chart(ecp, "ecp_features")

weights = {
    "Horrible": 0,
    "Bad": 1,
    "OK": 2,
    "Good": 3,
    "Excellent": 4,
}

"quality_docs"
"quality_community"
"quality_packages"
"quality_spack"
