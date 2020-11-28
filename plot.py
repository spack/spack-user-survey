#!/usr/bin/env python3

import sys

import pandas as pd
import matplotlib.pyplot as plt

import column_names as cols

# entire community
df = pd.read_csv("data/spack-user-survey-2020-responses.csv")

# get reasonable column names
df.columns = [cols.description_to_name[c.strip()] for c in df.columns]

# just members of ECP
ecp = df[df.in_ecp == "Yes"]

# only ECP
# df = df[""]

# make an arbitrary-length colormap
#cm = plt.get_cmap('rainbow')
#c = [cm(1.0 * i/len(df_thres)) for i in range(len(df_thres))]
#clist2 = {i:j for i, j in zip(df_thres[0].values, c)}

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
plt.savefig("in_ecp.pdf")


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
    plt.savefig("%s.pdf" % col)

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


# not pie charts
#two_pies("how_long_using")
#two_pies("app_area")
#two_pies("how_contributed")




#two_pies("spack_versions")
#two_pies("os")
#two_pies("python_version")
#two_pies("how_use_pkgs")
#two_pies("used_features")
#two_pies("cpus_next_year")
#two_pies("gpus_next_year")
#two_pies("compilers_next_year")
#two_pies("how_get_help")


"feature_importance_concretizer"
"feature_use_existing_installs"
"feature_separate_build_deps"
"feature_cloud_integration"
"feature_optimized_binaries"
"feature_developer_support"
"feature_pkg_notifications"
"feature_build_testing"
"feature_language_virtuals"
"feature_testing"
"feature_better_flag_handling"
"feature_windows"
"feature_custom"

"quality_docs"
"quality_community"
"quality_packages"
"quality_spack"
