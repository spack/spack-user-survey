#: Short to long names for columns in survey
names = {
    "timestamp"                      : "Timestamp",
    "user_type"                      : "What kind of user are you?",
    "workplace"                      : "Where do you work?",
    "country"                        : "What country are you in?",
    "in_ecp"                         : "Are you part of the U.S. Exascale Computing Project (ECP)?",
    "languages"                      : "What languages do you primarily program in?",
    "app_area"                       : "What are your primary application areas?",
    "how_find_out"                   : "How did you find out about Spack?",
    "how_long_using"                 : "How long have you been using Spack?",
    "how_contributed"                : "Have you contributed to Spack?",
    "spack_versions"                 : "What version(s) of Spack do you use?",
    "os"                             : "What OS do you use Spack on?",
    "num_installations"              : "How many software installations have you done with Spack in the past year?",
    "python_version"                 : "What Python version(s) do you use to run Spack?",
    "how_bad_only_py3"                : "How bad would it be if Spack dropped support for Python 2?",
    "how_use_pkgs"                   : "How do you get installed Spack packages into your environment?",

    "used_features"                  : "Which of the following Spack features do you use?",
    "local_repos"                    : "Do you have local Spack package repositories?",
    "local_mods"                     : "Do you have any local modifications to Spack itself?",

    "cpus_next_year"                 : "Which processors do you expect to use with Spack in the next year?",
    "gpus_next_year"                 : "Which GPUs do you expect to use with Spack in the next year?",
    "compilers_next_year"            : "Which compilers do you expect to use with Spack in the next year?",

    "feature_better_error_messages"  : "Rank these TBD Spack features by importance [Better error messages]",
    "feature_compiler_deps"          : "Rank these TBD Spack features by importance [Compilers as proper dependencies]",
    "feature_separate_build_deps"    : "Rank these TBD Spack features by importance [Separate concretization of build dependencies (build build tools with gcc/clang, not fancy compilers)]",
    "feature_optimized_binaries"     : "Rank these TBD Spack features by importance [Optimized, public binary packages]",
    "feature_build_testing"          : "Rank these TBD Spack features by importance [Build testing for every PR]",
    "feature_language_virtuals"      : "Rank these TBD Spack features by importance [Language virtual dependencies (e.g., depends_on(\"cxx@2017:\"))]",
    "feature_testing"                : "Rank these TBD Spack features by importance [Testing / integration with Pavilion2/ReFrame/other test tool]",
    "feature_better_flag_handling"   : "Rank these TBD Spack features by importance [Better compiler flag handling]",
    "feature_windows"                : "Rank these TBD Spack features by importance [Windows Support]",
    "feature_external_autodetect"    : "Rank these TBD Spack features by importance [Autodetection of system packages]",
    "feature_performance"            : "Rank these TBD Spack features by importance [Performance/speed improvements]",
    "feature_env_reproducibility"    : "Rank these TBD Spack features by importance [More reproducibility for environments]",
    "feature_gpu_compiler_support"   : "Rank these TBD Spack features by importance [Better GPU compiler support]",
    "feature_github_caches"          : "Rank these TBD Spack features by importance [Generate GitHub Actions pipelines + build caches]",
    "feature_developer_support"      : "Rank these TBD Spack features by importance [More developer features]",

    "feature_custom"                 : "What features NOT in the prior list would you like to see?",

    "would_attend_workshop"          : "If we had a (virtual) Spack user meeting, would you attend?",
    "did_tutorial"                   : "Have you done a Spack Tutorial?",
    "how_get_help"                   : "How do you get help with Spack when you need it?",
    "how_often_docs"                 : "How often do you consult the Spack documentation?",
    "commercial_support"             : "If there were commercial support for Spack, would you or your organization buy it?",
    "software_foundation"            : "If there were a software foundation (ala Linux Foundation or CNCF) around Spack would it make you more likely to use/recommend Spack?",

    "quality_docs"                   : "How would you rate the overall quality of ... [Spack documentation]",
    "quality_community"              : "How would you rate the overall quality of ... [Spack community]",
    "quality_packages"               : "How would you rate the overall quality of ... [Spack packages]",
    "quality_spack"                  : "How would you rate the overall quality of ... [Spack]",

    "long_workflow"                  : "Tell us briefly about your use case and your usual Spack workflow.",
    "long_what_helps_most"           : "What about Spack has helped you the most in the past year?",
    "long_pain_points"               : "What are the biggest pain points in Spack for your workflow? (feel free to link to issues)",
    "long_best_improvement"          : "What's the biggest thing we could do to improve Spack over the next year?",
    "long_desired_packages"          : "Are there key packages you'd like to see in Spack that are not included yet?",
    "long_other_comments"            : "Do you have any other comments for us?",

    "email"                          : "If you're interested in being contacted about your response, please leave us your email.",
    "edit_url"                       : "Form Response Edit URL",
}

description_to_name = {names[n]: n for n in names}
