# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "deep_tree_echo_model",
    srcs = [
        "__init__.py",
        "deep_tree_echo_model.py",
    ],
    deps = [
        "//ydf/cc:ydf",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model/decision_forest_model",
    ],
)

# Tests
# =====

py_test(
    name = "deep_tree_echo_model_test",
    srcs = ["deep_tree_echo_model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":deep_tree_echo_model",
        # absl/testing:absltest dep,
    ],
)
