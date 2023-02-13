from setuptools import setup, find_packages

test_packages = ["pytest>=7.2.0", "black>=22.10.0"]

yaml_packages = []

util_packages = []

docs_packages = []

dev_packages = test_packages + util_packages + docs_packages + yaml_packages

all_deps = yaml_packages

setup(
    name="PERFEX",
    version="0.1.1",
    packages=find_packages(include=["perfex", "perfex.*"]),
    extras_require={
        "dev": dev_packages,
        "test": test_packages,
        "all": all_deps,
        "yaml": yaml_packages,
    },
)
