import fastentrypoints
from setuptools import find_packages, setup


setup(
    name="AutoRegKD",
    version="1.0",
    package=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    description="Scripts to train DistilBART",
    entry_point={
        "console_scripts": [
            "autoregkd = autoregkd.__main__: main"
        ]
    }
)
