# setup.py
from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent

setup(
    name="quantile2spacetime",
    version="0.1.0",
    description="ML quantiles → latent Gaussian fields → coherent spatio-temporal simulation",
    long_description=(HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    url="https://github.com/<USER>/<REPO>",
    license="MIT",  # or BSD-3-Clause
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",

    # Keep core deps lightweight
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
    ],

    # Heavy / optional dependencies
    extras_require={
        # Quantile regression forests
        "qrf": [
            "quantile-forest",   # pip package name for quantile_forest
        ],

        # Torch + quantnn for QRNN
        "torch": [
            "torch",
            "quantnn",
        ],

        # If you truly need quantnn's keras side; otherwise you can omit this extra
        "keras": [
            "tensorflow",  # quantnn.models.keras typically needs TF backend installed
        ],

        # JAX for latent GRF simulation
        "jax": [
            "jax",
            "jaxlib",
        ],

        # Convenience: everything
        "all": [
            "quantile-forest",
            "torch",
            "quantnn",
            "jax",
            "jaxlib",
            "tensorflow",
        ],

        # Dev tooling
        "dev": [
            "pytest",
            "ruff",
            "black",
            "pre-commit",
        ],
    },

    include_package_data=True,
)
