"""CASC-RL: Cognitive Autonomous Satellite Constellation with Reinforcement Learning."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="casc-rl",
    version="1.0.0",
    author="[Author Names]",
    description="Cognitive Autonomous Satellite Constellation with Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/casc-rl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "casc-train=training.train_marl:main",
            "casc-eval=evaluation.experiment_runner:main",
            "casc-dashboard=visualization.dashboard:main",
        ],
    },
)
