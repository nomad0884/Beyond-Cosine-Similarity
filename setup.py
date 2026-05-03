from setuptools import setup, find_packages

setup(
    name="korean-news-dedup",
    version="0.1.0",
    author="intertek",
    description="Korean news deduplication with STS-NLI combined pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/korean-news-dedup",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Korean",
    ],
)
