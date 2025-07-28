from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="cholera-pdf-scraper",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python project for scraping and processing cholera-related PDF documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ds-cholera-pdf-scraper",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cholera-scraper=src.main:main",
        ],
    },
)
