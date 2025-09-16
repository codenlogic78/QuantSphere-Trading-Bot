"""
QuantSphere AI Trading Platform Setup
Author: Your Name
Created: 2024
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantsphere-ai-trading",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A sophisticated AI-powered trading platform with real-time market analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantsphere-ai-trading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quantsphere=bot:main",
        ],
    },
    keywords="trading, ai, finance, portfolio, quantitative, alpaca, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quantsphere-ai-trading/issues",
        "Source": "https://github.com/yourusername/quantsphere-ai-trading",
        "Documentation": "https://github.com/yourusername/quantsphere-ai-trading/wiki",
    },
)
