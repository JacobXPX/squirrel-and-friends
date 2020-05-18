from setuptools import setup, find_packages

setup(
    name="squirrel-and-friends",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "emoji==0.5.4", "nltk==3.5", "pyspellchecker==0.5.4",
        "numerizer==0.1.5", "lightgbm==2.3.1",
    ]
)
