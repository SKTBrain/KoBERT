from setuptools import setup


def install_requires():
    with open("requirements.txt") as f:
        lines = f.read().splitlines()
        install_requires = [line for line in lines]
        return install_requires


setup(
    name="kobert",
    version="0.1.2",
    url="https://github.com/SKTBrain/KoBERT",
    license="Apache-2.0",
    author="Heewon Jeon",
    author_email="madjakarta@gmail.com",
    description="Korean BERT pre-trained cased (KoBERT) ",
    packages=[
        "kobert",
    ],
    long_description=open("README.md", encoding="utf-8").read(),
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=install_requires(),
)
