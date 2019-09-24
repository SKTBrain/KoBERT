from setuptools import setup
from kobert import __version__


setup(name='kobert',
      version=__version__,
      url='https://github.com/SKTBrain/KoBERT',
      license='Apache-2.0',
      author='Heewon Jeon',
      author_email='madjakarta@gmail.com',
      description='Korean BERT pre-trained cased (KoBERT) ',
      packages=['kobert', ],
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      )