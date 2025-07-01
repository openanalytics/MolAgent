from setuptools import setup, find_namespace_packages

exec(open('automol/version.py').read())


extras_require = {
    'registry' : ['boto3'],
    'util' : ['boto3', 'IPython', 'seaborn', 'psutil'],
     'cadd_util': ['psycopg2']
      }
extras_require['all'] = list(set(item for k,v in extras_require.items() for item in v))

setup(name='automol',
      version=__version__,
      description='AutoMoL: pipeline for automated drug design models ',
      long_description="",
      long_description_content_type="text/markdown",
      url='https://sourcecode.jnj.com/scm/asx-ncnk/jnj_auto_ml.git',
      author='Mazen Ahmad, Natalia Dyubankova, Marvin Steijaert, Joris Tavernier',
      author_email='joris.tavernier@openanalytics.eu',
      license='All rights reserved, Open Analytics NV, 2021-2025.',
      packages=find_namespace_packages(),
      install_requires=[
          'rdkit', # installed by conda; including in install_requires will give an error in pip install
          'torch',
          'tqdm',
          'numpy',
          'pandas',
          'seaborn',
          'joblib',
          'scikit-learn', # (31 MB)
          'pymongo',
          'hyperopt',
          'xgboost', # (399 MB)
          'lightgbm',
          'py3nvml', # (458 kB)
          'matplotlib', # (35 MB)
          'plotly',
          'bokeh',
          'pdfkit',
          'jinja2',
          'kaleido',
          'psutil',
          'cython'
      ],
      dependency_links = [], # do not use dependency_links, as it is no longer supported by pip install
      extras_require = extras_require,
      zip_safe = False,
      package_data = {"": ["*.jpg", "*.html", "*.csv"]})
