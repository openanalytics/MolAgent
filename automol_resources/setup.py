from setuptools import setup, find_namespace_packages
long_description=''

setup(name='automol_resources',
      version='1.0.1',
      description='AutoMoL: pipeline for automated drug design models ',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='',
      author='Mazen Ahmad, Natalia Dyubankova, Marvin Steijaert, Joris Tavernier',
      author_email='joris.tavernier@openanalytics.eu',
      license='All rights reserved, Open Analytics NV, 2021-2025.',
      packages=find_namespace_packages(),
      zip_safe=False,

      package_data={"": ["*.pt","*.ckpt"]}
      )
