#!/bin/sh

set -e
rm -rf build capreolus.egg-info dist
python setup.py sdist bdist_wheel

echo
echo now run:
echo 'twine upload   dist/*'
