#!/bin/bash


pycodestyle --exclude=env/ .
if [ $? -eq 0 ] ; 
then
    echo "pycodestyle PASSED"
fi

pydocstyle --match-dir .
if [ $? -eq 0 ] ; 
then
    echo "pydocstyle PASSED"
fi

find . -type f -name "*.py" | xargs pylint --disable C0103 --ignore-paths=env