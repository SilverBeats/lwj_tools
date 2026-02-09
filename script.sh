clear
 sphinx-apidoc -o ./docs/source ./src/lwj_tools
 cd docs
 make clean && make html