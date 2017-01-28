c:\\windows\\system32\\taskkill.exe /im python.exe /f
python setup.py build_ext --inplace
cd ArchMM
type NUL > __init__.py
cd Cyfiles
type NUL > __init__.py
cd ../../