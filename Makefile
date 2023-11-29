all:
	venv/bin/python main.py

lib:
	venv/bin/python -m pip install networkx
	venv/bin/python -m pip install matplotlib
	venv/bin/python -m pip install PyQt5
