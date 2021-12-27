run:
	python3 main.py

install:
	pip install -r requirements.txt

lint:
	pylint ./
	black --check --target-version py39 -l 120 --exclude venv .

format:
	black --target-version py39 -l 120 --exclude venv .

