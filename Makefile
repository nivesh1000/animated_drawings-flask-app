.ONESHELL:
VENV_NAME=.venv
PYTHON=python3
PIP=$(VENV_NAME)/bin/pip
.PHONY: virtualenv install

help:
	@echo "Makefile Commands:"
	@echo "  virtualenv  - Creates a virtual environment named $(VENV_NAME), installs pip, and sets up dependencies."
	@echo "  install     - Installs dependencies from the requirements.txt file."
	@echo "  run         - Runs the main.py script with optional arguments passed via ARGS."
	@echo ""
	@echo "Usage Examples:"
	@echo "  make virtualenv     - Sets up a new virtual environment and installs dependencies."
	@echo "  make install        - Installs dependencies into the virtual environment."
	@echo "  make run ARGS='-q \"hello\"' - Runs main.py with the specified arguments."
	@echo ""
	@echo "Note: Remember to activate the virtual environment by running: source $(VENV_NAME)/bin/activate"

virtualenv:
	@echo "Creating virtualenv ..."
	@rm -rf $(VENV_NAME)
	@$(PYTHON) -m venv $(VENV_NAME)
	@$(PIP) install -U pip
	@echo "Virtual environment created."
	@make install

install:
	@echo "Installing dependencies ..."
	@$(PIP) install -r requirements.txt
	@echo "Dependencies installed."
	@echo "!!!Activate the virtual environment by running: source $(VENV_NAME)/bin/activate"

run:
	@$(PYTHON) run.py


