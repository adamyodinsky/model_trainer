.PHONY: setup init train predict summary

setup:
	poetry install

init: 
	poetry run python3 model_trainer/main.py init

train:
	poetry run python3 model_trainer/main.py train -s 1980-01-01

predict:
	poetry run python3 model_trainer/main.py predict -s 2015-01-01

summary:
	poetry run python3 model_trainer/main.py summary