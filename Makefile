 
init: 
	poetry run python3 model_trainer/main.py init

train:
	poetry run python3 model_trainer/main.py train

predict:
	poetry run python3 model_trainer/main.py predict