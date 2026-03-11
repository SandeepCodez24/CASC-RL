# CASC-RL Makefile
# Common commands for training, evaluation, and development

.PHONY: install train-world-model train-marl train-all evaluate dashboard test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

train-world-model:
	python training/train_world_model.py \
	  --config config/training.yaml \
	  --n_samples 100000 \
	  --epochs 200

train-marl:
	python training/train_marl.py \
	  --config config/training.yaml \
	  --world_model_ckpt checkpoints/world_model_best.pt \
	  --n_satellites 3 \
	  --episodes 10000

curriculum:
	python training/curriculum_training.py \
	  --stages 1,2,3,4,5,6 \
	  --auto_advance

train-all:
	python training/train_world_model.py && \
	python training/train_marl.py && \
	python training/curriculum_training.py

evaluate:
	python evaluation/experiment_runner.py --scenarios all

benchmark:
	python evaluation/experiment_runner.py \
	  --agents checkpoints/marl_final.pt \
	  --scenarios all \
	  --seeds 5

dashboard:
	python visualization/dashboard.py --port 8050

plots:
	python visualization/prediction_plots.py --output figures/

test:
	pytest tests/ -v --cov=. --cov-report=html

test-smoke:
	pytest tests/integration/ -k "smoke" --timeout=60

lint:
	python -m flake8 environment/ world_model/ agents/ marl/ coordination/ safety/ training/ evaluation/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

docker-build:
	docker build -t casc-rl .

docker-run:
	docker run --gpus all -v ./checkpoints:/app/checkpoints casc-rl

docker-up:
	docker-compose up
