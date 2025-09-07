import yaml
import os
import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import validate_images
from src.components.data_preprocessing import Datapreprocessing
from src.components.model_trainer import ModelTrainer
from src.utils.common import setup_logging,create_directory

def run_piepline(config_path="config/config.yaml"):
    config=yaml.safe_load(open(config_path))
    setup_logging(config.get('logging',{}).get('log_file','logs/app/log'))

    create_directory(os.path.dirname(config['training']['model_path']) or 'artifacts')
    create_directory(os.path.dirname(config.get('logging', {}).get('log_file', 'logs/app.log')))