import os
import logging




def create_directory(path):
    if path and not os.path.exists(path):
        os.makedirs(path)




def setup_logging(log_file="logs/app.log"):
    create_directory(os.path.dirname(log_file) or '.')
    logging.basicConfig(
        filename=log_file,
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
        )