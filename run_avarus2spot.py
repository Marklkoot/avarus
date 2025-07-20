import logging
import sys
import os
from dotenv import load_dotenv
import argparse

from executor import Executor

def main():
    parser= argparse.ArgumentParser(description="Avarus 2.0 Launch")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the config file.")
    args= parser.parse_args()

    print(f"Starting Avarus 2.0 with config={args.config}")
    executor= Executor(args.config)
    executor.run()

if __name__=="__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error(e, exc_info=True)
        sys.exit(1)
