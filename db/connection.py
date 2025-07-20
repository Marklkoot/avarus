import os
import mysql.connector
import logging

def get_db_connection():
    host = os.getenv('DB_HOST', 'localhost')
    user = os.getenv('DB_USER', 'avarus_user')
    password = os.getenv('DB_PASS', 'someStrongPassword')
    database = os.getenv('DB_NAME', 'avarus2')

    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
             # The fix => buffered cursors
            buffered=True
        )
        return conn
    except Exception as e:
        logging.error(f"DB connection error => {e}")
        raise
