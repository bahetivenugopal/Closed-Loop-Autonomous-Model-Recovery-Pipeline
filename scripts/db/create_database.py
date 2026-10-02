"""
Create MLOps database.

Connects to default 'postgres' database to create 'mlops' database.
Run this BEFORE init_database.py

This script:
1. Connects to PostgreSQL server (default 'postgres' database)
2. Creates 'mlops' database if it doesn't exist
3. Sets proper permissions
"""

import psycopg2
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("CREATING MLOPS DATABASE")
    logger.info("=" * 80)

    # Connection to default postgres database
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "postgres"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": "postgres",  # ← Connect to default database
        "user": os.getenv("POSTGRES_USER", "airflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
    }

    db_name = os.getenv("MLOPS_DB_NAME", "mlops")
    db_user = os.getenv("POSTGRES_USER", "airflow")

    try:
        # Connect to postgres database
        logger.info(f"Connecting to PostgreSQL at {db_config['host']}:{db_config['port']}")
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cur = conn.cursor()

        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()

        if exists:
            logger.info(f"✅ Database '{db_name}' already exists")
        else:
            # Create database
            logger.info(f"Creating database '{db_name}'...")
            cur.execute(f"CREATE DATABASE {db_name} OWNER {db_user}")
            logger.info(f"✅ Database '{db_name}' created successfully")

        cur.close()
        conn.close()

        logger.info("\n" + "=" * 80)
        logger.info("✅ DATABASE CREATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nNext step: python scripts/db/init_database.py")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Failed to create database: {e}")
        raise


if __name__ == "__main__":
    main()
