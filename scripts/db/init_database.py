"""
Initialize MLOps database.

Run once to create schema.
"""

import sys

sys.path.append("/app")

from src.storage.db_manager import get_db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("INITIALIZING MLOPS DATABASE")
    logger.info("=" * 80)

    db = get_db_manager()

    # Execute schema
    logger.info("Creating tables...")
    db.execute_script("/app/scripts/db/schema.sql")

    logger.info("✅ Database initialized successfully")

    # Verify tables exist
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
    """

    tables = db.execute_query(query)

    logger.info(f"\nCreated {len(tables)} tables:")
    for table in tables:
        logger.info(f"  - {table[0]}")


if __name__ == "__main__":
    main()
