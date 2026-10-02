"""
Database connection manager.

Uses PostgreSQL (Airflow's existing database).
Connection pooling for efficiency.
"""

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manage PostgreSQL connections.

    Uses Airflow's PostgreSQL instance (already running).
    """

    _pool = None

    @classmethod
    def initialize_pool(cls):
        """Initialize connection pool."""
        if cls._pool is None:
            # Use Airflow's PostgreSQL (or configure separately)
            db_config = {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("MLOPS_DB_NAME", "mlops"),  # Separate DB from Airflow
                "user": os.getenv("POSTGRES_USER", "airflow"),
                "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
            }

            cls._pool = SimpleConnectionPool(minconn=1, maxconn=10, **db_config)

            logger.info("Database connection pool initialized")

    @classmethod
    @contextmanager
    def get_connection(cls):
        """
        Get database connection from pool.

        Usage:
            with DatabaseManager.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT ...")
        """
        if cls._pool is None:
            cls.initialize_pool()

        conn = cls._pool.getconn()
        try:
            logger.debug(f"Got connection from pool, autocommit={conn.autocommit}")
            yield conn
            logger.debug("Committing transaction...")
            conn.commit()
            logger.debug("✅ Transaction committed")
        except Exception as e:
            logger.error(f"Database error, rolling back: {e}")
            conn.rollback()
            raise
        finally:
            cls._pool.putconn(conn)

    @classmethod
    def execute_query(cls, query: str, params: tuple = None):
        """
        Execute query and return results.

        Args:
            query: SQL query
            params: Query parameters (for parameterized queries)

        Returns:
            List of result rows
        """
        with cls.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rowcount = cur.rowcount
            logger.info(f"✅ EXECUTED: {query[:50]}... (rows affected: {rowcount})")

            if cur.description:  # SELECT query
                return cur.fetchall()
            else:  # INSERT/UPDATE/DELETE
                return rowcount

    @classmethod
    def execute_script(cls, script_path: str):
        """
        Execute SQL script file.

        Args:
            script_path: Path to .sql file
        """
        with open(script_path, "r") as f:
            script = f.read()

        with cls.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(script)

        logger.info(f"Executed script: {script_path}")


# Singleton instance
_db_manager = DatabaseManager()


def get_db_manager():
    """Get database manager singleton."""
    return _db_manager
