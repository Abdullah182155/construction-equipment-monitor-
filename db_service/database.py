"""
Database Connection and Session Management.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from db_service.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Database connection manager with auto table creation."""

    def __init__(self, url: str):
        self.url = url
        self.engine = None
        self.SessionLocal = None
        self._connect()

    def _connect(self):
        """Create engine and session factory."""
        try:
            self.engine = create_engine(
                self.url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            logger.info(f"Database connected: {self.url.split('@')[-1] if '@' in self.url else 'local'}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self):
        """Create all tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self):
        """Close database engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
