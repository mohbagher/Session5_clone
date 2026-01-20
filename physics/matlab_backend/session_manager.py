"""
MATLAB Engine Session Manager
==============================
Persistent, reusable MATLAB Engine for high-performance integration.
"""

# Conditional import for MATLAB Engine
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None  # Placeholder

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """MATLAB session metadata."""
    session_id: str
    start_time: datetime
    matlab_version: str
    available_toolboxes: list
    status: str  # 'connected', 'disconnected', 'error'


class MATLABSessionManager:
    """
    Singleton manager for persistent MATLAB Engine session.

    Features:
    - Single reusable session across runs
    - Automatic reconnection
    - Graceful failure handling
    - Session health monitoring
    """

    _instance = None
    _engine: Optional[Any] = None  # Type will be matlab.engine.MatlabEngine when available
    _session_info: Optional[SessionInfo] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            logger.info("MATLAB Session Manager initialized")

    def start_session(self, force_restart: bool = False) -> bool:
        """
        Start or reuse MATLAB Engine session.

        Args:
            force_restart: Force new session even if one exists

        Returns:
            True if session is ready, False if failed
        """

        if not MATLAB_AVAILABLE:
            logger.error("MATLAB Engine for Python not installed")
            self._session_info = SessionInfo(
                session_id="failed",
                start_time=datetime.now(),
                matlab_version="not_available",
                available_toolboxes=[],
                status='error'
            )
            return False

        # Check if session already exists and is healthy
        if not force_restart and self.is_session_active():
            logger.info("Reusing existing MATLAB session")
            return True

        # Stop old session if forcing restart
        if force_restart and self._engine is not None:
            self.stop_session()

        try:
            logger.info("Starting new MATLAB Engine session...")

            # Start MATLAB Engine
            self._engine = matlab.engine.start_matlab()

            # Get MATLAB version
            version = self._engine.version()

            # Get available toolboxes
            toolboxes = self._get_available_toolboxes()

            # Create session info
            self._session_info = SessionInfo(
                session_id=f"matlab_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                start_time=datetime.now(),
                matlab_version=version,
                available_toolboxes=toolboxes,
                status='connected'
            )

            logger.info(f"MATLAB Engine started successfully (v{version})")
            logger.info(f"Available toolboxes: {', '.join(toolboxes[:5])}...")

            return True

        except Exception as e:
            logger.error(f"Failed to start MATLAB Engine: {e}")
            self._session_info = SessionInfo(
                session_id="failed",
                start_time=datetime.now(),
                matlab_version="unknown",
                available_toolboxes=[],
                status='error'
            )
            return False

    def stop_session(self):
        """Stop MATLAB Engine session."""
        if self._engine is not None:
            try:
                self._engine.quit()
                logger.info("MATLAB Engine session stopped")
            except:
                logger.warning("Error stopping MATLAB Engine (may already be closed)")
            finally:
                self._engine = None
                if self._session_info:
                    self._session_info.status = 'disconnected'

    def restart_session(self) -> bool:
        """Restart MATLAB Engine session."""
        logger.info("Restarting MATLAB Engine session...")
        return self.start_session(force_restart=True)

    def is_session_active(self) -> bool:
        """Check if MATLAB session is active and responsive."""
        if self._engine is None:
            return False

        try:
            # Simple health check
            _ = self._engine.eval('1+1', nargout=0)
            return True
        except:
            logger.warning("MATLAB session is not responsive")
            return False

    def get_engine(self) -> Optional[Any]:
        """
        Get active MATLAB Engine instance.

        Returns:
            MATLAB Engine or None if not available
        """
        if not self.is_session_active():
            # Try to start session
            if not self.start_session():
                return None

        return self._engine

    def get_session_info(self) -> Optional[SessionInfo]:
        """Get current session metadata."""
        return self._session_info

    def _get_available_toolboxes(self) -> list:
        """Query MATLAB for installed toolboxes."""
        try:
            # Get toolbox info from MATLAB
            toolbox_info = self._engine.ver(nargout=1)

            # Extract toolbox names
            toolboxes = []
            for item in toolbox_info:
                toolboxes.append(item['Name'])

            return toolboxes
        except:
            logger.warning("Could not query MATLAB toolboxes")
            return []

    def execute_function(self, func_name: str, *args, **kwargs):
        """
        Execute MATLAB function safely.

        Args:
            func_name: Name of MATLAB function
            *args: Positional arguments for function
            **kwargs: Keyword arguments (nargout, etc.)

        Returns:
            Function output or None if failed
        """
        engine = self.get_engine()
        if engine is None:
            raise RuntimeError("MATLAB Engine not available")

        try:
            func = getattr(engine, func_name)
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing MATLAB function '{func_name}': {e}")
            raise


# Global session manager instance
_session_manager = MATLABSessionManager()


def get_session_manager() -> MATLABSessionManager:
    """Get global MATLAB session manager."""
    return _session_manager