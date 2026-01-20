"""
MATLAB Backend Module
=====================
Phase 2: MATLAB Engine integration for verified channel generation.
"""

from physics.matlab_backend.session_manager import (
    MATLABSessionManager,
    get_session_manager,
    SessionInfo
)

from physics.matlab_backend.toolbox_registry import (
    ToolboxManager,
    ToolboxInfo,
    ScenarioTemplate,
    TOOLBOX_REGISTRY,
    SCENARIO_TEMPLATES,
    CAPABILITY_TO_SCENARIO
)

from physics.matlab_backend.matlab_source import (
    MATLABEngineSource
)

from physics.matlab_backend.script_generator import (
    MATLABScriptGenerator
)

__all__ = [
    # Session management
    'MATLABSessionManager',
    'get_session_manager',
    'SessionInfo',

    # Toolbox registry
    'ToolboxManager',
    'ToolboxInfo',
    'ScenarioTemplate',
    'TOOLBOX_REGISTRY',
    'SCENARIO_TEMPLATES',
    'CAPABILITY_TO_SCENARIO',

    # Channel source
    'MATLABEngineSource',

    # Script generation
    'MATLABScriptGenerator'
]