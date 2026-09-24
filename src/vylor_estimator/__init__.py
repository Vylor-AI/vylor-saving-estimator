"""vylor-savings-estimator: Estimate Vylor MCP savings on Claude sessions."""

__version__ = "0.1.0"

from vylor_estimator.classifier import (
    ClassifiedTurn,
    ITurnClassifier,
    PatternTurnClassifier,
    VylorTool,
    classify_turns,
)
from vylor_estimator.discovery import (
    ClaudeSessionDiscoverer,
    ISessionDiscoverer,
    discover_sessions,
)
from vylor_estimator.parser import (
    ClaudeJsonlParser,
    ISessionParser,
    ToolCall,
    Turn,
    parse_files,
)
from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
    compute_turn_cost,
    get_pricing,
)
from vylor_estimator.report import (
    HtmlReportRenderer,
    IReportRenderer,
    JsonReportRenderer,
    ReportRendererFactory,
    TerminalReportRenderer,
    print_report,
    write_html_report,
    write_json_report,
)
from vylor_estimator.savings import (
    Aggregate,
    ConservativeSavingsEngine,
    ISavingsEngine,
    SavingsReport,
    TurnSavings,
    compute_savings,
)
from vylor_estimator.service import (
    DateFilter,
    EstimatorService,
    create_estimator,
)

__all__ = [
    "__version__",
    "ISessionDiscoverer",
    "ClaudeSessionDiscoverer",
    "discover_sessions",
    "ISessionParser",
    "ClaudeJsonlParser",
    "parse_files",
    "ToolCall",
    "Turn",
    "ITurnClassifier",
    "PatternTurnClassifier",
    "ClassifiedTurn",
    "VylorTool",
    "classify_turns",
    "IPricingCalculator",
    "ClaudePricingCalculator",
    "get_pricing",
    "compute_turn_cost",
    "ISavingsEngine",
    "ConservativeSavingsEngine",
    "compute_savings",
    "Aggregate",
    "SavingsReport",
    "TurnSavings",
    "IReportRenderer",
    "ReportRendererFactory",
    "TerminalReportRenderer",
    "HtmlReportRenderer",
    "JsonReportRenderer",
    "print_report",
    "write_html_report",
    "write_json_report",
    "DateFilter",
    "EstimatorService",
    "create_estimator",
]
