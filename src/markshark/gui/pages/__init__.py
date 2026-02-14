"""
Major workflow pages for the MarkShark GUI.
"""

from .quick_grade import QuickGradePage
from .review_panel import ReviewPanelPage
from .template_manager import TemplateManagerPage
from .mock_data_utility import MockDataPage
from .settings import SettingsPage
from .align_only import AlignOnlyPage
from .score_only import ScoreOnlyPage
from .report_only import ReportOnlyPage
from .pdf_tools import PdfToolsPage
from .lms_integration import LmsIntegrationPage
from .map_viewer import MapViewerPage
from .key_builder import KeyBuilderPage
from .welcome_page import WelcomePage

__all__ = [
    "WelcomePage",
    "QuickGradePage",
    "ReviewPanelPage",
    "TemplateManagerPage",
    "MockDataPage",
    "SettingsPage",
    "AlignOnlyPage",
    "ScoreOnlyPage",
    "ReportOnlyPage",
    "PdfToolsPage",
    "LmsIntegrationPage",
    "MapViewerPage",
    "KeyBuilderPage",
]
