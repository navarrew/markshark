#!/usr/bin/env python3
"""
MarkShark Template Manager
Manages bubble sheet templates and their corresponding YAML configuration files.

Each template consists of:
- master_template.pdf: The blank bubble sheet PDF
- bubblemap.yaml: The bubble zone configuration file
- Optional metadata in the YAML for display names and descriptions
"""

import os
import json
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BubbleSheetTemplate:
    """Represents a bubble sheet template with its configuration"""
    template_id: str  # Directory name or unique identifier
    display_name: str  # Human-readable name for UI
    template_pdf_path: Path
    bubblemap_yaml_path: Path
    description: str = ""
    num_questions: Optional[int] = None
    num_choices: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert template to dictionary for JSON/YAML serialization"""
        return {
            'template_id': self.template_id,
            'display_name': self.display_name,
            'template_pdf_path': str(self.template_pdf_path),
            'bubblemap_yaml_path': str(self.bubblemap_yaml_path),
            'description': self.description,
            'num_questions': self.num_questions,
            'num_choices': self.num_choices,
        }
    
    def __str__(self) -> str:
        """String representation for dropdowns"""
        if self.description:
            return f"{self.display_name} - {self.description}"
        return self.display_name


class TemplateManager:
    """Manages bubble sheet templates directory and template discovery"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager
        
        Args:
            templates_dir: Path to the directory containing template folders.
                          If None, uses the default location (see get_default_templates_dir())
        """
        if templates_dir is None:
            templates_dir = self.get_default_templates_dir()
        
        self.templates_dir = Path(templates_dir).expanduser().resolve()
        self._templates_cache: Optional[List[BubbleSheetTemplate]] = None
        self._archived_templates_cache: Optional[List[BubbleSheetTemplate]] = None

        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Archive directory and preferences file paths
        self.archived_dir = self.templates_dir / "archived"
        self.preferences_file = self.templates_dir / ".preferences.json"

        logger.info(f"TemplateManager initialized with directory: {self.templates_dir}")
    
    @staticmethod
    def get_default_templates_dir() -> Path:
        """
        Get the default templates directory.
        
        Priority:
        1. Environment variable MARKSHARK_TEMPLATES_DIR
        2. Package installation directory (src/markshark/templates)
        3. Current working directory + templates/
        """
        # Check environment variable
        env_dir = os.environ.get('MARKSHARK_TEMPLATES_DIR')
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        
        # Try to find package installation directory
        try:
            # This should work if markshark is installed as a package
            import markshark
            package_dir = Path(markshark.__file__).parent
            templates_dir = package_dir / "templates"
            if templates_dir.exists() or package_dir.exists():
                return templates_dir
        except (ImportError, AttributeError):
            pass
        
        # Fall back to current directory
        return Path.cwd() / "templates"

    def _load_preferences(self) -> Dict[str, any]:
        """
        Load template preferences from .preferences.json file.

        Returns:
            Dictionary with preferences:
            {
                "order": ["template_id1", "template_id2", ...],
                "favorites": ["template_id1", ...]
            }
        """
        if not self.preferences_file.exists():
            return {"order": [], "favorites": []}

        try:
            with open(self.preferences_file, 'r') as f:
                prefs = json.load(f)
                # Ensure required keys exist
                if "order" not in prefs:
                    prefs["order"] = []
                if "favorites" not in prefs:
                    prefs["favorites"] = []
                return prefs
        except Exception as e:
            logger.warning(f"Error loading preferences from {self.preferences_file}: {e}")
            return {"order": [], "favorites": []}

    def _save_preferences(self, preferences: Dict[str, any]) -> None:
        """
        Save template preferences to .preferences.json file.

        Args:
            preferences: Dictionary with order and favorites lists
        """
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            logger.debug(f"Saved preferences to {self.preferences_file}")
        except Exception as e:
            logger.error(f"Error saving preferences to {self.preferences_file}: {e}")

    @staticmethod
    def _find_answer_layouts_in_yaml(yaml_data: dict) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Find answer_layouts in YAML data, supporting both legacy and multi-page formats.
        
        Returns:
            Tuple of (found, num_questions, num_choices)
        """
        if not isinstance(yaml_data, dict):
            return False, None, None
        
        num_questions = None
        num_choices = None
        
        # Check for multi-page format (page_1, page_2, etc.)
        page_keys = [k for k in yaml_data.keys() if k.startswith('page_')]
        
        if page_keys:
            # Multi-page format
            total_questions = 0
            for page_key in sorted(page_keys):
                page_data = yaml_data.get(page_key, {})
                if isinstance(page_data, dict):
                    answer_layouts = page_data.get('answer_layouts', [])
                    if answer_layouts:
                        for layout in answer_layouts:
                            if isinstance(layout, dict):
                                # Count questions (numrows for row-selection layouts)
                                total_questions += layout.get('numrows', 0)
                                # Get num_choices from numcols (or labels length)
                                if num_choices is None:
                                    num_choices = layout.get('numcols')
                                    if num_choices is None and 'labels' in layout:
                                        num_choices = len(layout['labels'])
            
            if total_questions > 0:
                num_questions = total_questions
                return True, num_questions, num_choices
        
        # Check for legacy flat format (answer_layouts at top level)
        if 'answer_layouts' in yaml_data:
            answer_layouts = yaml_data['answer_layouts']
            if isinstance(answer_layouts, list):
                total_questions = 0
                for layout in answer_layouts:
                    if isinstance(layout, dict):
                        total_questions += layout.get('numrows', 0)
                        if num_choices is None:
                            num_choices = layout.get('numcols')
                            if num_choices is None and 'labels' in layout:
                                num_choices = len(layout['labels'])
                if total_questions > 0:
                    num_questions = total_questions
                return True, num_questions, num_choices
        
        # Check for old answer_rows format (legacy)
        if 'answer_rows' in yaml_data:
            answer_rows = yaml_data['answer_rows']
            if isinstance(answer_rows, list):
                num_questions = len(answer_rows)
                if answer_rows and isinstance(answer_rows[0], dict):
                    choices = answer_rows[0].get('choices', [])
                    num_choices = len(choices) if choices else None
                return True, num_questions, num_choices
        
        return False, None, None
    
    def _scan_templates_in_dir(self, directory: Path) -> List[BubbleSheetTemplate]:
        """
        Internal method to scan templates in a specific directory.

        Args:
            directory: Directory to scan for templates

        Returns:
            List of BubbleSheetTemplate objects
        """
        templates = []

        if not directory.exists():
            return templates

        # Iterate through subdirectories
        for subdir in sorted(directory.iterdir()):
            if not subdir.is_dir():
                continue

            # Skip hidden directories
            if subdir.name.startswith('.'):
                continue
            
            # Look for required files
            pdf_path = subdir / "master_template.pdf"
            yaml_path = subdir / "bubblemap.yaml"
            
            if not pdf_path.exists():
                logger.debug(f"Skipping {subdir.name}: missing master_template.pdf")
                continue
            
            if not yaml_path.exists():
                logger.debug(f"Skipping {subdir.name}: missing bubblemap.yaml")
                continue
            
            # Try to load metadata from YAML
            template_id = subdir.name
            display_name = template_id.replace('_', ' ').title()
            description = ""
            num_questions = None
            num_choices = None
            
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    
                    # Look for metadata section
                    if isinstance(yaml_data, dict):
                        metadata = yaml_data.get('metadata', {})
                        if metadata:
                            display_name = metadata.get('display_name', display_name)
                            description = metadata.get('description', description)
                            num_questions = metadata.get('num_questions', num_questions)
                            num_choices = metadata.get('num_choices', num_choices)
                            # Also check total_questions (alternate key)
                            if num_questions is None:
                                num_questions = metadata.get('total_questions', num_questions)
                        
                        # Try to infer num_questions and num_choices from answer layouts
                        if num_questions is None or num_choices is None:
                            found, inferred_questions, inferred_choices = self._find_answer_layouts_in_yaml(yaml_data)
                            if num_questions is None:
                                num_questions = inferred_questions
                            if num_choices is None:
                                num_choices = inferred_choices
                
            except Exception as e:
                logger.warning(f"Error reading metadata from {yaml_path}: {e}")
            
            template = BubbleSheetTemplate(
                template_id=template_id,
                display_name=display_name,
                template_pdf_path=pdf_path,
                bubblemap_yaml_path=yaml_path,
                description=description,
                num_questions=num_questions,
                num_choices=num_choices,
            )


            templates.append(template)
            logger.debug(f"Found template: {template}")

        return templates

    def scan_templates(self, force_refresh: bool = False, apply_ordering: bool = True) -> List[BubbleSheetTemplate]:
        """
        Scan the templates directory and return a list of available (non-archived) templates.

        Directory structure expected:
        templates/
            template_name_1/
                master_template.pdf
                bubblemap.yaml
            template_name_2/
                master_template.pdf
                bubblemap.yaml
            archived/
                old_template/
                    master_template.pdf
                    bubblemap.yaml

        Args:
            force_refresh: If True, ignore cache and rescan directory
            apply_ordering: If True, apply custom ordering from preferences

        Returns:
            List of BubbleSheetTemplate objects (ordered by preferences if apply_ordering=True)
        """
        if self._templates_cache is not None and not force_refresh:
            return self._templates_cache

        if not self.templates_dir.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_dir}")
            self._templates_cache = []
            return []

        templates = self._scan_templates_in_dir(self.templates_dir)

        # Apply custom ordering if requested
        if apply_ordering:
            templates = self._apply_ordering(templates)

        self._templates_cache = templates
        return templates

    def scan_archived_templates(self, force_refresh: bool = False) -> List[BubbleSheetTemplate]:
        """
        Scan the archived directory and return a list of archived templates.

        Args:
            force_refresh: If True, ignore cache and rescan directory

        Returns:
            List of archived BubbleSheetTemplate objects
        """
        if self._archived_templates_cache is not None and not force_refresh:
            return self._archived_templates_cache

        if not self.archived_dir.exists():
            self._archived_templates_cache = []
            return []

        templates = self._scan_templates_in_dir(self.archived_dir)
        self._archived_templates_cache = templates
        return templates

    def _apply_ordering(self, templates: List[BubbleSheetTemplate]) -> List[BubbleSheetTemplate]:
        """
        Apply custom ordering to templates based on preferences.

        Args:
            templates: List of templates to order

        Returns:
            Ordered list of templates (favorites first, then custom order, then alphabetical)
        """
        preferences = self._load_preferences()
        order = preferences.get("order", [])
        favorites = preferences.get("favorites", [])

        # Create dictionaries for fast lookup
        template_dict = {t.template_id: t for t in templates}
        ordered_templates = []
        remaining_templates = set(template_dict.keys())

        # First, add favorites in their specified order
        for fav_id in favorites:
            if fav_id in order:
                # Favorite is in custom order, will be added in order section
                continue
            if fav_id in template_dict:
                ordered_templates.append(template_dict[fav_id])
                remaining_templates.discard(fav_id)

        # Then, add templates in custom order
        for template_id in order:
            if template_id in template_dict:
                ordered_templates.append(template_dict[template_id])
                remaining_templates.discard(template_id)

        # Finally, add remaining templates alphabetically
        remaining_sorted = sorted(remaining_templates, key=lambda tid: template_dict[tid].display_name.lower())
        for template_id in remaining_sorted:
            ordered_templates.append(template_dict[template_id])

        return ordered_templates
    
    def get_template(self, template_id: str) -> Optional[BubbleSheetTemplate]:
        """
        Get a specific template by its ID (directory name).
        
        Args:
            template_id: The template identifier (directory name)
            
        Returns:
            BubbleSheetTemplate object or None if not found
        """
        templates = self.scan_templates()
        for template in templates:
            if template.template_id == template_id:
                return template
        return None
    
    def get_template_names(self) -> List[str]:
        """
        Get a list of template display names for UI dropdowns.
        
        Returns:
            List of display names
        """
        templates = self.scan_templates()
        return [template.display_name for template in templates]
    
    def get_template_by_display_name(self, display_name: str) -> Optional[BubbleSheetTemplate]:
        """
        Get a template by its display name.
        
        Args:
            display_name: The display name shown in UI
            
        Returns:
            BubbleSheetTemplate object or None if not found
        """
        templates = self.scan_templates()
        for template in templates:
            if template.display_name == display_name:
                return template
        return None
    
    def create_example_template(self, template_id: str = "example_50q_5choice") -> Path:
        """
        Create an example template directory structure with sample files.
        Useful for first-time setup or documentation.
        
        Args:
            template_id: Name for the example template directory
            
        Returns:
            Path to the created template directory
        """
        template_dir = self.templates_dir / template_id
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create example bubblemap.yaml (multi-page format)
        example_yaml = {
            'metadata': {
                'display_name': 'Example 50 Question Test',
                'description': '50 questions, 5 choices (A-E)',
                'pages': 1,
                'total_questions': 50,
                'schema_version': 1,
            },
            'page_1': {
                'answer_layouts': [
                    {
                        'x_topleft': 0.1,
                        'y_topleft': 0.1,
                        'x_bottomright': 0.9,
                        'y_bottomright': 0.9,
                        'radius_pct': 0.008,
                        'numrows': 50,
                        'numcols': 5,
                        'bubble_shape': 'circle',
                        'selection_axis': 'row',
                        'labels': 'ABCDE',
                    }
                ],
            },
        }
        
        yaml_path = template_dir / "bubblemap.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(example_yaml, f, default_flow_style=False, sort_keys=False)
        
        # Note: master_template.pdf would need to be created separately
        # This is just a placeholder
        pdf_path = template_dir / "master_template.pdf"
        pdf_path.touch()  # Create empty file as placeholder
        
        logger.info(f"Created example template at: {template_dir}")
        return template_dir
    
    def validate_template(self, template: BubbleSheetTemplate) -> Tuple[bool, List[str]]:
        """
        Validate that a template has all required files and valid structure.
        
        Args:
            template: BubbleSheetTemplate to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check PDF exists and is readable
        if not template.template_pdf_path.exists():
            errors.append(f"Template PDF not found: {template.template_pdf_path}")
        elif not template.template_pdf_path.is_file():
            errors.append(f"Template PDF path is not a file: {template.template_pdf_path}")
        
        # Check YAML exists and is readable
        if not template.bubblemap_yaml_path.exists():
            errors.append(f"Bubblemap YAML not found: {template.bubblemap_yaml_path}")
        elif not template.bubblemap_yaml_path.is_file():
            errors.append(f"Bubblemap YAML path is not a file: {template.bubblemap_yaml_path}")
        else:
            # Try to load and validate YAML structure
            try:
                with open(template.bubblemap_yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    
                if not isinstance(yaml_data, dict):
                    errors.append("Bubblemap YAML must contain a dictionary/mapping")
                else:
                    # Check for answer layouts using the helper that supports all formats
                    found, _, _ = self._find_answer_layouts_in_yaml(yaml_data)
                    if not found:
                        errors.append(
                            "Bubblemap YAML missing answer layouts. "
                            "Expected 'answer_layouts' in page_X sections (multi-page format) "
                            "or at top level (legacy format)"
                        )
                    
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML syntax: {e}")
            except Exception as e:
                errors.append(f"Error reading YAML: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors

    def archive_template(self, template_id: str) -> bool:
        """
        Archive a template by moving it to the archived directory.

        Args:
            template_id: The template identifier (directory name)

        Returns:
            True if successful, False otherwise
        """
        template = self.get_template(template_id)
        if not template:
            logger.error(f"Template '{template_id}' not found")
            return False

        # Create archive directory if it doesn't exist
        self.archived_dir.mkdir(parents=True, exist_ok=True)

        # Source and destination paths
        src_dir = self.templates_dir / template_id
        dst_dir = self.archived_dir / template_id

        # Check if destination already exists
        if dst_dir.exists():
            logger.error(f"Archived template '{template_id}' already exists in archive")
            return False

        try:
            # Move the template directory
            shutil.move(str(src_dir), str(dst_dir))
            logger.info(f"Archived template '{template_id}' to {dst_dir}")

            # Clear caches
            self._templates_cache = None
            self._archived_templates_cache = None

            return True
        except Exception as e:
            logger.error(f"Error archiving template '{template_id}': {e}")
            return False

    def unarchive_template(self, template_id: str) -> bool:
        """
        Unarchive a template by moving it from archived back to the main templates directory.

        Args:
            template_id: The template identifier (directory name)

        Returns:
            True if successful, False otherwise
        """
        # Source and destination paths
        src_dir = self.archived_dir / template_id
        dst_dir = self.templates_dir / template_id

        # Check if source exists
        if not src_dir.exists():
            logger.error(f"Archived template '{template_id}' not found in archive")
            return False

        # Check if destination already exists
        if dst_dir.exists():
            logger.error(f"Template '{template_id}' already exists in active templates")
            return False

        try:
            # Move the template directory back
            shutil.move(str(src_dir), str(dst_dir))
            logger.info(f"Unarchived template '{template_id}' to {dst_dir}")

            # Clear caches
            self._templates_cache = None
            self._archived_templates_cache = None

            return True
        except Exception as e:
            logger.error(f"Error unarchiving template '{template_id}': {e}")
            return False

    def set_template_order(self, ordered_template_ids: List[str]) -> bool:
        """
        Set custom ordering for templates.

        Args:
            ordered_template_ids: List of template IDs in desired order

        Returns:
            True if successful, False otherwise
        """
        preferences = self._load_preferences()
        preferences["order"] = ordered_template_ids

        try:
            self._save_preferences(preferences)
            # Clear cache to force reordering on next scan
            self._templates_cache = None
            logger.info(f"Updated template order: {ordered_template_ids}")
            return True
        except Exception as e:
            logger.error(f"Error setting template order: {e}")
            return False

    def toggle_favorite(self, template_id: str) -> bool:
        """
        Toggle a template as favorite (pinned to top).

        Args:
            template_id: The template identifier

        Returns:
            True if now favorited, False if unfavorited, None on error
        """
        preferences = self._load_preferences()
        favorites = preferences.get("favorites", [])

        if template_id in favorites:
            favorites.remove(template_id)
            is_favorite = False
            logger.info(f"Removed '{template_id}' from favorites")
        else:
            favorites.append(template_id)
            is_favorite = True
            logger.info(f"Added '{template_id}' to favorites")

        preferences["favorites"] = favorites

        try:
            self._save_preferences(preferences)
            # Clear cache to force reordering on next scan
            self._templates_cache = None
            return is_favorite
        except Exception as e:
            logger.error(f"Error toggling favorite for '{template_id}': {e}")
            return None

    def is_favorite(self, template_id: str) -> bool:
        """
        Check if a template is marked as favorite.

        Args:
            template_id: The template identifier

        Returns:
            True if template is favorite, False otherwise
        """
        preferences = self._load_preferences()
        favorites = preferences.get("favorites", [])
        return template_id in favorites

    def move_template_up(self, template_id: str) -> bool:
        """
        Move a template up one position in the custom order.

        Args:
            template_id: The template identifier

        Returns:
            True if successful, False otherwise
        """
        preferences = self._load_preferences()
        order = preferences.get("order", [])

        # If template is not in order yet, add all current templates to order first
        if template_id not in order:
            templates = self.scan_templates(apply_ordering=False)
            order = [t.template_id for t in templates]

        try:
            idx = order.index(template_id)
            if idx > 0:
                order[idx], order[idx - 1] = order[idx - 1], order[idx]
                preferences["order"] = order
                self._save_preferences(preferences)
                self._templates_cache = None
                logger.debug(f"Moved template '{template_id}' up in order")
                return True
            else:
                logger.debug(f"Template '{template_id}' is already at the top")
                return False
        except ValueError:
            logger.error(f"Template '{template_id}' not found in order")
            return False

    def move_template_down(self, template_id: str) -> bool:
        """
        Move a template down one position in the custom order.

        Args:
            template_id: The template identifier

        Returns:
            True if successful, False otherwise
        """
        preferences = self._load_preferences()
        order = preferences.get("order", [])

        # If template is not in order yet, add all current templates to order first
        if template_id not in order:
            templates = self.scan_templates(apply_ordering=False)
            order = [t.template_id for t in templates]

        try:
            idx = order.index(template_id)
            if idx < len(order) - 1:
                order[idx], order[idx + 1] = order[idx + 1], order[idx]
                preferences["order"] = order
                self._save_preferences(preferences)
                self._templates_cache = None
                logger.debug(f"Moved template '{template_id}' down in order")
                return True
            else:
                logger.debug(f"Template '{template_id}' is already at the bottom")
                return False
        except ValueError:
            logger.error(f"Template '{template_id}' not found in order")
            return False


# Convenience functions for CLI/GUI integration
def list_available_templates(templates_dir: Optional[str] = None) -> List[BubbleSheetTemplate]:
    """
    Convenience function to list all available templates.
    
    Args:
        templates_dir: Optional custom templates directory
        
    Returns:
        List of BubbleSheetTemplate objects
    """
    manager = TemplateManager(templates_dir)
    return manager.scan_templates()


def get_template_by_name(template_name: str, templates_dir: Optional[str] = None) -> Optional[BubbleSheetTemplate]:
    """
    Convenience function to get a template by display name or ID.
    
    Args:
        template_name: Display name or template ID
        templates_dir: Optional custom templates directory
        
    Returns:
        BubbleSheetTemplate object or None if not found
    """
    manager = TemplateManager(templates_dir)
    
    # Try by display name first
    template = manager.get_template_by_display_name(template_name)
    if template:
        return template
    
    # Fall back to template ID
    return manager.get_template(template_name)


def generate_mock_dataset_for_template(
    template: BubbleSheetTemplate,
    out_dir: str,
    num_students: int = 100,
    seed: int = 42,
    dpi: int = 300,
    verbose: bool = True,
) -> Dict:
    """
    Generate a mock dataset for a specific template.

    This is a convenience function that wraps the mock_dataset module
    and works with BubbleSheetTemplate objects.

    Args:
        template: BubbleSheetTemplate to generate data for
        out_dir: Output directory for generated files
        num_students: Number of fake students to generate
        seed: Random seed for reproducibility
        dpi: DPI for rendered images
        verbose: Print progress messages

    Returns:
        Dictionary with paths to generated files
    """
    from .mock_dataset import generate_mock_dataset

    return generate_mock_dataset(
        template_path=str(template.template_pdf_path),
        bubblemap_path=str(template.bubblemap_yaml_path),
        out_dir=out_dir,
        num_students=num_students,
        seed=seed,
        dpi=dpi,
        verbose=verbose,
    )


__all__ = [
    'BubbleSheetTemplate',
    'TemplateManager',
    'list_available_templates',
    'get_template_by_name',
    'generate_mock_dataset_for_template',
]
