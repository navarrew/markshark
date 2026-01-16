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
        
        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def scan_templates(self, force_refresh: bool = False) -> List[BubbleSheetTemplate]:
        """
        Scan the templates directory and return a list of available templates.
        
        Directory structure expected:
        templates/
            template_name_1/
                master_template.pdf
                bubblemap.yaml
            template_name_2/
                master_template.pdf
                bubblemap.yaml
        
        Args:
            force_refresh: If True, ignore cache and rescan directory
            
        Returns:
            List of BubbleSheetTemplate objects
        """
        if self._templates_cache is not None and not force_refresh:
            return self._templates_cache
        
        templates = []
        
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_dir}")
            self._templates_cache = templates
            return templates
        
        # Iterate through subdirectories
        for subdir in sorted(self.templates_dir.iterdir()):
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
                        
                        # Try to infer num_questions from answer_rows if not in metadata
                        if num_questions is None and 'answer_rows' in yaml_data:
                            num_questions = len(yaml_data['answer_rows'])
                        
                        # Try to infer num_choices from first answer row
                        if num_choices is None and 'answer_rows' in yaml_data:
                            answer_rows = yaml_data['answer_rows']
                            if answer_rows and isinstance(answer_rows[0], dict):
                                choices = answer_rows[0].get('choices', [])
                                num_choices = len(choices) if choices else None
                
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
        
        self._templates_cache = templates
        return templates
    
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
        
        # Create example bubblemap.yaml
        example_yaml = {
            'metadata': {
                'display_name': 'Example 50 Question Test',
                'description': '50 questions, 5 choices (A-E)',
                'num_questions': 50,
                'num_choices': 5,
            },
            'answer_rows': [],  # This would be populated with actual bubble coordinates
            'name_zone': {},  # This would be populated with actual zone coordinates
            'id_zone': {},  # This would be populated with actual zone coordinates
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
                    # Check for required keys
                    if 'answer_rows' not in yaml_data:
                        errors.append("Bubblemap YAML missing 'answer_rows' key")
                    
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML syntax: {e}")
            except Exception as e:
                errors.append(f"Error reading YAML: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors


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


__all__ = [
    'BubbleSheetTemplate',
    'TemplateManager',
    'list_available_templates',
    'get_template_by_name',
]
