#!/usr/bin/env python3
"""
Test script for new template manager features (archive and ordering).
Run this from the repository root to test the new functionality.
"""

import sys
from pathlib import Path

# Add src to path so we can import markshark
sys.path.insert(0, str(Path(__file__).parent / "src"))

from markshark.template_manager import TemplateManager

def test_template_manager():
    """Test template manager archive and ordering features."""

    print("=" * 60)
    print("Testing MarkShark Template Manager")
    print("=" * 60)

    # Initialize template manager
    print("\n1. Initializing template manager...")
    tm = TemplateManager()
    print(f"   Templates directory: {tm.templates_dir}")
    print(f"   Archive directory: {tm.archived_dir}")
    print(f"   Preferences file: {tm.preferences_file}")

    # Scan templates
    print("\n2. Scanning active templates...")
    templates = tm.scan_templates(force_refresh=True)
    print(f"   Found {len(templates)} active template(s):")
    for t in templates:
        is_fav = tm.is_favorite(t.template_id)
        fav_marker = "⭐" if is_fav else "  "
        print(f"   {fav_marker} {t.display_name} ({t.template_id})")
        if t.num_questions:
            print(f"      Questions: {t.num_questions}, Choices: {t.num_choices}")

    # Scan archived templates
    print("\n3. Scanning archived templates...")
    archived = tm.scan_archived_templates(force_refresh=True)
    print(f"   Found {len(archived)} archived template(s):")
    for t in archived:
        print(f"   📦 {t.display_name} ({t.template_id})")

    # Test preferences
    print("\n4. Testing preferences...")
    prefs = tm._load_preferences()
    print(f"   Current order: {prefs.get('order', [])}")
    print(f"   Favorites: {prefs.get('favorites', [])}")

    # Test archive functionality (if we have templates)
    if templates:
        print("\n5. Testing archive functionality...")
        test_template = templates[0]
        print(f"   Template to test with: {test_template.display_name} ({test_template.template_id})")

        # Test favorite toggle
        print("\n   5a. Testing favorite toggle...")
        was_fav = tm.is_favorite(test_template.template_id)
        print(f"       Before: Favorite = {was_fav}")

        tm.toggle_favorite(test_template.template_id)
        is_fav_now = tm.is_favorite(test_template.template_id)
        print(f"       After toggle: Favorite = {is_fav_now}")

        # Toggle back
        tm.toggle_favorite(test_template.template_id)
        print(f"       Toggled back: Favorite = {tm.is_favorite(test_template.template_id)}")

        # Test ordering
        if len(templates) > 1:
            print("\n   5b. Testing template ordering...")
            print(f"       Moving {test_template.template_id} down...")
            success = tm.move_template_down(test_template.template_id)
            print(f"       Success: {success}")

            # Check new order
            reordered = tm.scan_templates(force_refresh=True)
            print(f"       New order: {[t.template_id for t in reordered]}")

            # Move back up
            print(f"       Moving {test_template.template_id} up...")
            tm.move_template_up(test_template.template_id)
            restored = tm.scan_templates(force_refresh=True)
            print(f"       Restored order: {[t.template_id for t in restored]}")

        # Note: We're NOT actually testing archive/unarchive to avoid
        # modifying the user's template directory
        print("\n   Note: Skipping actual archive/unarchive to preserve templates")
        print("         Archive/unarchive can be tested via the GUI")

    print("\n" + "=" * 60)
    print("✅ Template Manager Tests Complete!")
    print("=" * 60)
    print("\nFeatures available:")
    print("  ⭐ Favorite/pin templates to top of list")
    print("  ⬆️⬇️ Reorder templates with up/down buttons")
    print("  📦 Archive inactive templates (hidden from dropdowns)")
    print("  ↩️ Restore archived templates back to active")
    print("\nTo test in GUI:")
    print("  1. Run: streamlit run src/markshark/app_streamlit.py")
    print("  2. Go to '5) Template manager' page")
    print("  3. Use the buttons to manage templates")
    print()


if __name__ == "__main__":
    try:
        test_template_manager()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
