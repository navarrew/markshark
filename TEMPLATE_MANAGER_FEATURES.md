# Template Manager Enhancements

This document describes the new template organization features added to MarkShark.

## Overview

Teachers can now better organize their bubble sheet templates with two new features:

1. **Template Archiving** - Hide inactive templates from dropdown menus
2. **Custom Ordering** - Arrange templates in your preferred order with favorites at the top

## Features

### 1. Archive/Restore Templates

**Problem Solved:** If you only use 1-2 templates regularly, the dropdown menus can get crowded with unused templates.

**Solution:** Archive inactive templates to hide them from the main dropdown menus. Archived templates are stored in `templates/.archived/` and can be easily restored when needed.

**How to Use:**
- In the GUI, go to **"5) Template manager"** page
- Click the **"📦 Archive"** button on any template to archive it
- Archived templates appear in a separate section at the bottom
- Click **"↩️ Restore to Active"** to bring them back

**Where Archived Templates Appear:**
- They're hidden from the main dropdown menus on Quick Grade, Align, and Score pages
- They're shown in an expandable "📦 Archived Templates" section below the main dropdown
- You can still select and use them if needed, but they don't clutter the main list

### 2. Custom Template Ordering

**Problem Solved:** You want your most-used templates to appear at the top of dropdown menus.

**Solution:** Reorder templates with up/down buttons and mark favorites to pin them to the top.

**How to Use:**
- In the GUI, go to **"5) Template manager"** page
- Use **"⬆️ Up"** and **"⬇️ Down"** buttons to rearrange templates
- Click **"☆ Favorite"** to pin a template (shows as **"⭐ Unfav"** when favorited)
- Favorite templates appear with a ⭐ star icon in all dropdown menus

**Ordering Rules:**
1. Favorite templates appear first (in the order you favorited them)
2. Then templates in your custom order
3. Finally, any remaining templates alphabetically

## Technical Details

### File Structure

```
templates/
├── .preferences.json          # Stores ordering and favorites
├── .archived/                 # Archived templates folder
│   └── old_template/
│       ├── master_template.pdf
│       └── bubblemap.yaml
├── active_template_1/
│   ├── master_template.pdf
│   └── bubblemap.yaml
└── active_template_2/
    ├── master_template.pdf
    └── bubblemap.yaml
```

### Preferences File Format

The `.preferences.json` file stores your customizations:

```json
{
  "order": [
    "template_id_1",
    "template_id_2",
    "template_id_3"
  ],
  "favorites": [
    "template_id_1"
  ]
}
```

- **order**: List of template IDs in your preferred display order
- **favorites**: List of template IDs marked as favorites (shown with ⭐)

### API Methods

New methods added to `TemplateManager` class:

#### Archive Management
- `archive_template(template_id)` - Move template to `.archived/`
- `unarchive_template(template_id)` - Restore template from archive
- `scan_archived_templates()` - Get list of archived templates

#### Ordering & Favorites
- `set_template_order(ordered_ids)` - Set custom order for all templates
- `move_template_up(template_id)` - Move template up one position
- `move_template_down(template_id)` - Move template down one position
- `toggle_favorite(template_id)` - Add/remove template from favorites
- `is_favorite(template_id)` - Check if template is favorited

### GUI Updates

All template selection dropdowns now show:
1. Active templates in custom order with ⭐ for favorites
2. Expandable "📦 Archived Templates" section (only if archived templates exist)

## Examples

### Example 1: Organizing for a Single Course

If you teach one course and only use one bubble sheet:

1. Archive all other templates → Clean dropdown with just your template
2. Archived templates are out of the way but still accessible if needed

### Example 2: Multiple Courses with Different Exam Lengths

If you teach 3 courses with different exam lengths:

1. Favorite your 3 main templates → They appear at the top with ⭐
2. Archive old/unused templates → Dropdown only shows what you need
3. Order your favorites by frequency of use

### Example 3: Department with Many Templates

If you're in a department with 10+ shared templates:

1. Favorite the 2-3 you use → Quick access to your templates
2. Keep others active for colleagues → They can do the same
3. Archive very old templates → Reduce clutter for everyone

## Testing

A test script is provided to verify the functionality:

```bash
python test_template_manager.py
```

This tests:
- Template scanning (active and archived)
- Favorite toggle functionality
- Template ordering (up/down)
- Preferences file creation and loading

## Migration Notes

### Backward Compatibility

- Existing templates work without modification
- If `.preferences.json` doesn't exist, templates appear alphabetically (old behavior)
- Archive folder (`.archived/`) is created automatically when needed
- No breaking changes to existing code

### Upgrading

No special upgrade steps needed. The new features are opt-in:
- Templates work as before if you don't use archive/ordering
- Start using features anytime via the Template Manager page

## Future Enhancements

Possible future additions:
- Bulk archive/restore operations
- Template groups/categories
- Import/export template preferences
- Template usage analytics
- Search/filter for large template libraries

## Support

If you encounter issues:
1. Check that `.preferences.json` is valid JSON
2. Verify archive folder permissions
3. Use "🔄 Refresh Template List" button in GUI
4. Check MarkShark logs for error messages
