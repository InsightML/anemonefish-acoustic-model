#!/bin/bash
# Cleanup macOS metadata files that cause Docker build issues on NAS volumes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Cleaning up macOS metadata files in: $WORKSPACE_DIR"
echo ""

# Count files before cleanup
APPLEDOUBLE_COUNT=$(find "$WORKSPACE_DIR" -name "._*" -type f 2>/dev/null | wc -l | xargs)
DS_STORE_COUNT=$(find "$WORKSPACE_DIR" -name ".DS_Store" -type f 2>/dev/null | wc -l | xargs)

echo "Found:"
echo "  - $APPLEDOUBLE_COUNT AppleDouble files (._*)"
echo "  - $DS_STORE_COUNT .DS_Store files"
echo ""

if [ "$APPLEDOUBLE_COUNT" -eq 0 ] && [ "$DS_STORE_COUNT" -eq 0 ]; then
    echo "✓ No cleanup needed!"
    exit 0
fi

# Delete AppleDouble files (resource forks)
echo "Removing AppleDouble files..."
find "$WORKSPACE_DIR" -name "._*" -type f -delete 2>/dev/null || true

# Delete .DS_Store files
echo "Removing .DS_Store files..."
find "$WORKSPACE_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "To prevent these files from being created, you can:"
echo "1. Disable .DS_Store creation on network volumes:"
echo "   defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true"
echo ""
echo "2. Add to your .gitignore (already done):"
echo "   ._*"
echo "   .DS_Store"

