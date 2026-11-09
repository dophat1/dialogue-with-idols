"""Public portfolio entry point with a bundled read-only book library."""
import os
import runpy
from pathlib import Path

os.environ['DIALOGUE_DEMO_MODE'] = '1'

runpy.run_path(str(Path(__file__).with_name('discussion_app.py')), run_name='__main__')
