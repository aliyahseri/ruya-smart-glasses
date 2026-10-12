import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

try:
    import orchestrator
    print('Success')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()