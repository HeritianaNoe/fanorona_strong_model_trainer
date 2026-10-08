from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
files = list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
for f in files:
    py_compile.compile(str(f), doraise=True)
print(f'SOURCE CHECK PASSED: {len(files)} Python files compiled.')
