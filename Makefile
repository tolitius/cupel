PYTHON ?= python3.11
PORT   ?= 8042

.PHONY: run bump build deploy

run:
	$(PYTHON) -m uvicorn cupel.server:app --reload --port $(PORT)

# increment the patch version in pyproject.toml (e.g. 0.1.89 => 0.1.90)
bump:
	@$(PYTHON) -c "import re, pathlib; p = pathlib.Path('pyproject.toml'); s = p.read_text(); m = re.search(r'^version = \"(\\d+)\\.(\\d+)\\.(\\d+)\"', s, re.M); v = f'{m[1]}.{m[2]}.{int(m[3]) + 1}'; p.write_text(s[:m.start()] + f'version = \"{v}\"' + s[m.end():]); print(f'version {m[1]}.{m[2]}.{m[3]} => {v}')"

build:
	rm -rf dist/
	$(PYTHON) -m build

deploy: bump build
	$(PYTHON) -m twine upload --verbose dist/*
