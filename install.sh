#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# cupel installer — https://cupel.run/install
#
# usage:
#   curl -fsSL https://cupel.run/install | bash
#
# what it does:
#   1. finds python3 >= 3.11 on your system
#   2. creates an isolated venv at ~/.cupel/venv
#   3. installs cupel from PyPI into that venv
#   4. creates a wrapper script at ~/.cupel/bin/cupel
#   5. adds ~/.cupel/bin to your PATH (via shell rc file)
#
# what it does NOT do:
#   - install python for you (see error message for hints)
#   - require sudo or root
#   - touch your project files, configs, eval sets, or results
#
# supported:
#   - macOS (arm64, x86_64)
#   - Linux (most distros with python 3.11+)
#
# not yet supported:
#   - Windows (use: pip install cupel)
#   - auto-installing python when missing or too old
#
# future plans:
#   - if python < 3.11, offer to bootstrap via `uv` which can
#     download and manage python versions independently
#   - windows powershell installer (irm cupel.run/install.ps1 | iex)
#   - version pinning: curl -fsSL https://cupel.run/install/0.2.0 | bash
#   - checksum verification of the installed package
#
# uninstall:
#   rm -rf ~/.cupel/venv ~/.cupel/bin
#   # then remove the PATH line from your shell rc
#   # your configs (~/.cupel/.env) and project data are untouched
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CUPEL_HOME="${HOME}/.cupel"
CUPEL_VENV="${CUPEL_HOME}/venv"
CUPEL_BIN="${CUPEL_HOME}/bin"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

# ── colors ──

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  DIM='\033[2m'
  BOLD='\033[1m'
  RESET='\033[0m'
else
  RED='' GREEN='' YELLOW='' DIM='' BOLD='' RESET=''
fi

info()  { echo -e "  ${GREEN}▸${RESET} $1"; }
warn()  { echo -e "  ${YELLOW}▸${RESET} $1"; }
fail()  { echo -e "\n  ${RED}✘${RESET} $1\n"; exit 1; }

# ── detect os ──

detect_os() {
  case "$(uname -s)" in
    Darwin) OS="macos" ;;
    Linux)  OS="linux" ;;
    *)      fail "unsupported OS: $(uname -s). on Windows, use: pip install cupel" ;;
  esac
}

# ── find python 3.11+ ──

find_python() {
  local candidates=("python3" "python")
  PYTHON=""

  for cmd in "${candidates[@]}"; do
    if command -v "$cmd" &>/dev/null; then
      local version
      version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
      local major minor
      major=$(echo "$version" | cut -d. -f1)
      minor=$(echo "$version" | cut -d. -f2)

      if [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
        PYTHON="$cmd"
        PYTHON_VERSION="$version"
        return
      fi
    fi
  done

  # nothing found — give os-specific help
  echo ""
  fail "python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ not found.

  $(python_install_hint)"
}

python_install_hint() {
  case "$OS" in
    macos)
      echo "install it with:

    brew install python@3.12

  or download from https://www.python.org/downloads/" ;;
    linux)
      if command -v apt &>/dev/null; then
        echo "install it with:

    sudo apt update && sudo apt install python3.12 python3.12-venv

  (you may need: sudo add-apt-repository ppa:deadsnakes/ppa)"
      elif command -v dnf &>/dev/null; then
        echo "install it with:

    sudo dnf install python3.12"
      elif command -v pacman &>/dev/null; then
        echo "install it with:

    sudo pacman -S python"
      else
        echo "install python 3.11+ from https://www.python.org/downloads/"
      fi ;;
  esac
}

# ── create or upgrade venv ──

setup_venv() {
  if [ -d "$CUPEL_VENV" ]; then
    info "existing install found — upgrading"
    "${CUPEL_VENV}/bin/pip" install --upgrade cupel --quiet || fail "upgrade failed"
  else
    info "creating venv at ${DIM}${CUPEL_VENV}${RESET}"
    "$PYTHON" -m venv "$CUPEL_VENV" || fail "failed to create venv (is python3-venv installed?)"

    info "installing cupel"
    "${CUPEL_VENV}/bin/pip" install --upgrade pip --quiet 2>/dev/null
    "${CUPEL_VENV}/bin/pip" install cupel --quiet || fail "pip install cupel failed"
  fi
}

# ── create wrapper script ──

create_wrapper() {
  mkdir -p "$CUPEL_BIN"
  cat > "${CUPEL_BIN}/cupel" << 'EOF'
#!/usr/bin/env bash
# cupel wrapper — delegates to the venv installation
exec "${HOME}/.cupel/venv/bin/cupel" "$@"
EOF
  chmod +x "${CUPEL_BIN}/cupel"
}

# ── ensure PATH ──

ensure_path() {
  local bin_path="$CUPEL_BIN"

  # already on PATH?
  if echo "$PATH" | tr ':' '\n' | grep -qx "$bin_path"; then
    return
  fi

  local line="export PATH=\"${bin_path}:\$PATH\"  # cupel"
  local shell_name rc_file

  shell_name=$(basename "${SHELL:-bash}")
  case "$shell_name" in
    zsh)  rc_file="${HOME}/.zshrc" ;;
    bash)
      # prefer .bashrc, fall back to .bash_profile on mac
      if [ -f "${HOME}/.bashrc" ]; then
        rc_file="${HOME}/.bashrc"
      else
        rc_file="${HOME}/.bash_profile"
      fi ;;
    fish)
      # fish uses a different syntax
      mkdir -p "${HOME}/.config/fish"
      rc_file="${HOME}/.config/fish/config.fish"
      line="fish_add_path ${bin_path}  # cupel" ;;
    *)    rc_file="${HOME}/.profile" ;;
  esac

  # don't add if already present
  if [ -f "$rc_file" ] && grep -q "# cupel" "$rc_file" 2>/dev/null; then
    return
  fi

  echo "" >> "$rc_file"
  echo "$line" >> "$rc_file"
  MODIFIED_RC="$rc_file"
}

# ── main ──

main() {
  echo ""
  echo -e "  ${BOLD}cupel installer${RESET}"
  echo ""

  detect_os
  find_python
  info "found python ${PYTHON_VERSION} ${DIM}($(command -v "$PYTHON"))${RESET}"

  setup_venv
  create_wrapper
  ensure_path

  # verify
  local installed_version
  installed_version=$("${CUPEL_VENV}/bin/cupel" --version 2>/dev/null | sed 's/^cupel //' || echo "unknown")

  echo ""
  echo -e "  ${GREEN}✔${RESET} cupel ${BOLD}${installed_version}${RESET} installed"
  echo ""
  echo -e "    ${DIM}start here:${RESET}   cupel"
  echo -e "    ${DIM}seed intel:${RESET}   cupel init"
  echo -e "    ${DIM}cli bench:${RESET}    cupel run"
  echo -e "    ${DIM}uninstall:${RESET}    rm -rf ~/.cupel/venv ~/.cupel/bin"
  echo -e "    ${DIM}upgrade:${RESET}      curl -fsSL https://cupel.run/install | bash"
  echo ""

  if [ -n "${MODIFIED_RC:-}" ]; then
    warn "added ${CUPEL_BIN} to PATH in ${MODIFIED_RC}"
    echo -e "    restart your shell or run: ${BOLD}source ${MODIFIED_RC}${RESET}"
    echo ""
  fi
}

main
