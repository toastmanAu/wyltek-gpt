#!/usr/bin/env bash
# wyltek-gpt installer — non-destructive smart install
# ──────────────────────────────────────────────────────────────────
# Detects what you already have, prompts before installing what's missing,
# never removes anything. Safe to re-run any time. Tested on Ubuntu 22.04+.
# Best-effort support for Debian, Fedora, Arch, macOS via Homebrew.
#
# Usage:
#   ./install.sh           # interactive — asks before each install
#   ./install.sh --yes     # install all missing (still asks for sudo)
#   ./install.sh --check   # audit only, install nothing
# ──────────────────────────────────────────────────────────────────

set -uo pipefail

YES=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --check)  CHECK_ONLY=1 ;;
    --help|-h)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg"; exit 1 ;;
  esac
done

# ─── colors / spinners (terminal aesthetic) ────────────────────────
if [ -t 1 ]; then
  C_DIM=$'\033[2m'; C_RED=$'\033[31m'; C_GRN=$'\033[32m'
  C_YLW=$'\033[33m'; C_CYN=$'\033[36m'; C_RST=$'\033[0m'
else
  C_DIM=""; C_RED=""; C_GRN=""; C_YLW=""; C_CYN=""; C_RST=""
fi
ok()    { echo "  ${C_GRN}✓${C_RST} $*"; }
miss()  { echo "  ${C_DIM}✗${C_RST} $*"; }
info()  { echo "${C_CYN}»${C_RST} $*"; }
warn()  { echo "${C_YLW}!${C_RST} $*"; }
fail()  { echo "${C_RED}✗${C_RST} $*"; }

# ─── package manager detection ─────────────────────────────────────
PKG=""
if command -v apt-get >/dev/null 2>&1; then PKG=apt
elif command -v dnf >/dev/null 2>&1;     then PKG=dnf
elif command -v pacman >/dev/null 2>&1;  then PKG=pacman
elif command -v brew >/dev/null 2>&1;    then PKG=brew
fi
[ -z "$PKG" ] && fail "no supported package manager found (apt/dnf/pacman/brew)" && exit 1

info "detected package manager: $PKG"

# Map of binary → package name per pkg manager. Format: bin|apt|dnf|pacman|brew
# Empty cell means "no known package, will skip". Multi-package values use ' '.
PKGMAP=$(cat <<'MAP'
ffmpeg|ffmpeg|ffmpeg|ffmpeg|ffmpeg
pandoc|pandoc|pandoc|pandoc|pandoc
libreoffice|libreoffice|libreoffice|libreoffice|libreoffice
weasyprint|weasyprint|weasyprint|weasyprint|weasyprint
ocrmypdf|ocrmypdf|ocrmypdf|ocrmypdf|ocrmypdf
tesseract|tesseract-ocr|tesseract|tesseract|tesseract
ebook-convert|calibre|calibre|calibre|calibre
pdftotext|poppler-utils|poppler-utils|poppler|poppler
vips|libvips-tools|vips-tools|libvips|vips
cwebp|webp|libwebp-tools|libwebp|webp
rsvg-convert|librsvg2-bin|librsvg2-tools|librsvg|librsvg
gif2webp|webp|libwebp-tools|libwebp|webp
potrace|potrace|potrace|potrace|potrace
sox|sox|sox|sox|sox
exiftool|libimage-exiftool-perl|perl-Image-ExifTool|perl-image-exiftool|exiftool
pngquant|pngquant|pngquant|pngquant|pngquant
xmlstarlet|xmlstarlet|xmlstarlet|xmlstarlet|xmlstarlet
mlr|miller|miller|miller|miller
sqlite3|sqlite3|sqlite|sqlite|sqlite
jq|jq|jq|jq|jq
unzip|unzip|unzip|unzip|unzip
MAP
)

PIPX_TOOLS="markitdown ocrmypdf weasyprint csvkit"

pkg_for() {
  local bin="$1"
  echo "$PKGMAP" | awk -F'|' -v b="$bin" -v p="$PKG" '
    $1 == b {
      if (p == "apt")    print $2
      if (p == "dnf")    print $3
      if (p == "pacman") print $4
      if (p == "brew")   print $5
    }'
}

install_pkg() {
  local pkgs="$1"
  case "$PKG" in
    apt)    sudo apt-get install -y $pkgs ;;
    dnf)    sudo dnf install -y $pkgs ;;
    pacman) sudo pacman -S --needed --noconfirm $pkgs ;;
    brew)   brew install $pkgs ;;
  esac
}

prompt_yn() {
  local q="$1"
  [ "$YES" = "1" ] && return 0
  read -r -p "$q [Y/n] " ans </dev/tty
  case "${ans,,}" in n|no) return 1 ;; *) return 0 ;; esac
}

# ─── audit ─────────────────────────────────────────────────────────
declare -A INSTALLED MISSING
TOOLS=(
  # required
  python3 ffmpeg
  # documents
  pandoc libreoffice weasyprint ocrmypdf pdftotext ebook-convert
  # images
  vips cwebp dwebp gif2webp rsvg-convert potrace pngquant exiftool
  # audio
  sox
  # data
  jq xmlstarlet mlr sqlite3
  # archives
  unzip
)

info "auditing installed tools..."
echo
for t in "${TOOLS[@]}"; do
  if command -v "$t" >/dev/null 2>&1; then
    INSTALLED[$t]=1
    ok "$t"
  else
    MISSING[$t]=1
    miss "$t"
  fi
done

echo
info "auditing pipx tools..."
echo
PIPX_AVAILABLE=$(command -v pipx >/dev/null && echo yes || echo no)
[ "$PIPX_AVAILABLE" = "yes" ] && ok "pipx" || miss "pipx"
declare -A PIPX_MISSING
for t in $PIPX_TOOLS; do
  if command -v "$t" >/dev/null 2>&1; then
    ok "$t (pipx)"
  else
    PIPX_MISSING[$t]=1
    miss "$t (pipx)"
  fi
done

echo
info "auditing extras..."
echo
DUCKDB_OK=0; command -v duckdb >/dev/null && { ok duckdb; DUCKDB_OK=1; } || miss duckdb
YQ_OK=0;     command -v yq >/dev/null     && { ok yq;     YQ_OK=1; } || miss yq
WHISPER_OK=0;command -v whisper >/dev/null && { ok whisper; WHISPER_OK=1; } || miss "whisper (optional, audio transcription)"

if [ "$CHECK_ONLY" = "1" ]; then
  echo
  info "audit only — no changes made"
  exit 0
fi

# ─── install missing ───────────────────────────────────────────────
echo
[ ${#MISSING[@]} -eq 0 ] && [ "$PIPX_AVAILABLE" = "yes" ] && [ ${#PIPX_MISSING[@]} -eq 0 ] && \
  [ "$DUCKDB_OK" = "1" ] && [ "$YQ_OK" = "1" ] && {
    info "everything's already installed."
    exit 0
  }

if [ ${#MISSING[@]} -gt 0 ]; then
  TO_INSTALL=""
  for bin in "${!MISSING[@]}"; do
    p=$(pkg_for "$bin")
    [ -n "$p" ] && TO_INSTALL="$TO_INSTALL $p"
  done
  TO_INSTALL=$(echo "$TO_INSTALL" | tr ' ' '\n' | sort -u | xargs)
  if [ -n "$TO_INSTALL" ]; then
    info "system packages to install: $TO_INSTALL"
    if prompt_yn "install via $PKG?"; then
      install_pkg "$TO_INSTALL"
    fi
  fi
fi

if [ "$PIPX_AVAILABLE" = "no" ]; then
  if prompt_yn "install pipx (needed for python-based converters)?"; then
    case "$PKG" in
      apt)    sudo apt-get install -y pipx && pipx ensurepath ;;
      dnf)    sudo dnf install -y pipx && pipx ensurepath ;;
      pacman) sudo pacman -S --needed --noconfirm python-pipx && pipx ensurepath ;;
      brew)   brew install pipx && pipx ensurepath ;;
    esac
    PIPX_AVAILABLE=yes
  fi
fi

if [ "$PIPX_AVAILABLE" = "yes" ] && [ ${#PIPX_MISSING[@]} -gt 0 ]; then
  for t in "${!PIPX_MISSING[@]}"; do
    if prompt_yn "install $t via pipx?"; then
      pipx install "$t" || warn "failed: $t"
    fi
  done
fi

if [ "$DUCKDB_OK" = "0" ] && prompt_yn "install duckdb (data swiss-army CLI)?"; then
  ARCH=$(uname -m)
  case "$ARCH" in x86_64) DUCK_ARCH=amd64 ;; aarch64|arm64) DUCK_ARCH=aarch64 ;; *) DUCK_ARCH="" ;; esac
  if [ -n "$DUCK_ARCH" ]; then
    URL="https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-${DUCK_ARCH}.zip"
    [ "$(uname -s)" = "Darwin" ] && URL="https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-osx-universal.zip"
    TMP=$(mktemp -d)
    curl -L "$URL" -o "$TMP/duckdb.zip" && sudo unzip -o "$TMP/duckdb.zip" -d /usr/local/bin/
    rm -rf "$TMP"
  else
    warn "unknown arch $ARCH — install duckdb manually from https://duckdb.org/docs/installation/"
  fi
fi

if [ "$YQ_OK" = "0" ] && prompt_yn "install yq (yaml/json/xml/toml interconverter)?"; then
  case "$PKG" in
    apt)    command -v snap >/dev/null && sudo snap install yq || warn "snap not available; install yq manually from https://github.com/mikefarah/yq/releases" ;;
    dnf)    sudo dnf install -y yq ;;
    pacman) sudo pacman -S --needed --noconfirm go-yq ;;
    brew)   brew install yq ;;
  esac
fi

if [ "$WHISPER_OK" = "0" ] && [ "$PIPX_AVAILABLE" = "yes" ]; then
  if prompt_yn "install openai-whisper (audio → text, optional but powerful)?"; then
    pipx install openai-whisper || warn "failed: openai-whisper (large download, may need --pip-args='--prefer-binary')"
  fi
fi

# ─── Python venv for the app itself ────────────────────────────────
echo
if prompt_yn "set up the python venv and install fastapi/uvicorn?"; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
  ok "venv ready — start the app with:  ./.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000"
fi

# ─── ollama check ──────────────────────────────────────────────────
echo
if ! command -v ollama >/dev/null 2>&1; then
  warn "ollama is not installed — wyltek-gpt is a frontend for it."
  warn "install from https://ollama.com/download then run: ollama pull qwen2.5:7b"
else
  ok "ollama is installed"
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "ollama is reachable at http://localhost:11434"
  else
    warn "ollama is installed but not running — start it with: ollama serve"
  fi
fi

echo
info "done. re-run anytime: ./install.sh --check"
