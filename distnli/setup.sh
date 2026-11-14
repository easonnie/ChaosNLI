# Source this file from Bash or Zsh to select this project's modules.
if [ -n "${BASH_VERSION:-}" ]; then
    _chaosnli_setup_file="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _chaosnli_setup_file="${(%):-%x}"
else
    echo "Please source setup.sh from Bash or Zsh." >&2
    return 1
fi

export DIR_TMP="$(cd "$(dirname "$_chaosnli_setup_file")" && pwd)"
export PYTHONPATH="$DIR_TMP/src${PYTHONPATH:+:$PYTHONPATH}"
unset _chaosnli_setup_file
