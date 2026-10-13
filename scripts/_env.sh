# Sourced by every script here. Settings are spelled STEERCA_*; the CARL_*
# spelling used in the papers' commands still works, and this copies any CARL_X
# that is set (and has no STEERCA_X) across before the script reads it.
for _var in $(compgen -e | grep '^CARL_' || true); do
  _new="STEERCA_${_var#CARL_}"
  if [ -z "${!_new+x}" ]; then
    export "$_new=${!_var}"
  fi
done
unset _var _new
