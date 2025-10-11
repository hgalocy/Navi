#########################################################
#Paths
#########################################################
# Llama.cpp server
LLAMA_BIN="${HOME}/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="${HOME}/models/llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_ARGS="-c 4096 -t 4 -ngl 0 --port 8080 --host 127.0.0.1"

# Mic and Speaker
# Vosk + Piper
AREC_DEVICE="${AREC_DEVICE:-default}" #"plughw:2,0"
VOSK_MODEL_DIR="/home/freezypaws/models/vosk/en-us"
ASR_SCRIPT="NaviSpeaks/navi_speaks_mvp.py"

#########################################################
#Setup
#########################################################
SESSION="navi"
LOG_DIR="${HOME}/navi_logs"
mkdir -p "${LOG_DIR}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }
}

need tmux
need bash
need "${LLAMA_BIN}"

#########################################################
#Start Navi
#########################################################
ACTIVATE="${ACTIVATE:-}"
export AREC_DEVICE ASR_SCRIPT VOSK_MODEL_DIR

# Commands to run in each tmux window (stdout/stderr go to logs)
# Run llama LLM
CMD_LLAMA="stdbuf -oL -eL \"${LLAMA_BIN}\" -m \"${LLAMA_MODEL}\" ${LLAMA_ARGS} \
  > \"${LOG_DIR}/llama.log\" 2>&1"

# CMD_PIPER="need_piper(){ command -v \"${PIPER_BIN}\" >/dev/null 2>&1 || exit 127; }; \
#   need_piper; \
#   stdbuf -oL -eL \"${PIPER_BIN}\" --server --host 127.0.0.1 --port ${PIPER_PORT} -m \"${PIPER_MODEL}\" \
#   >> \"${LOG_DIR}/piper.log\" 2>&1"

# Run piper/vosk script
CMD_ASR="$ACTIVATE stdbuf -oL -eL bash -lc 'set -Eeuo pipefail; \
  : \"\${AREC_DEVICE:?}\"; : \"\${ASR_SCRIPT:?}\"; \
  OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONWARNINGS=default \
  arecord -q -f S16_LE -c1 -r16000 -D \"\${AREC_DEVICE}\" \
    | python3 -u -X dev -X faulthandler \"\${ASR_SCRIPT}\"' \
  > \"${LOG_DIR}/asr_loop.log\" 2>&1"


# Create sessions
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n "llm"
  tmux send-keys -t "${SESSION}:llm" "${CMD_LLAMA}" C-m

  # tmux new-window -t "${SESSION}" -n "piper"
  # tmux send-keys -t "${SESSION}:piper" "${CMD_PIPER}" C-m

  tmux new-window -t "${SESSION}" -n "asr"
  tmux send-keys -t "${SESSION}:asr" "${CMD_ASR}" C-m

  # tmux new-window -t "${SESSION}" -n "monitor"
  # tmux send-keys -t "${SESSION}:monitor" "${CMD_HTOP}" C-m

  echo "Started tmux session '${SESSION}'. Logs: ${LOG_DIR}"
else
  echo "tmux session '${SESSION}' already exists. Attaching…"
fi


echo "Starting up Navi..."
echo "Turning on Navi's brain..."

echo "Starting up Navi's senses..."
echo "Enabling Navi's mouth and ears..."
# arecord -q -f S16_LE -c1 -r16000 -D plughw:2,0 -B 2000000 -F 20000 | python3 ~/sandbox/NaviSpeaks/navi_speaks_mvp.py