#!/usr/bin/env bash
# Verify the quantized magpie GGUFs against the f32 reference, per quant:
#
#   1. load   -- `magpie-cli info` must succeed;
#   2. replay -- teacher-forced replay (test_e2e_replay) against the NeMo
#      reference dump with MAGPIE_REPLAY_ATOL relaxed, recording the reported
#      logit drift max|d| (informational for quants; f32 runs the strict gate);
#   3. asr    -- synthesize a fixed sentence (seed 1234) and round-trip it
#      through parakeet.cpp ASR; the transcript is word-compared to the input
#      (case/punctuation-insensitive).
#
# Usage:
#   PARAKEET_CLI=/path/to/parakeet-cli PARAKEET_ASR_MODEL=/path/to/ctc.gguf \
#       scripts/verify_quants.sh [quant ...]
#
# Env:
#   MAGPIE_BUILD_DIR    build tree with magpie-cli + test_e2e_replay (build-q)
#   MAGPIE_REF_DUMP     reference dump (models/ref_dump_en_speaker0.gguf)
#   MAGPIE_VERIFY_OUT   output dir for wavs/logs (default: mktemp -d)
#   PARAKEET_CLI        parakeet-cli binary (ASR stage skipped when unset)
#   PARAKEET_ASR_MODEL  parakeet CTC model gguf
#
# Exit code: 0 when every requested quant loads, replays structurally
# (step/frame/code bookkeeping) and -- where ASR is enabled -- f16/q8_0/q6_k
# transcripts match the input text. q5_k/q4_k transcripts are reported but
# only failure to load/synthesize fails the script (degradation is expected).
set -u

cd "$(dirname "$0")/.."
ROOT=$PWD
BUILD=${MAGPIE_BUILD_DIR:-build-q}
REF=${MAGPIE_REF_DUMP:-models/ref_dump_en_speaker0.gguf}
OUT=${MAGPIE_VERIFY_OUT:-$(mktemp -d)}
CLI=$BUILD/examples/cli/magpie-cli
REPLAY=$BUILD/tests/test_e2e_replay
TEXT="Hello world, this is a test of the text to speech system."
SEED=1234
QUANTS=("${@:-f32 f16 q8_0 q6_k q5_k q4_k}")
[ $# -eq 0 ] && QUANTS=(f32 f16 q8_0 q6_k q5_k q4_k)

mkdir -p "$OUT"
fail=0

normalize() { # lowercase, strip punctuation, squeeze whitespace
    tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:] \n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//'
}
want=$(printf '%s' "$TEXT" | normalize)

printf '%-6s %-9s %-6s %-14s %-14s %-6s %s\n' \
    quant "size(MB)" load "step0 max|d|" "nocache max|d|" asr transcript
echo "----------------------------------------------------------------------------------------"

for q in ${QUANTS[@]}; do
    model=models/magpie-tts-multilingual-357m-$q.gguf
    if [ ! -f "$model" ]; then echo "$q: $model missing"; fail=1; continue; fi
    size=$(du -m "$model" | cut -f1)

    # -- 1. load ------------------------------------------------------------
    if "$CLI" info --model "$model" > "$OUT/info_$q.log" 2>&1; then
        load=ok
    else
        load=FAIL; fail=1
    fi

    # -- 2. teacher-forced replay drift vs the f32 reference dump ------------
    # f32 keeps the strict default gate (atol 1e-4); quants relax the logit
    # tolerance so the run reports drift while still gating the structural
    # bookkeeping (step count, kept frames, replayed codes, EOS timing).
    replay_env=(MAGPIE_MODEL="$model" MAGPIE_REF_DUMP="$REF")
    [ "$q" != f32 ] && replay_env+=(MAGPIE_REPLAY_ATOL=1e9)
    if env "${replay_env[@]}" "$REPLAY" > "$OUT/replay_$q.log" 2>&1; then :; else
        echo "  [$q] replay FAILED (see $OUT/replay_$q.log)"; fail=1
    fi
    d0=$(grep -o 'kv-cache step0\] n=[0-9]* max|d|=[0-9.e+-]*' "$OUT/replay_$q.log" \
         | grep -o 'max|d|=[0-9.e+-]*' | cut -d= -f2)
    dn=$(grep -o 'no-cache\] steps=[0-9]* max|d|=[0-9.e+-]*' "$OUT/replay_$q.log" \
         | grep -o 'max|d|=[0-9.e+-]*' | cut -d= -f2)

    # -- 3. synthesis + ASR round-trip ----------------------------------------
    wav=$OUT/hello_$q.wav
    asr=-; got="(asr skipped: PARAKEET_CLI unset)"
    if ! "$CLI" say --model "$model" --text "$TEXT" --seed $SEED \
            --output "$wav" > "$OUT/say_$q.log" 2>&1; then
        asr=FAIL; got="(synthesis failed, see $OUT/say_$q.log)"; fail=1
    elif [ -n "${PARAKEET_CLI:-}" ]; then
        raw=$("$PARAKEET_CLI" transcribe --model "$PARAKEET_ASR_MODEL" \
              --input "$wav" --decoder ctc 2> "$OUT/asr_$q.log")
        got=$(printf '%s' "$raw" | normalize)
        if [ "$got" = "$want" ]; then
            asr=match
        else
            asr=DIFF
            # transcript equality is a hard gate only for f32/f16/q8_0/q6_k
            case $q in f32|f16|q8_0|q6_k) fail=1 ;; esac
        fi
    fi

    printf '%-6s %-9s %-6s %-14s %-14s %-6s %s\n' \
        "$q" "$size" "$load" "${d0:--}" "${dn:--}" "$asr" "$got"
done

echo
echo "expected: $want"
echo "artifacts in $OUT"
exit $fail
