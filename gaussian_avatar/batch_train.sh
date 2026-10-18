#!/bin/bash

# Batch training script for GSA with tmux sessions
# Usage: ./batch_train.sh

# Configuration
BASE_INPUT_DIR="/robby/share/Avatar/datasets/vhap_nersemble_v2_results/export"
BASE_OUTPUT_DIR="/robby/share/Avatar/hongyul/GSAFitting"
NUM_GPUS=8
MAX_CONCURRENT_PER_GPU=10
MAX_TOTAL_SESSIONS=80
LOG_DIR="./training_logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_FILE="$LOG_DIR/training_queue.txt"
RUNNING_FILE="$LOG_DIR/running_trainings.txt"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize log files
SUCCESS_LOG="$LOG_DIR/successful_trainings.log"
FAILED_LOG="$LOG_DIR/failed_trainings.log"
SKIPPED_LOG="$LOG_DIR/skipped_trainings.log"
GENERAL_LOG="$LOG_DIR/batch_training.log"

# Clear previous logs
> "$SUCCESS_LOG"
> "$FAILED_LOG"
> "$SKIPPED_LOG"
> "$GENERAL_LOG"
> "$QUEUE_FILE"
> "$RUNNING_FILE"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$GENERAL_LOG"
}

# Function to find UNION folder for given ID
find_union_folder() {
    local id="$1"
    local id_dir="$BASE_INPUT_DIR/$id"

    if [ ! -d "$id_dir" ]; then
        return 1
    fi

    # Find UNION* folder in the ID directory
    local union_folder=$(find "$id_dir" -maxdepth 1 -type d -name "UNION*" | head -1)

    if [ -n "$union_folder" ]; then
        echo "$union_folder"
        return 0
    else
        return 1
    fi
}

# Function to check if training is actually completed by checking output file or directory
is_training_completed() {
    local id="$1"
    local output_path="$BASE_OUTPUT_DIR/$id"
    # Check if either the checkpoint file exists or the output directory exists
    [ -f "$output_path/chkpnt600000.pth" ] || [ -d "$output_path" ]
}

# Function to get next available slot (GPU with least load)
get_next_gpu_slot() {
    local min_count=$MAX_CONCURRENT_PER_GPU
    local best_gpu=-1

    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        local running_count=$(grep "^${gpu}:" "$RUNNING_FILE" 2>/dev/null | wc -l)
        if [ $running_count -lt $min_count ]; then
            min_count=$running_count
            best_gpu=$gpu
        fi
    done

    if [ $best_gpu -ge 0 ]; then
        echo $best_gpu
        return 0
    else
        return 1  # All GPU slots full
    fi
}

# Function to check and start next training if slot available
check_and_start_next() {
    # Find next training in queue that hasn't been started
    while IFS=: read -r id input_path output_path; do
        if [ -n "$id" ]; then
            # Skip if already completed
            if is_training_completed "$id"; then
                continue
            fi

            # Skip if already running
            if grep ":${id}$" "$RUNNING_FILE" >/dev/null 2>&1; then
                continue
            fi

            # Try to get an available GPU slot
            local gpu_id=$(get_next_gpu_slot)
            if [ $? -eq 0 ]; then
                log_message "Starting next training: ID $id on GPU $gpu_id"
                start_training "$id" "$gpu_id" "$input_path" "$output_path"
                return 0
            else
                # No available slots
                return 1
            fi
        fi
    done < "$QUEUE_FILE"

    return 1  # No more trainings to start
}

# Function to add training to running list (with file locking)
add_to_running() {
    local gpu_id="$1"
    local id="$2"
    (
        flock -x 200
        echo "${gpu_id}:${id}" >> "$RUNNING_FILE"
    ) 200>"$RUNNING_FILE.lock"
}

# Function to remove training from running list (with file locking)
remove_from_running() {
    local gpu_id="$1"
    local id="$2"
    (
        flock -x 200
        grep -v "^${gpu_id}:${id}$" "$RUNNING_FILE" > "${RUNNING_FILE}.tmp" 2>/dev/null || touch "${RUNNING_FILE}.tmp"
        mv "${RUNNING_FILE}.tmp" "$RUNNING_FILE"
    ) 200>"$RUNNING_FILE.lock"
}

# Function to create tmux session and run training
start_training() {
    local id="$1"
    local gpu_id="$2"
    local input_path="$3"
    local output_path="$4"

    local session_name="gsa_train_${id}"

    # Create output directory if it doesn't exist
    mkdir -p "$output_path"

    # Create tmux session
    tmux new-session -d -s "$session_name"

    # Set up the training command
    local train_cmd="cd '/robby/share/Avatar/hongyul/code/GSA' && CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py -s '$input_path' -m '$output_path' --eval --white_background --bind_to_mesh --use_rgb"

    # Create a wrapper script that handles logging and cleanup
    local wrapper_script="$LOG_DIR/wrapper_${id}.sh"
    cat > "$wrapper_script" << EOF
#!/bin/bash
# Set absolute paths to avoid directory issues
SCRIPT_DIR="$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/training_logs"
SUCCESS_LOG="$SCRIPT_DIR/training_logs/successful_trainings.log"
FAILED_LOG="$SCRIPT_DIR/training_logs/failed_trainings.log"
GENERAL_LOG="$SCRIPT_DIR/training_logs/batch_training.log"

echo "Starting training for ID: $id on GPU: $gpu_id" | tee -a "\$GENERAL_LOG"
echo "Input: $input_path" | tee -a "\$GENERAL_LOG"
echo "Output: $output_path" | tee -a "\$GENERAL_LOG"
echo "Command: $train_cmd" | tee -a "\$GENERAL_LOG"
echo "---" | tee -a "\$GENERAL_LOG"

# Run the training command
if $train_cmd 2>&1 | tee "\$LOG_DIR/training_${id}.log"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: ID $id completed successfully" | tee -a "\$SUCCESS_LOG" "\$GENERAL_LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ID $id failed with exit code \$?" | tee -a "\$FAILED_LOG" "\$GENERAL_LOG"
fi

# Remove from running list when training finishes (with file locking)
(
    flock -x 200
    grep -v "^${gpu_id}:${id}$" "\$LOG_DIR/running_trainings.txt" > "\$LOG_DIR/running_trainings.txt.tmp" 2>/dev/null || touch "\$LOG_DIR/running_trainings.txt.tmp"
    mv "\$LOG_DIR/running_trainings.txt.tmp" "\$LOG_DIR/running_trainings.txt"
) 200>"\$LOG_DIR/running_trainings.txt.lock"

echo "Training for ID $id finished. You can exit this tmux session."
echo "Press Enter to continue..."
read
EOF

    chmod +x "$wrapper_script"

    # Send the wrapper script command to tmux session
    tmux send-keys -t "$session_name" "bash '$wrapper_script'" Enter

    # Add to running list
    add_to_running "$gpu_id" "$id"

    log_message "Started training for ID: $id on GPU: $gpu_id in tmux session: $session_name"
}

# Main execution
log_message "Starting batch training script"
log_message "Base input directory: $BASE_INPUT_DIR"
log_message "Base output directory: $BASE_OUTPUT_DIR"
log_message "Number of GPUs: $NUM_GPUS"
log_message "Log directory: $LOG_DIR"

# IDs to process
IDS=(057 078 100 123 145 177 196 220 242 262 284 307 326 382 419 440 468 490 538
     058 079 102 124 146 178 197 221 243 263 285 308 327 383 421 441 469 491 539
     059 080 104 126 147 179 198 222 244 264 286 309 328 384 422 442 470 492
     060 081 105 127 148 180 199 223 245 266 287 310 330 385 424 447 471 494
     061 082 106 128 150 181 200 224 246 267 288 311 331 386 425 448 472 495
     063 083 107 129 151 182 201 225 247 269 289 312 332 388 426 449 473 496
     064 085 108 130 162 183 202 229 248 270 290 313 333 403 427 450 474 497
     065 088 109 132 163 184 204 230 249 271 293 314 350 405 428 451 475 498
     067 089 110 133 164 185 206 231 250 272 294 315 368 406 429 453 476 499
     068 090 111 134 165 186 207 232 251 274 295 316 370 407 430 455 478 500
     069 091 112 135 166 187 209 233 252 275 297 317 371 409 431 456 479 505
     070 092 113 136 167 188 210 234 253 276 298 318 372 410 432 457 480 509
     071 093 114 138 168 189 211 235 254 277 299 319 373 411 433 460 481 510
     072 094 115 139 170 190 212 236 256 278 301 320 374 412 434 462 483 515
     073 095 116 140 171 191 214 237 257 279 302 321 376 413 435 463 484 516
     074 096 117 141 173 192 215 238 258 280 303 322 377 415 436 464 485 521
     075 097 120 142 174 193 216 239 259 281 304 323 378 416 437 465 487 522
     076 098 121 143 175 194 217 240 260 282 305 324 379 417 438 466 488 525
     077 099 122 144 176 195 218 241 261 283 306 325 380 418 439 467 489 532)

total_ids=${#IDS[@]}
processed_count=0
successful_count=0
skipped_count=0

log_message "Total IDs to process: $total_ids"

# First, build the queue of valid training tasks
log_message "Building training queue..."
for id in "${IDS[@]}"; do
    # Check if training is already completed
    if is_training_completed "$id"; then
        log_message "SKIPPED: ID $id already completed (chkpnt600000.pth exists)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPED: ID $id - Already completed (chkpnt600000.pth exists)" >> "$SKIPPED_LOG"
        skipped_count=$((skipped_count + 1))
        processed_count=$((processed_count + 1))
        continue
    fi

    union_folder=$(find_union_folder "$id")
    if [ $? -eq 0 ]; then
        input_path="$union_folder"
        output_path="$BASE_OUTPUT_DIR/$id"
        echo "${id}:${input_path}:${output_path}" >> "$QUEUE_FILE"
        log_message "Added to queue: ID $id"
    else
        log_message "SKIPPED: No UNION folder found for ID: $id"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPED: ID $id - No UNION folder found" >> "$SKIPPED_LOG"
        skipped_count=$((skipped_count + 1))
    fi
    processed_count=$((processed_count + 1))
done

total_queue_size=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo 0)
log_message "Queue built: $total_queue_size valid trainings, $skipped_count skipped"

# Start batch of trainings (up to MAX_TOTAL_SESSIONS)
log_message "Starting batch of trainings (max: $MAX_TOTAL_SESSIONS)..."
started_count=0

while [ $started_count -lt $MAX_TOTAL_SESSIONS ]; do
    if check_and_start_next; then
        started_count=$((started_count + 1))
        log_message "Started training $started_count/$MAX_TOTAL_SESSIONS"
        sleep 2  # Small delay between starts
    else
        log_message "No more trainings available to start"
        break
    fi
done

log_message "Batch training script completed"
log_message "Total processed: $processed_count"
log_message "Training sessions started: $started_count"
log_message "Skipped: $skipped_count"
log_message "Queue size: $total_queue_size"
log_message ""
if [ $started_count -eq $MAX_TOTAL_SESSIONS ] && [ $total_queue_size -gt $started_count ]; then
    log_message "Reached maximum sessions limit ($MAX_TOTAL_SESSIONS)"
    log_message "Run the script again to start the next batch of trainings"
elif [ $started_count -lt $total_queue_size ]; then
    remaining=$((total_queue_size - started_count))
    log_message "Started $started_count/$total_queue_size trainings"
    log_message "Run the script again to start more trainings ($remaining remaining)"
else
    log_message "All available trainings have been started"
fi
log_message ""
log_message "To monitor training sessions:"
log_message "  - Check status: ./monitor_training.sh status"
log_message "  - List all sessions: tmux list-sessions"
log_message "  - Attach to session: tmux attach-session -t gsa_train_XXX"
log_message "  - Kill session: tmux kill-session -t gsa_train_XXX"
log_message ""
log_message "Log files:"
log_message "  - General log: $GENERAL_LOG"
log_message "  - Successful trainings: $SUCCESS_LOG"
log_message "  - Failed trainings: $FAILED_LOG"
log_message "  - Skipped trainings: $SKIPPED_LOG"
log_message "  - Individual training logs: $LOG_DIR/training_XXX.log"
