#!/bin/bash
usage() {
    echo "$0 [METHOD] [OUTPUT_PREFIX]"
    echo "All valid METHOD: bim, mim, cw"
    echo "The OUTPUT_PREFIX could be ommitted, and the results would be saved to current directory."
}

if [ -z "$1" ]; then
    usage
    exit 0
fi

if [ -z "$2" ]; then
    OUTPUT_PREFIX="$(realpath "./")/"
else
    OUTPUT_PREFIX="$(realpath "$2")"
    if [ -d "${OUTPUT_PREFIX}" ]; then
        OUTPUT_PREFIX="${OUTPUT_PREFIX}/"
    fi
fi

cd "$(dirname "$0")"

# Allow oversubscribe in MPI
export OMPI_MCA_rmaps_base_oversubscribe=yes

L_INF_EPS=$(echo "8.0 / 255.0" | bc -l)
L_INF_ALPHA=$(echo "1.0 / 255.0" | bc -l)
L_INF_SPSA_LR=${L_INF_ALPHA}

L_2_EPS="1.0"
L_2_ALPHA="0.15"
L_2_SPSA_LR="0.005"

ITERATION="100"
MAX_QUERIES="20000"

BATCH_SIZE="500"
COUNT="500"

run_bim() {
    method='bim'
    for goal in t ut; do
        echo "Running ${method} (${goal}, l_inf)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_inf \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --magnitude ${L_INF_EPS} --alpha ${L_INF_ALPHA} \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --magnitude ${L_2_EPS} --alpha ${L_2_ALPHA} \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_pgd() {
    method='pgd'
    for goal in t ut; do
        echo "Running ${method} (${goal}, l_inf)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_inf \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --magnitude ${L_INF_EPS} --alpha ${L_INF_ALPHA} \
            --rand-init-magnitude $(echo "${L_INF_EPS} / 10" | bc -l) \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --magnitude ${L_2_EPS} --alpha ${L_2_ALPHA} \
            --rand-init-magnitude $(echo "${L_2_EPS} / 10" | bc -l) \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_mim() {
    method='mim'
    for goal in t ut; do
        echo "Running ${method} (${goal}, l_inf)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_inf \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --magnitude ${L_INF_EPS} --alpha ${L_INF_ALPHA} --decay-factor 1.0 \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --magnitude ${L_2_EPS} --alpha ${L_2_ALPHA} --decay-factor 1.0 \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_cw() {
    method='cw'
    for goal in t ut tm; do
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --cs 1.0 \
            --iteration ${ITERATION} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_deepfool() {
    method='deepfool'
    goal='ut'
    echo "Running ${method} (${goal}, l_inf)..."
    python3 -m ares.benchmark.iteration_cli \
        --method ${method} \
        --goal ${goal} --distance-metric l_inf \
        --batch-size ${BATCH_SIZE} \
        --dataset cifar10 --offset 0 --count ${COUNT} \
        --overshot 0.02 \
        --iteration ${ITERATION} \
        --logger \
        ../example/cifar10/resnet56.py \
        --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
    echo "Running ${method} (${goal}, l_2)..."
    python3 -m ares.benchmark.iteration_cli \
        --method ${method} \
        --goal ${goal} --distance-metric l_2 \
        --batch-size ${BATCH_SIZE} \
        --dataset cifar10 --offset 0 --count ${COUNT} \
        --overshot 0.02 \
        --iteration ${ITERATION} \
        --logger \
        ../example/cifar10/resnet56.py \
        --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
}

run_nes() {
    method='nes'
    for goal in ut t tm; do
        echo "Running ${method} (${goal}, l_inf)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_inf \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --samples-per-draw 500 \
            --magnitude ${L_INF_EPS} --sigma 1e-3 \
            --lr ${L_INF_ALPHA} --min-lr $(echo "${L_INF_ALPHA} / 10.0" | bc -l) --lr-tuning --plateau-length 20 \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --samples-per-draw 500 \
            --magnitude ${L_2_EPS} --sigma 1e-3 \
            --lr ${L_2_ALPHA} --min-lr $(echo "${L_2_ALPHA} / 10.0" | bc -l) --lr-tuning --plateau-length 20 \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_spsa() {
    method='spsa'
    for goal in ut t tm; do
        echo "Running ${method} (${goal}, l_inf)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_inf \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --samples-per-draw 500 \
            --magnitude ${L_INF_EPS} --sigma 1e-3 --lr ${L_INF_SPSA_LR} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --samples-per-draw 500 \
            --magnitude ${L_2_EPS} --sigma 1e-3 --lr ${L_2_SPSA_LR} \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_nattack() {
    method='nattack'
    for goal in ut t tm; do
        echo "Running ${method} (${goal}, l_inf)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_inf \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --samples-per-draw 500 \
            --magnitude ${L_INF_EPS} --sigma 0.1 --lr 0.02 \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_inf.npy"
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --samples-per-draw 500 \
            --magnitude ${L_2_EPS} --sigma 0.1 --lr 0.02 \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_boundary() {
    method='boundary'
    for goal in ut t; do
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --max-directions 25 --spherical-step 1e-2 --source-step 1e-2 --step-adaptation 1.5 --maxprocs 25 \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

run_evolutionary() {
    method='evolutionary'
    for goal in ut t; do
        echo "Running ${method} (${goal}, l_2)..."
        python3 -m ares.benchmark.iteration_cli \
            --method ${method} \
            --goal ${goal} --distance-metric l_2 \
            --batch-size ${BATCH_SIZE} \
            --dataset cifar10 --offset 0 --count ${COUNT} \
            --max-queries ${MAX_QUERIES} \
            --mu 1e-2 --sigma 3e-2 --decay-factor 0.99 --c 0.001 --maxprocs 25 \
            --logger \
            ../example/cifar10/resnet56.py \
            --output "${OUTPUT_PREFIX}${method}_${goal}_l_2.npy"
    done
}

if [ "$1" == "bim" ]; then
    run_bim
    exit 0
fi

if [ "$1" == "pgd" ]; then
    run_pgd
    exit 0
fi

if [ "$1" == "mim" ]; then
    run_mim
    exit 0
fi

if [ "$1" == "cw" ]; then
    run_cw
    exit 0
fi

if [ "$1" == "deepfool" ]; then
    run_deepfool
    exit 0
fi

if [ "$1" == "nes" ]; then
    run_nes
    exit 0
fi

if [ "$1" == "spsa" ]; then
    run_spsa
    exit 0
fi

if [ "$1" == "nattack" ]; then
    run_nattack
    exit 0
fi

if [ "$1" == "boundary" ]; then
    run_boundary
    exit 0
fi

if [ "$1" == "evolutionary" ]; then
    run_evolutionary
    exit 0
fi